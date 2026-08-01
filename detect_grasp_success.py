"""放置前抓取确认（TCP 外力判抓）。

从主流程 real_bottle_pick_place_interactive3_point_completion_with_yolo2_dual.py 抽出，
集中管理「机器人到底有没有把物体抓在手上」这一判定，主文件只需三行调用。

判定原理
--------
夹爪闭合并把物体抬离桌面后，物体重力完全由末端承担，UR 的 `getActualTCPForce()`
（已扣除机器人自身模型重力）会读到一个稳定的外力分量；空抓时该外力接近零
（仅夹爪自重残差与噪声）。因此取一段窗口内外力幅值的【均值】与阈值比较即可判抓。

时序（与运动并行，主线程零等待）
--------------------------------
  ① 抓取后竖直抬升到抓取高位（transfer_role==1）执行完 → start_monitor() 起后台线程，
     固定采样 window_s 秒；
  ② 关节1摆动到放置点正上方 —— 采样与该运动并行进行；
  ③ 到放置点正上方、下落前（transfer_role==3）→ confirm_grasp() 取结果判定。
     摆动一般比 window_s 长，线程早已自行结束，取结果零等待；摆动更快时才补等剩余窗口，
     保证每次判抓的采样时长恒为 window_s，统计口径一致、阈值可稳定标定。

线程安全
--------
采样走 rtde_robot.get_tcp_force() → 底层 `_rtde_r.getActualTCPForce()`，是只读的
RTDE receive 接口，与主线程运动控制用的 `_rtde_c` 接口分离，跨线程读取安全。

可调参数（constants.py 的 REAL_PIPELINE_CONFIG）
-----------------------------------------------
  place_grip_force_check_enabled : 总开关，False 时 confirm_grasp() 恒返回 True（照常放置）
  place_grip_force_threshold     : 外力幅值阈值 (N)，均值 ≥ 阈值判定已抓到
  place_grip_force_window_s      : 采样窗口 (s)，后台采样与回退现场采样共用
  place_grip_force_rate_hz       : 采样频率 (Hz)
"""
from __future__ import annotations

import threading
from time import time, sleep
from typing import Optional, Tuple

import numpy as np

from yanjiuyuan.constants import REAL_PIPELINE_CONFIG

LOG = "[real_pipeline]"

__all__ = [
    "TcpForceMonitor",
    "sample_tcp_force_mean",
    "start_monitor",
    "confirm_grasp",
    "stop_monitor",
    "abort_place",
]


# --------------------------------------------------------------------------- #
# 配置读取
# --------------------------------------------------------------------------- #
def _cfg_enabled() -> bool:
    return bool(REAL_PIPELINE_CONFIG.get("place_grip_force_check_enabled", True))


def _cfg_threshold() -> float:
    return float(REAL_PIPELINE_CONFIG.get("place_grip_force_threshold", 10.0))


def _cfg_window_s() -> float:
    return float(REAL_PIPELINE_CONFIG.get("place_grip_force_window_s", 1.0))


def _cfg_rate_hz() -> float:
    return float(REAL_PIPELINE_CONFIG.get("place_grip_force_rate_hz", 20.0))


def _force_magnitude(rtde_robot) -> float:
    """读取一帧 TCP 外力并返回幅值 (N)。get_tcp_force() 返回 6D (Fx,Fy,Fz,Tx,Ty,Tz)。"""
    f = np.asarray(rtde_robot.get_tcp_force(), dtype=float).reshape(-1)
    return float(np.linalg.norm(f))


def _supported(rtde_robot) -> bool:
    return rtde_robot is not None and hasattr(rtde_robot, "get_tcp_force")


# --------------------------------------------------------------------------- #
# 后台固定窗口采样
# --------------------------------------------------------------------------- #
class TcpForceMonitor:
    """后台线程采样 TCP 外力幅值【固定 window_s 秒】，主线程一次性取【均值】判抓。

    用法::

        mon = TcpForceMonitor(rtde_robot, rate_hz=20.0, window_s=1.0)
        mon.start()                                  # 抬升到抓取高位后立即启动
        ...                                          # 机器人继续摆动到放置点上方（采样并行）
        mean, n, fmin, fmax = mon.stop_and_result()  # 到放置点上方取结果（通常零等待）
    """

    def __init__(self, rtde_robot, rate_hz: float, window_s: float):
        self._robot = rtde_robot
        self._dt = 1.0 / max(float(rate_hz), 1.0)
        self._window_s = float(window_s)
        self._t_start = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._total = 0.0
        self._count = 0
        self._min = float("inf")
        self._max = 0.0

    @property
    def window_s(self) -> float:
        return self._window_s

    def _run(self) -> None:
        t0 = self._t_start or time()
        warned = False
        # 固定窗口：跑满 window_s 自行退出（_stop 仅供异常路径提前终止）
        while not self._stop.is_set() and (time() - t0) < self._window_s:
            try:
                mag = _force_magnitude(self._robot)
                with self._lock:
                    self._total += mag
                    self._count += 1
                    if mag < self._min:
                        self._min = mag
                    if mag > self._max:
                        self._max = mag
            except Exception as exc:
                if not warned:  # 只报一次，避免高频刷屏
                    print(f"{LOG} ⚠️ TCP 外力后台采样异常（后续同类异常不再重复打印）: {exc}")
                    warned = True
            sleep(self._dt)

    def start(self) -> bool:
        """启动后台采样线程（采样固定 window_s 秒后自行结束）。返回是否启动成功。"""
        if self._thread is not None:
            return True
        try:
            self._t_start = time()
            self._thread = threading.Thread(target=self._run, name="tcp_force_monitor", daemon=True)
            self._thread.start()
            return True
        except Exception as exc:
            print(f"{LOG} ⚠️ TCP 外力后台采样线程启动失败: {exc}")
            self._thread = None
            return False

    def stop_and_result(self) -> Tuple[float, int, float, float]:
        """等固定窗口跑满后返回 (均值N, 样本数, 最小值N, 最大值N)。无有效样本时均为 0。

        摆动耗时 > window_s 时线程已结束，join 立即返回（零等待）；
        主线程提前到达时最多补等剩余窗口，保证采样时长恒为 window_s。
        """
        if self._thread is not None:
            remain = self._window_s - (time() - self._t_start) if self._t_start else 0.0
            try:
                self._thread.join(timeout=max(0.0, remain) + 0.2)
            except Exception:
                pass
            self._stop.set()  # 兜底：join 超时仍未退出时强制停止
            self._thread = None
        with self._lock:
            n = self._count
            mean = (self._total / n) if n > 0 else 0.0
            fmin = self._min if n > 0 else 0.0
            fmax = self._max if n > 0 else 0.0
        return mean, n, fmin, fmax

    def stop(self) -> None:
        """静默停止（异常清理用），不取结果。"""
        self._stop.set()
        self._thread = None


# --------------------------------------------------------------------------- #
# 阻塞采样（回退用）
# --------------------------------------------------------------------------- #
def _blocking_sample(
    rtde_robot,
    window_s: float,
    rate_hz: float,
    stop_event: Optional[threading.Event] = None,
) -> Tuple[float, int]:
    """阻塞采样，返回 (外力幅值均值N, 有效样本数)。样本数为 0 表示一帧都没读到。"""
    dt = 1.0 / max(float(rate_hz), 1.0)
    t0 = time()
    total = 0.0
    count = 0
    warned = False
    while (stop_event is None or not stop_event.is_set()) and (time() - t0) < float(window_s):
        try:
            total += _force_magnitude(rtde_robot)
            count += 1
        except Exception as exc:
            if not warned:  # 只报一次，避免高频刷屏
                print(f"{LOG} ⚠️ get_tcp_force 采样异常（后续同类异常不再重复打印）: {exc}")
                warned = True
        sleep(dt)
    return ((total / count) if count > 0 else 0.0), count


def sample_tcp_force_mean(
    rtde_robot,
    window_s: float,
    rate_hz: float,
    stop_event: Optional[threading.Event] = None,
) -> float:
    """在 window_s 秒内以 rate_hz 频率阻塞采样，返回外力幅值（norm）的【平均值】，单位 N。

    仅在后台监控未启动 / 无有效样本时作为兜底使用（会阻塞机器人线程）。
    """
    return _blocking_sample(rtde_robot, window_s, rate_hz, stop_event)[0]


# --------------------------------------------------------------------------- #
# 主流程对外接口
# --------------------------------------------------------------------------- #
def start_monitor(rtde_robot) -> Optional[TcpForceMonitor]:
    """在【抬升到抓取高位】后调用：按配置起后台采样线程。

    未启用 / 机器人不支持 get_tcp_force / 线程启动失败时返回 None（后续 confirm_grasp
    会自动回退为现场阻塞采样），调用方无需再做条件判断。
    """
    if not _cfg_enabled() or not _supported(rtde_robot):
        return None
    window_s = _cfg_window_s()
    mon = TcpForceMonitor(rtde_robot, rate_hz=_cfg_rate_hz(), window_s=window_s)
    if not mon.start():
        return None
    print(f"{LOG} 🧪 已抬升到抓取高位，启动后台 TCP 外力采样 {window_s:.2f}s（与摆动到放置点上方并行）")
    return mon


def confirm_grasp(rtde_robot, monitor: Optional[TcpForceMonitor] = None) -> bool:
    """在【放置点正上方、下落前】调用：返回是否确实抓到了物体。

    优先取 monitor 的后台采样结果（零等待）；无监控 / 无有效样本时回退现场阻塞采样。
    检测关闭或机器人不支持时返回 True（fail-open，照常放置）。
    调用方拿到 False 后应中止放置（见 abort_place），不要执行下放/松手/离开。
    """
    if not _cfg_enabled() or not _supported(rtde_robot):
        return True

    threshold = _cfg_threshold()
    mean, n, fmin, fmax = (0.0, 0, 0.0, 0.0)
    if monitor is not None:
        mean, n, fmin, fmax = monitor.stop_and_result()

    if n > 0:
        source = f"后台采样 n={n}, min={fmin:.2f}N, max={fmax:.2f}N"
    else:
        window_s = _cfg_window_s()
        print(f"{LOG} ⚠️ 后台外力采样无有效样本，回退现场阻塞采样 {window_s:.2f}s")
        mean, n = _blocking_sample(rtde_robot, window_s, _cfg_rate_hz())
        source = f"现场采样 {window_s:.2f}s, n={n}"

    if n <= 0:
        # 连回退采样也一帧没读到（RTDE 通信故障）→ 无法判定。此时【不能】按空抓处理：
        # 若手里其实握着物体，中止放置会 open_gripper 把物体扔在半空。
        # 故 fail-open：当作已抓到，照常走完放置流程（真空抓时也只是空放一次，无害）。
        print(f"{LOG} ⚠️ TCP 外力一帧都未读到，无法判抓 → fail-open：按【已抓到】处理，照常放置。")
        return True

    has_object = mean >= threshold
    print(
        f"{LOG} 🔍 放置前抓取确认：TCP 外力均值={mean:.2f}N "
        f"(阈值={threshold:.2f}N, {source}) → "
        f"{'✅ 已抓到物体，继续下放' if has_object else '⚠️ 未抓到物体（空抓）'}"
    )
    return has_object


def abort_place(rtde_robot) -> None:
    """判定空抓后的收尾：打印说明并松开夹爪。调用方随后 break 出 segment 循环即可。"""
    print(f"{LOG} ⚠️ 未抓到物体，中止放置：不执行下放/松手/离开，"
          f"直接回移拍照点→抓取起点进入下一轮抓取。")
    try:
        rtde_robot.open_gripper()
    except Exception as exc:
        print(f"{LOG} ⚠️ 中止放置时松手失败: {exc}")


def stop_monitor(monitor: Optional[TcpForceMonitor]) -> None:
    """兜底清理：异常/提前 break 时静默停掉后台采样线程，避免线程残留。"""
    if monitor is None:
        return
    try:
        monitor.stop()
    except Exception:
        pass
