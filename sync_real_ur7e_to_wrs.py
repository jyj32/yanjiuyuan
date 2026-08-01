"""
Synchronize the real UR7e + DH76 gripper state to a WRS UR7e visualization.

Default real-robot usage:
    python yanjiuyuan/sync_real_ur7e_to_wrs.py

Use a simulated signal to test the WRS window without RTDE:
    python yanjiuyuan/sync_real_ur7e_to_wrs.py --mock
"""

from __future__ import annotations

import argparse
import atexit
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


DEFAULT_HOME_CONF = np.array([
    math.pi / 2.0,
    -math.pi / 2.0,
    math.pi / 2.0,
    -math.pi,
    -math.pi / 2.0,
    0.0,
])

DH76_JAW_MIN = 0.042
DH76_JAW_MAX = 0.118


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a WRS UR7e DH76 robot and synchronize it with the real UR7e via RTDE."
    )
    parser.add_argument("--robot-ip", default="192.168.125.30", help="UR controller IP address.")
    parser.add_argument("--gp-port", default="COM4", help="DH76 gripper serial port passed to UR7EDH76_RTDE.")
    parser.add_argument("--period", type=float, default=0.05, help="Visualization update period in seconds.")
    parser.add_argument("--mock", action="store_true", help="Use a local moving pose instead of connecting to RTDE.")
    parser.add_argument("--no-env", action="store_true", help="Do not draw the reference table/camera/box models.")
    parser.add_argument("--show-obstacle-collision", action="store_true", help="Show obstacle collision primitives.")
    parser.add_argument("--show-joint-frames", action="store_true", help="Draw joint frames on the robot.")
    parser.add_argument("--hide-tcp-frame", action="store_true", help="Hide the TCP frame.")
    parser.add_argument("--hide-flange-frame", action="store_true", help="Hide the flange frame.")
    parser.add_argument("--alpha", type=float, default=0.85, help="Robot mesh alpha.")
    parser.add_argument("--print-interval", type=float, default=1.0, help="Console status print interval in seconds.")
    parser.add_argument("--max-failures", type=int, default=20, help="Stop updating after this many consecutive failures.")
    parser.add_argument("--once", action="store_true", help="Read and draw one pose, then keep the WRS window open.")
    return parser.parse_args()


class RealUR7ERTDEState:

    def __init__(self, robot_ip: str = None, gp_port: str = None, robot_x=None):
        from wrs.robot_con.ur.ur7e_dh76_rtde import UR7EDH76_RTDE

        # robot_x 由调用方传入已存在的实例（通常是进程级单例 UR7EDH76_RTDE）。
        # 此时本对象只是“借用”，close() 不得断开共享连接，否则会切断主线程/其它调用方的会话。
        if robot_x is not None:
            self._robot_x = robot_x
            self._borrowed = True
        else:
            self._robot_x = UR7EDH76_RTDE(robot_ip=robot_ip, gp_port=gp_port)
            self._borrowed = False
        self._closed = False

    def get_jnt_values(self) -> np.ndarray:
        return np.asarray(self._robot_x.get_jnt_values(), dtype=float)

    def get_tcp_pos(self) -> Optional[np.ndarray]:
        try:
            pos, _ = self._robot_x.get_pose()
        except Exception:
            return None
        return np.asarray(pos, dtype=float)

    def get_jaw_width(self) -> Optional[float]:
        try:
            jaw_width = self._robot_x.get_jaw_width()
        except Exception:
            return None
        if jaw_width is None:
            return None
        return float(jaw_width)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # 借用共享单例时，不在此断开（交由进程级 disconnect_rtde_robot() 统一回收）。
        if self._borrowed:
            return
        try:
            self._robot_x.disconnect()
        except Exception as exc:
            print(f"Warning: RTDE disconnect failed: {exc}")

    def suspend(self) -> None:
        """Suspend RTDE connections before long non-RTDE operations."""
        if self._closed:
            return
        try:
            self._robot_x.suspend_rtde()
        except Exception as exc:
            print(f"Warning: RTDE suspend failed: {exc}")

    def resume(self) -> None:
        """Resume RTDE connections after a suspended operation."""
        if self._closed:
            return
        try:
            self._robot_x.resume_rtde()
        except Exception as exc:
            print(f"Warning: RTDE resume failed: {exc}")


class MockUR7EState:

    def __init__(self, center_conf: np.ndarray):
        self._center_conf = np.asarray(center_conf, dtype=float)
        self._start_time = time.monotonic()

    def get_jnt_values(self) -> np.ndarray:
        t = time.monotonic() - self._start_time
        phase = np.array([0.0, 0.6, 1.1, 1.7, 2.2, 2.8])
        amplitude = np.array([0.18, 0.12, 0.16, 0.13, 0.10, 0.20])
        return self._center_conf + amplitude * np.sin(t * 0.7 + phase)

    def get_tcp_pos(self) -> Optional[np.ndarray]:
        return None

    def get_jaw_width(self) -> Optional[float]:
        t = time.monotonic() - self._start_time
        normalized = 0.5 + 0.5 * math.sin(t * 0.8)
        return DH76_JAW_MIN + normalized * (DH76_JAW_MAX - DH76_JAW_MIN)

    def close(self) -> None:
        return None

    def suspend(self) -> None:
        return None

    def resume(self) -> None:
        return None


@dataclass
class SyncState:
    robot: object
    provider: object
    mesh_model: Optional[object] = None
    last_jnt_values: Optional[np.ndarray] = None
    last_jaw_width: Optional[float] = None
    last_log_time: float = 0.0
    failure_count: int = 0
    update_count: int = 0
    stopped: bool = False


def validate_jnt_values(jnt_values: np.ndarray) -> np.ndarray:
    jnt_values = np.asarray(jnt_values, dtype=float).reshape(-1)
    if jnt_values.shape != (6,):
        raise ValueError(f"Expected 6 joint values, got shape {jnt_values.shape}.")
    if not np.all(np.isfinite(jnt_values)):
        raise ValueError(f"Joint values contain NaN or inf: {jnt_values}.")
    return jnt_values


def clamp_jaw_width_for_robot(robot, jaw_width: Optional[float]) -> Optional[float]:
    if jaw_width is None:
        return None
    jaw_width = float(jaw_width)
    if not np.isfinite(jaw_width):
        return None
    if getattr(robot, "hnd", None) is not None:
        jaw_min, jaw_max = robot.hnd.jaw_range
    else:
        jaw_min, jaw_max = DH76_JAW_MIN, DH76_JAW_MAX
    clamped = float(np.clip(jaw_width, jaw_min, jaw_max))
    if abs(clamped - jaw_width) > 1e-6:
        print(f"Warning: clamped jaw width from {jaw_width:.6f} to {clamped:.6f} m.")
    return clamped


def read_provider_jaw_width(provider) -> Optional[float]:
    if not hasattr(provider, "get_jaw_width"):
        return None
    try:
        return provider.get_jaw_width()
    except Exception as exc:
        print(f"Warning: failed to read gripper jaw width: {exc}")
        return None


def draw_synced_robot(
    base,
    state: SyncState,
    jnt_values: np.ndarray,
    args: argparse.Namespace,
    jaw_width: Optional[float] = None,
) -> None:
    if state.mesh_model is not None:
        state.mesh_model.detach()

    state.robot.goto_given_conf(jnt_values=jnt_values)
    jaw_width = clamp_jaw_width_for_robot(state.robot, jaw_width)
    if jaw_width is not None:
        state.robot.jaw_to(jawwidth=jaw_width)

    state.mesh_model = state.robot.gen_meshmodel(
        alpha=args.alpha,
        toggle_tcp_frame=not args.hide_tcp_frame,
        toggle_flange_frame=not args.hide_flange_frame,
        toggle_jnt_frames=args.show_joint_frames,
    )
    state.mesh_model.attach_to(base)
    state.last_jnt_values = jnt_values
    state.last_jaw_width = jaw_width


def maybe_print_status(state: SyncState, args: argparse.Namespace) -> None:
    now = time.monotonic()
    if args.print_interval <= 0 or now - state.last_log_time < args.print_interval:
        return
    state.last_log_time = now
    jnt_deg = np.degrees(state.last_jnt_values)
    msg = f"synced #{state.update_count}: joints(deg)={np.round(jnt_deg, 2).tolist()}"
    if state.last_jaw_width is not None:
        msg += f", jaw_width(m)={state.last_jaw_width:.4f}"
    tcp_pos = state.provider.get_tcp_pos()
    if tcp_pos is not None:
        msg += f", tcp_pos(m)={np.round(tcp_pos, 4).tolist()}"
    print(msg)


def make_sync_task(base, state: SyncState, args: argparse.Namespace):

    def update(task):
        if state.stopped:
            return task.done
        try:
            jnt_values = validate_jnt_values(state.provider.get_jnt_values())
            jaw_width = read_provider_jaw_width(state.provider)
            draw_synced_robot(base, state, jnt_values, args, jaw_width=jaw_width)
            state.failure_count = 0
            state.update_count += 1
            maybe_print_status(state, args)
        except Exception as exc:
            state.failure_count += 1
            print(f"Warning: sync update failed ({state.failure_count}/{args.max_failures}): {exc}")
            if state.failure_count >= args.max_failures:
                print("Stopping sync task after too many consecutive failures.")
                state.stopped = True
                state.provider.close()
                return task.done

        return task.again

    return update


def build_scene(args: argparse.Namespace) -> None:
    from wrs import mgm, wd
    from wrs.robot_sim.robots.ur7e.ur7e_withouttable_dh76 import UR7EDH76

    base = wd.World(cam_pos=[2.0, -1.6, 1.2], lookat_pos=[0.4, -0.25, 0.3])
    mgm.gen_frame(ax_length=0.25, ax_radius=0.004).attach_to(base)

    if not args.no_env:
        from yanjiuyuan.mech_eye_ur7e_pointcloud_env import attach_obstacles

        attach_obstacles(base, show_collision=args.show_obstacle_collision)

    robot_s = UR7EDH76(enable_cc=True)
    provider = MockUR7EState(DEFAULT_HOME_CONF) if args.mock else RealUR7ERTDEState(args.robot_ip, args.gp_port)
    state = SyncState(robot=robot_s, provider=provider)
    atexit.register(provider.close)

    first_jnt_values = validate_jnt_values(provider.get_jnt_values())
    first_jaw_width = read_provider_jaw_width(provider)
    draw_synced_robot(base, state, first_jnt_values, args, jaw_width=first_jaw_width)
    print(f"Initial joints(deg): {np.round(np.degrees(first_jnt_values), 2).tolist()}")
    if state.last_jaw_width is not None:
        print(f"Initial jaw_width(m): {state.last_jaw_width:.6f}")

    if args.once:
        print("One-shot pose drawn. The WRS window will stay open.")
        provider.close()
    else:
        base.taskMgr.doMethodLater(
            max(args.period, 0.001),
            make_sync_task(base, state, args),
            "sync_real_ur7e_to_wrs",
            appendTask=True,
        )
    base.run()


def main() -> None:
    args = parse_args()
    build_scene(args)


if __name__ == "__main__":
    main()
