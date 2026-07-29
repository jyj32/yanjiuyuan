"""Utilities for turning WRS pick-and-place motion into RTDE execution steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from wrs.basis import robot_math as rm


@dataclass
class RtdeMotionSegment:
    name: str
    command: str
    start_index: int
    end_index: int
    path: list[list[float]] = field(default_factory=list)
    direction: Optional[list[float]] = None
    distance: Optional[float] = None
    jaw_width: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Native moveL target (base-frame 6D pose: x, y, z, rx, ry, rz) and speeds.
    pose: Optional[list[float]] = None
    vel: Optional[float] = None
    acc: Optional[float] = None


@dataclass
class RtdeExecutionPlan:
    segments: list[RtdeMotionSegment]
    metadata: dict[str, Any] = field(default_factory=dict)


class RtdeExecutionError(RuntimeError):
    """Raised when an RTDE execution segment fails and the plan is aborted."""

    def __init__(
        self,
        message: str,
        segment: Optional[RtdeMotionSegment] = None,
        log: Optional[list[dict[str, Any]]] = None,
        result: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.segment = segment
        self.log = [] if log is None else log
        self.result = result


MOVE_L_COMPLIANT_KWARGS = {
    "force",
    "vel",
    "lateral_tolerance",
    "lateral_stop_tolerance",
    "rotation_tolerance",
    "max_tcp_force",
    "max_tcp_torque",
    "timeout",
    "stall_timeout",
    "stall_distance",
    "damping",
    "gain_scaling",
    "zero_ft_sensor",
    "check_safety_limits",
    "force_frame",
    "direction_axis_threshold",
    "compliant_axes",
    "use_actual_tcp_z_direction",
    "dwell_after_stop",
    "singularity_fallback",
}

# Default speeds for native moveL transfer segments (m/s, m/s^2). Pick-and-place
# is conservative; tune via segment.vel / segment.acc or the planner caller.
MOVE_L_DEFAULT_VEL = 0.25
MOVE_L_DEFAULT_ACC = 0.5


def _move_l_compliant_kwargs(
    metadata: dict[str, Any],
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    kwargs = {key: value for key, value in dict(metadata).items() if key in MOVE_L_COMPLIANT_KWARGS}
    if overrides:
        unsupported = sorted(set(overrides) - MOVE_L_COMPLIANT_KWARGS)
        if unsupported:
            raise ValueError(f"Unsupported moveL_compliant kwargs: {unsupported}")
        kwargs.update(overrides)
    return kwargs


def _as_array(vector: Iterable[float]) -> np.ndarray:
    return np.asarray(vector, dtype=float)


def _unit_vector(vector: Iterable[float]) -> np.ndarray:
    vector = _as_array(vector)
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        raise ValueError("zero-length vector cannot be normalized")
    return vector / length


def _pose_tcp(object_pose, grasp) -> tuple[np.ndarray, np.ndarray]:
    pos = _as_array(object_pose.pos) + _as_array(object_pose.rotmat).dot(_as_array(grasp.ac_pos))
    rotmat = _as_array(object_pose.rotmat).dot(_as_array(grasp.ac_rotmat))
    return pos, rotmat


def _nearest_index(points: np.ndarray, target: np.ndarray, start: int = 0, stop: Optional[int] = None) -> int:
    if stop is None:
        stop = len(points)
    if stop <= start:
        raise ValueError("invalid nearest-index range")
    distances = np.linalg.norm(points[start:stop] - target.reshape(1, 3), axis=1)
    return int(start + np.argmin(distances))


def _slice_path(jv_list: list[np.ndarray], start: int, end: int) -> list[list[float]]:
    if end < start:
        return []
    return [np.asarray(conf, dtype=float).tolist() for conf in jv_list[start : end + 1]]
def _robot_base_pose_world(robot) -> tuple[np.ndarray, np.ndarray]:
    arm = getattr(robot, "arm", getattr(robot, "manipulator", None))
    if arm is None:
        return np.zeros(3), np.eye(3)
    base_pos = np.asarray(getattr(arm, "pos", np.zeros(3)), dtype=float).reshape(3)
    base_rotmat = np.asarray(getattr(arm, "rotmat", np.eye(3)), dtype=float).reshape(3, 3)
    return base_pos, base_rotmat


def _world_points_to_robot_base(points_world: np.ndarray, base_pos: np.ndarray, base_rotmat: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=float).reshape((-1, 3))
    return (base_rotmat.T @ (points_world - base_pos).T).T


def _tcp_positions_world_and_base(robot, jv_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_pos, base_rotmat = _robot_base_pose_world(robot)
    tcp_positions_world = []
    backup = getattr(robot, "backup_state", None)
    restore = getattr(robot, "restore_state", None)
    if backup is not None:
        backup()
    try:
        for conf in jv_list:
            tcp_pos, _tcp_rotmat = robot.fk(jnt_values=conf)
            tcp_positions_world.append(np.asarray(tcp_pos, dtype=float))
    finally:
        if restore is not None:
            restore()
    tcp_positions_world = np.asarray(tcp_positions_world, dtype=float)
    tcp_positions_base = _world_points_to_robot_base(tcp_positions_world, base_pos, base_rotmat)
    return tcp_positions_world, tcp_positions_base, base_pos, base_rotmat


def _find_pre_pick_index_by_tcp_path(tcp_positions: np.ndarray, pick_index: int, approach_distance: float,
                                     collinear_angle_tol: float = 0.26) -> int:
    """Locate the start of the straight approach segment before ``pick_index``.

    从 pick 向前回溯：只要相邻路点的步进方向与"进入 pick 的方向"共线
    （夹角 <= collinear_angle_tol，默认约 15 度），就认为仍处于直线接近段内；
    方向一旦拐弯（进入 RRT 自由段）立即停止。这样终点锁定在直线接近段起点，
    不受 RRT 抖动虚增弧长、approach_distance 与实际路径不一致、浮点边界等影响。
    ``approach_distance`` 仅作为直线距离上限，防止把恰好共线的长直线 RRT 段全部吞入。
    """
    if pick_index <= 0:
        return 0
    target_distance = max(float(approach_distance), 0.0)
    pick_pos = tcp_positions[pick_index]
    ref_dir = None
    boundary = pick_index
    for index in range(pick_index, 0, -1):
        step = tcp_positions[index] - tcp_positions[index - 1]
        step_norm = float(np.linalg.norm(step))
        if step_norm < 1e-9:
            boundary = index - 1  # 重复路点（夹爪动作等原地停留），跳过
            continue
        step_dir = step / step_norm
        if ref_dir is None:
            ref_dir = step_dir  # 以最靠近 pick 的非零步作为接近方向基准
        cos_angle = float(np.clip(np.dot(step_dir, ref_dir), -1.0, 1.0))
        if math.acos(cos_angle) > collinear_angle_tol:
            break  # 方向拐弯：已越过直线接近段起点
        boundary = index - 1
        if target_distance > 1e-6:
            straight = float(np.linalg.norm(tcp_positions[boundary] - pick_pos))
            if straight >= target_distance * 0.999:
                break  # 直线距离已覆盖期望接近距离（留 0.1% 浮点容差）
    if boundary < pick_index:
        return boundary
    return max(0, pick_index - 1)


def _find_jaw_transition(ev_list: list[Any], start: int, direction: str, tolerance: float = 1e-5) -> Optional[int]:
    values = [None if value is None else float(np.asarray(value).reshape(-1)[0]) for value in ev_list]
    for index in range(max(start + 1, 1), len(values)):
        prev = values[index - 1]
        current = values[index]
        if prev is None or current is None:
            continue
        delta = current - prev
        if direction == "close" and delta < -tolerance:
            return index
        if direction == "open" and delta > tolerance:
            return index
    return None


def _tcp_pose_base_6d(robot, conf, base_pos: np.ndarray, base_rotmat: np.ndarray) -> np.ndarray:
    """Full base-frame TCP pose (x, y, z, rx, ry, rz) for a joint config."""
    tcp_pos_world, tcp_rotmat_world = robot.fk(jnt_values=conf)
    tcp_pos_world = np.asarray(tcp_pos_world, dtype=float).reshape(3)
    tcp_rotmat_world = np.asarray(tcp_rotmat_world, dtype=float).reshape(3, 3)
    pos_base = base_rotmat.T @ (tcp_pos_world - base_pos)
    rotmat_base = base_rotmat.T @ tcp_rotmat_world
    rotvec = rm.rotmat_to_wvec(rotmat_base)
    return np.concatenate([pos_base, np.asarray(rotvec, dtype=float).reshape(3)])


def _transfer_waypoint_indices(transfer_base_pos: np.ndarray, min_angle: float = 0.5) -> Optional[tuple[int, int]]:
    """Find the two interior direction-change indices of a 3-segment, axis-aligned
    transfer path (lift -> horizontal -> lower) so it can be replayed as moveL.

    Returns ``(a, b)`` with ``0 < a < b < n`` or ``None`` if the geometry is not a
    clean 3-segment path (in which case the caller should fall back to a joint path).
    """
    transfer_base_pos = np.asarray(transfer_base_pos, dtype=float).reshape(-1, 3)
    n = transfer_base_pos.shape[0]
    if n < 4:
        return None
    steps = np.diff(transfer_base_pos, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    if norms.max() < 1e-9:
        return None
    dirs = steps / (norms.reshape(-1, 1) + 1e-12)
    changes = []
    for i in range(len(dirs) - 1):
        cosang = float(np.clip(np.dot(dirs[i], dirs[i + 1]), -1.0, 1.0))
        if math.acos(cosang) > min_angle:
            changes.append(i + 1)
    if len(changes) < 2:
        return None
    a, b = changes[0], changes[1]
    if not (0 < a < b < n):
        return None
    return a, b


def _emit_transfer_segments(
    segments: list,
    robot,
    jv_list: list[np.ndarray],
    tcp_positions_base: np.ndarray,
    post_pick_start: int,
    open_index: int,
    transfer_movel_waypoints_world: Optional[list] = None,
    place_pos_base: Optional[list] = None,
    depart_distance: Optional[float] = None,
) -> None:
    """Emit the transfer as up to three native ``moveL`` segments (lift / horizontal /
    lower). The transfer keeps TCP orientation constant, so every moveL target reuses
    the base-frame rotation vector captured at the grasp point.

    When ``transfer_movel_waypoints_world`` (a 3-tuple of world-frame TCP positions for
    the lift-top / above-place / place targets) is provided, those exact targets are
    used. Otherwise the two direction-change points are auto-detected from the joint
    path. Falls back to a single ``move_jntspace_path`` segment when the transfer
    geometry cannot be parsed.

    Each emitted ``moveL`` segment additionally carries ``anchor_to_actual_tcp``
    metadata so the executor can recompute the three target poses from the *actual*
    TCP pose read at the grasp point (``getActualTCPPose``), instead of trusting the
    simulated forward-kinematics pose. ``place_pos_base`` is the base-frame place
    target and ``depart_distance`` is the vertical lift-off distance.
    """
    base_pos, base_rotmat = _robot_base_pose_world(robot)
    rotvec = _tcp_pose_base_6d(robot, jv_list[post_pick_start], base_pos, base_rotmat)[3:].tolist()

    if transfer_movel_waypoints_world is not None and len(transfer_movel_waypoints_world) == 3:
        waypoint_base_pos = []
        for w in transfer_movel_waypoints_world:
            wpos = np.asarray(w, dtype=float).reshape(3)
            waypoint_base_pos.append((base_rotmat.T @ (wpos - base_pos)).tolist())
    else:
        transfer_base_pos = np.asarray(tcp_positions_base[post_pick_start : open_index + 1], dtype=float)
        wp = _transfer_waypoint_indices(transfer_base_pos)
        if wp is None:
            segments.append(
                RtdeMotionSegment(
                    name="transfer_to_place",
                    command="move_jntspace_path",
                    start_index=post_pick_start,
                    end_index=open_index,
                    path=_slice_path(jv_list, post_pick_start, open_index),
                )
            )
            return
        a, b = wp
        waypoint_global_idx = [post_pick_start + a, post_pick_start + b, open_index]
        waypoint_base_pos = [
            np.asarray(tcp_positions_base[widx], dtype=float).reshape(3).tolist() for widx in waypoint_global_idx
        ]

    _anchor = place_pos_base is not None and depart_distance is not None
    if _anchor and len(waypoint_base_pos) == 3:
        # 让仿真回退 pose（dry-run / 实机读取失败）也对齐显式基坐标系放置点：
        # wp2 水平段用放置点 XY（保持抬起高度），wp3 直接落到放置点。
        _pb = np.asarray(place_pos_base, dtype=float).reshape(3)
        waypoint_base_pos[1] = [float(_pb[0]), float(_pb[1]), float(waypoint_base_pos[1][2])]
        waypoint_base_pos[2] = _pb.tolist()
    for k, pos in enumerate(waypoint_base_pos):
        _meta = {"path_type": "pick_place_movel", "transfer_subsegment": k + 1}
        if _anchor:
            _meta["anchor_to_actual_tcp"] = True
            _meta["transfer_role"] = k + 1
            _meta["place_pos_base"] = list(place_pos_base)
            _meta["depart_distance"] = float(depart_distance)
            _meta["grasp_rotvec_fallback"] = list(rotvec)
        segments.append(
            RtdeMotionSegment(
                name=f"transfer_to_place_movel_{k + 1}",
                command="moveL",
                start_index=post_pick_start,
                end_index=open_index,
                pose=list(pos) + rotvec,
                vel=MOVE_L_DEFAULT_VEL,
                acc=MOVE_L_DEFAULT_ACC,
                metadata=_meta,
            )
        )


def _emit_push_leave(
    segments: list,
    robot,
    jv_list: list,
    pick_index: int,
    base_pos_world: np.ndarray,
    base_rotmat_world: np.ndarray,
    depart_distance: float,
    vel: float,
    acc: float,
) -> None:
    """为 push 计划追加一段竖直 moveL 离开（运行时以**实际** TCP 位姿沿基系 +Z 抬起 depart_distance）。

    规划期先放一个 FK 计算出的竖直兜底位姿（robot.fk → 世界 → 基系，仅 Z 叠加 depart_distance）；
    真正的目标位姿会在 dual 执行端、该段执行前用 ``rtde_robot.getActualTCPPose()`` 重算
    （与抓取上移一致），所以这里**不挂** ``anchor_to_actual_tcp``——否则会误入执行端的
    “搬运段重算”逻辑（该逻辑硬性要求 ``place_pos_base`` 元数据，push 没有会 KeyError 回退，
    导致沿用的 FK 位姿帧有偏差、moveL 冲向错误位置触发保护性停机）。

    抬升仅在 Z 分量叠加 depart_distance，其余（x/y/姿态）沿用接触点实际值 → 竖直向上。
    """
    actual_6d = _tcp_pose_base_6d(robot, jv_list[pick_index], base_pos_world, base_rotmat_world)
    target_6d = np.asarray(actual_6d, dtype=float).reshape(6).copy()
    target_6d[2] += float(depart_distance)  # 仅基系 +Z 抬高 depart_distance（规划期兜底；执行期会被实际 TCP 覆盖）
    segments.append(
        RtdeMotionSegment(
            name="push_leave_movel",
            command="moveL",
            start_index=pick_index,
            end_index=pick_index,
            pose=target_6d.tolist(),
            vel=float(vel),
            acc=float(acc),
            metadata={"path_type": "push_leave", "depart_distance": float(depart_distance)},
        )
    )


def build_pick_place_rtde_plan(
    robot,
    mot_data,
    pick_pose,
    place_pose,
    grasp,
    planner_name: str = "WRS PickPlacePlanner",
    compliant_force: float = 10.0,
    compliant_vel: float = 0.02,
    compliant_lateral_tolerance: float = 0.01,
    compliant_lateral_stop_tolerance: float | str | None = "auto",
    compliant_force_frame: str = "direction",
    compliant_axes: Any = None,
    compliant_zero_ft_sensor: bool = False,
    compliant_max_tcp_force: float = 50.0,
    compliant_timeout: Optional[float] = None,
    compliant_dwell_after_stop: float = 2.0,
    transfer_movel_waypoints_world: Optional[list] = None,
    transfer_movel_place_pos_base: Optional[list] = None,
    # 放置后力控竖直下压（将物体坐实/压实）。默认开启：机器人竖直下落到放置点后、
    # 松开手爪前，沿实际 TCP 工具 +Z（竖直向下）以合规力控下压一段距离并保压，
    # 到位或触力即停，然后才松开手爪。
    place_press_enabled: bool = True,
    place_press_distance: float = 0.10,
    place_press_force: float = 40.0,
    place_press_vel: float = 0.08,
    place_press_lateral_tolerance: float = 0.01,
    place_press_lateral_stop_tolerance: float | str | None = "auto",
    place_press_zero_ft_sensor: bool = True,
    place_press_max_tcp_force: float = 50.0,
    place_press_timeout: Optional[float] = None,
    place_press_dwell_after_stop: float = 2.0,
    # 推开（push）专用：力控接触后闭合手爪，再竖直 moveL 离开并张开。
    # 仅在 mot_data.path_type == "push" 时生效，替代 pick 的 transfer_to_place + place_press。
    push_depart_distance: float = 0.35,    # 竖直离开距离 (m)
    push_leave_vel: float = 0.1,          # 离开 moveL 速度 (m/s)
    push_leave_acc: float = 0.3,          # 离开 moveL 加速度 (m/s^2)
    # 推开后追加“往箱子中心的水平力控推”（仅 push 路径生效）。
    # push_center_pos_world 为箱子中心（sim 世界系）；为 None 时不生成该段。
    # 力控方向 = 接触点→箱子中心的水平方向，并经 robot.get_real_tcp_pose_sim 换算到
    # 【真实机器人基系】（moveL_compliant 的 direction 按真实基系解释）。
    push_center_pos_world: Optional[list] = None,
    push_center_distance: float = 0.10,            # 往箱子中心推的距离 (m)
    push_center_force: float = 30.0,               # 力控推力 (N)
    push_center_vel: float = 0.06,                 # 力控速度 (m/s)
    push_center_lateral_tolerance: float = 0.02,
    push_center_lateral_stop_tolerance: float | str | None = "auto",
    push_center_zero_ft_sensor: bool = True,
    push_center_max_tcp_force: float = 25.0,
    push_center_timeout: Optional[float] = 10.0,
    push_center_dwell_after_stop: float = 2.0,
) -> RtdeExecutionPlan:
    """Split WRS MotionData into RTDE-friendly execution segments.
    # 完整路径规划
    The approach-to-object segment is marked as ``moveL_compliant`` and also
    keeps its joint path so execution can opt out of compliant motion.
    """
    jv_list = [np.asarray(conf, dtype=float) for conf in mot_data.jv_list]
    if not jv_list:
        raise ValueError("mot_data contains no joint path")
    ev_list = list(mot_data.ev_list)
    tcp_positions_world, tcp_positions_base, base_pos_world, base_rotmat_world = _tcp_positions_world_and_base(robot, jv_list)

    pick_tcp_pos, pick_tcp_rotmat = _pose_tcp(pick_pose, grasp)
    grasp_jaw_width = float(getattr(grasp, "jaw_width", 0.0))
    pick_approach_jaw_width = float(getattr(grasp, "pick_approach_jaw_width", grasp_jaw_width))
    approach_distance = float(getattr(grasp, "approach_distance", 0.0))
    pick_index = _nearest_index(tcp_positions_world, pick_tcp_pos)

    # 固定方向 = 抓取姿态的工具 +Z，与规划侧 ApproachPlanner / sim_pick 完全一致。
    approach_direction_world = _unit_vector(np.asarray(pick_tcp_rotmat[:, 2], dtype=float))
    approach_direction_base = base_rotmat_world.T @ approach_direction_world

    if approach_distance > 1e-6:
        # 规划侧透传固定几何：pre_pick_pos = pick - 工具Z * approach_distance。
        # 直接用该精确 pre_pick_pos 在整条 FK 轨迹中反查直线接近段起点索引，
        # 不再从抖动路点反向累计弧长 / 共线启发式（受 RRT 抖动、浮点边界影响）。
        pre_pick_pos_world = pick_tcp_pos - approach_direction_world * approach_distance
        pre_pick_index = _nearest_index(tcp_positions_world, pre_pick_pos_world, start=0, stop=pick_index + 1)
        compliant_distance = approach_distance  # 用规划固定距离，而非路径反推
        pre_pick_source = "planned_geometry"
    else:
        # 回退：grasp 未透传 approach_distance，仍用共线检测从路径里定位直线起点。
        pre_pick_index = _find_pre_pick_index_by_tcp_path(tcp_positions_world, pick_index, 0.0)
        compliant_distance = None
        pre_pick_source = "tcp_path_search"
    if pre_pick_index > pick_index:
        pre_pick_index = pick_index

    pre_pick_tcp_pos_world = tcp_positions_world[pre_pick_index]
    pick_path_tcp_pos_world = tcp_positions_world[pick_index]
    pre_pick_tcp_pos_base = tcp_positions_base[pre_pick_index]
    pick_path_tcp_pos_base = tcp_positions_base[pick_index]

    if compliant_distance is None or compliant_distance <= 1e-6:
        # 回退：用直线段两端 TCP 反推距离 / 方向（与既有行为一致）。
        approach_delta_base = pick_path_tcp_pos_base - pre_pick_tcp_pos_base
        compliant_distance = float(np.linalg.norm(approach_delta_base))
        if compliant_distance <= 1e-6:
            approach_direction_base = base_rotmat_world.T @ approach_direction_world
            compliant_distance = approach_distance
    approach_direction_world = base_rotmat_world @ approach_direction_base

    close_index = _find_jaw_transition(ev_list, pick_index, "close")
    if close_index is None:
        close_index = pick_index
    open_index = _find_jaw_transition(ev_list, close_index, "open")
    forced_open_index = getattr(mot_data, "forced_open_index", None)
    if bool(getattr(mot_data, "force_final_open_gripper", False)):
        if forced_open_index is None:
            forced_open_index = len(jv_list) - 1
        forced_open_index = int(np.clip(int(forced_open_index), 0, len(jv_list) - 1))
        if open_index is None or open_index < forced_open_index:
            open_index = forced_open_index

    is_push_path = str(getattr(mot_data, "path_type", "")) == "push"
    segments: list[RtdeMotionSegment] = []
    # 1.在 RRT 接近路径（move_to_pre_pick）之前先把夹爪设到接近宽度 pick_approach_jaw_width，
    # 使整个 RRT 运动期间手爪宽度与规划时的碰撞假设一致（RRT 是按该张开宽度做避障规划的，
    # 否则执行时夹爪仍继承上一状态=张开 0.118，可能与规划时的窄宽度碰撞假设不一致）。
    segments.append(
        RtdeMotionSegment(
            name="set_pick_approach_jaw_width",
            command="close_gripper_to",
            start_index=0,
            end_index=0,
            jaw_width=pick_approach_jaw_width,
            metadata={"grasp_jaw_width": grasp_jaw_width, "timing": "before_move_to_pre_pick"},
        )
    )
    # 2.接近路径
    if pre_pick_index > 0:
        segments.append(
            RtdeMotionSegment(
                name="move_to_pre_pick",
                command="move_jntspace_path",
                start_index=0,
                end_index=pre_pick_index,
                path=_slice_path(jv_list, 0, pre_pick_index),
            )
        )
    # 3.抓取路径或推开路径的下压
    segments.append(
        RtdeMotionSegment(
            name="approach_pick_compliant",
            command="moveL_compliant",
            start_index=pre_pick_index,
            end_index=pick_index,
            path=_slice_path(jv_list, pre_pick_index, pick_index),
            direction=approach_direction_base.tolist(),
            distance=compliant_distance,
            metadata={
                "force": float(compliant_force),
                "vel": float(compliant_vel),
                "lateral_tolerance": float(compliant_lateral_tolerance),
                "lateral_stop_tolerance": compliant_lateral_stop_tolerance,
                "force_frame": str(compliant_force_frame),
                "compliant_axes": compliant_axes,
                "zero_ft_sensor": bool(compliant_zero_ft_sensor),
                "max_tcp_force": float(compliant_max_tcp_force),
                "timeout": compliant_timeout,
                "dwell_after_stop": float(compliant_dwell_after_stop),
                "direction_source": "real_tool_z_via_getActualTCPPose",
                "use_actual_tcp_z_direction": True, # 沿真实手爪 z 轴方向力控
                "pre_pick_source": pre_pick_source,
                "planned_start_tcp": pre_pick_tcp_pos_base.tolist(),
                "planned_goal_tcp": pick_path_tcp_pos_base.tolist(),
                "planned_start_tcp_base": pre_pick_tcp_pos_base.tolist(),
                "planned_goal_tcp_base": pick_path_tcp_pos_base.tolist(),
                "planned_start_tcp_world": pre_pick_tcp_pos_world.tolist(),
                "planned_goal_tcp_world": pick_path_tcp_pos_world.tolist(),
                # 力控方向
                "approach_direction_world": approach_direction_world.tolist(),
                "robot_base_pos_world": base_pos_world.tolist(),
                "robot_base_rotmat_world": base_rotmat_world.tolist(),
            },
        )
    )
    if is_push_path:    # 推开动作
        # ---- push 剖面：力控接触(推瓶) → [水平力控推向箱子中心] → moveL 竖直上抬离开 ----
        # 1) approach_pick_compliant 已把瓶子沿推压方向（工具 +Z）推离原位，即“推”的动作；
        # 2) push_to_center_compliant（可选）：手爪保持原状态（张开）沿“接触点→箱子中心”
        #    水平方向力控推一段，把瓶子从箱壁/角落拖回箱子中央，方向经 get_real_tcp_pose_sim 换算到真实基系；
        # 3) moveL 竖直上抬离开（不再夹/放，保持推完时的手爪状态；用 FK 实际 TCP 位姿上移）。
        # 竖直上抬方向 = 机器人基系 +Z（_emit_push_leave 仅对 z 叠加 depart_distance）。
        # 4.5 往箱子中心的水平力控推（可选）：在接触点（手爪张开）沿“接触点→箱子中心”
        # 的水平方向再力控推一段，把瓶子从箱壁/角落拖回箱子中央，便于下一轮抓取。
        # 力控方向必须是【真实机器人基系】方向（moveL_compliant 按真实基系解释 direction）：
        # 用 robot.get_real_tcp_pose_sim 分别换算“接触点”与“朝中心平移后的目标点”的真实
        # 法兰盘位姿，两者位置差即真实基系下的推向（再压平 Z 分量保证严格水平）。
        if push_center_pos_world is not None and float(push_center_distance) > 1e-6:
            _center_world = np.asarray(push_center_pos_world, dtype=float).reshape(3)
            # 接触点→中心 的方向（sim 基系，压平 Z → 水平）
            _d_world = _center_world - np.asarray(pick_tcp_pos, dtype=float).reshape(3)
            _d_base = base_rotmat_world.T @ _d_world
            _d_base[2] = 0.0
            _d_norm = float(np.linalg.norm(_d_base))
            if _d_norm > 1e-6:
                _d_base_unit = _d_base / _d_norm
                _direction_real = None
                _direction_source = "sim_base_horizontal_fallback"
                _get_real = getattr(robot, "get_real_tcp_pose_sim", None)
                if _get_real is not None:
                    try:
                        # 目标点 = 接触点 + 水平单位方向 * 推距（sim 世界系）
                        _tgt_world = (
                            np.asarray(pick_tcp_pos, dtype=float).reshape(3)
                            + (base_rotmat_world @ _d_base_unit) * float(push_center_distance)
                        )
                        _real_contact = np.asarray(
                            _get_real(np.asarray(pick_tcp_pos, dtype=float).reshape(3), pick_tcp_rotmat),
                            dtype=float,
                        ).reshape(6)
                        _real_target = np.asarray(
                            _get_real(_tgt_world, pick_tcp_rotmat), dtype=float
                        ).reshape(6)
                        _d_real = _real_target[:3] - _real_contact[:3]
                        _d_real[2] = 0.0  # 真实基系下同样压平 Z，保证严格水平
                        _d_real_norm = float(np.linalg.norm(_d_real))
                        if _d_real_norm > 1e-6:
                            _direction_real = (_d_real / _d_real_norm).tolist()
                            _direction_source = "real_base_via_get_real_tcp_pose_sim"
                    except Exception as _exc:
                        print(f"[rtde_plan] ⚠️ push_center 真实方向换算失败，回退 sim 基系方向: {_exc}")
                if _direction_real is None:
                    _direction_real = _d_base_unit.tolist()
                segments.append(
                    RtdeMotionSegment(
                        name="push_to_center_compliant",
                        command="moveL_compliant",
                        start_index=pick_index,
                        end_index=pick_index,
                        direction=list(_direction_real),
                        distance=float(push_center_distance),
                        metadata={
                            "force": float(push_center_force),
                            "vel": float(push_center_vel),
                            "lateral_tolerance": float(push_center_lateral_tolerance),
                            "lateral_stop_tolerance": push_center_lateral_stop_tolerance,
                            "force_frame": "direction",
                            "compliant_axes": None,
                            "zero_ft_sensor": bool(push_center_zero_ft_sensor),
                            "max_tcp_force": float(push_center_max_tcp_force),
                            "timeout": push_center_timeout,
                            "dwell_after_stop": float(push_center_dwell_after_stop),
                            # ⭐ 关键：禁止执行端把方向覆盖成“实际工具 +Z”，
                            # 该段方向就是上面换算好的真实基系水平方向。
                            "use_actual_tcp_z_direction": False,
                            "direction_source": _direction_source,
                            "path_type": "push_center",
                            "push_center_pos_world": _center_world.tolist(),
                            "contact_tcp_pos_world": np.asarray(pick_tcp_pos, dtype=float).reshape(3).tolist(),
                            # 触力/卡滞/侧偏/到位均视为正常软停，不中止计划
                            "allowed_stop_reasons": ["distance", "max_tcp_force", "stalled", "lateral_deviation"],
                        },
                    )
                )
            else:
                print("[rtde_plan] ⚠️ push_center: 接触点与箱子中心水平重合，跳过该段")
        # 4.6 push_to_center_compliant 推完后不再夹/放，直接竖直上抬离开（保持原手爪状态）。
        # 6. moveL 竖直上抬离开（从接触点**实际** TCP 位姿沿基系 +Z 抬起 push_depart_distance）
        _emit_push_leave(
            segments,
            robot,
            jv_list,
            pick_index,
            base_pos_world,
            base_rotmat_world,
            push_depart_distance,
            push_leave_vel,
            push_leave_acc,
        )
        # 7. 上抬后保持张开，由外层循环 move_to_capture_point 回拍照点重新检测（无需再闭合）
    else:   # 不推开，抓取
        # 4.闭合手爪，抓取
        segments.append(
            RtdeMotionSegment(
                name="close_gripper",
                command="close_gripper",
                start_index=close_index,
                end_index=close_index,
                jaw_width=grasp_jaw_width,
            )
        )
        # 5.离开路径，放置 —— 统一使用原生 moveL 三段式（实际 TCP 位姿 → 上抬 → 放置点）
        post_pick_start = min(close_index, len(jv_list) - 1)
        if open_index is not None and open_index > post_pick_start:
            # 放置目标在机器人基坐标系下的位置，执行端以【实际 TCP 位姿】为基准
            # 生成三段原生 moveL 时需要它。
            if transfer_movel_place_pos_base is not None:
                _place_pos_base = np.asarray(transfer_movel_place_pos_base, dtype=float).reshape(3).tolist()
            else:
                _place_pos_attr = getattr(place_pose, "pos", None)
                if _place_pos_attr is None:
                    _place_pos_attr = place_pose[0]
                _place_pos_world = np.asarray(_place_pos_attr, dtype=float).reshape(3)
                _place_pos_base = (base_rotmat_world.T @ (_place_pos_world - base_pos_world)).tolist()
            _depart_distance = getattr(mot_data, "transfer_movel_depart_distance", None)
            _emit_transfer_segments(
                segments,
                robot,
                jv_list,
                tcp_positions_base,
                post_pick_start,
                open_index,
                transfer_movel_waypoints_world=transfer_movel_waypoints_world,
                place_pos_base=_place_pos_base,
                depart_distance=_depart_distance,
            )
            # 6.放置时力控竖直下压（将物体坐实/压实）—— 在下落到放置点之后、松开手爪之前
            if place_press_enabled and place_press_distance > 1e-6:
                segments.append(
                    RtdeMotionSegment(
                        name="place_press_compliant",
                        command="moveL_compliant",
                        start_index=open_index,
                        end_index=open_index,
                        direction=[0.0, 0.0, -1.0],  # 世界系竖直向下（base 系 -Z），不随手爪姿态变化
                        distance=float(place_press_distance),
                        metadata={
                            "force": float(place_press_force),
                            "vel": float(place_press_vel),
                            "lateral_tolerance": float(place_press_lateral_tolerance),
                            "lateral_stop_tolerance": place_press_lateral_stop_tolerance,
                            "force_frame": "direction",
                            "compliant_axes": None,
                            "zero_ft_sensor": bool(place_press_zero_ft_sensor),
                            "max_tcp_force": float(place_press_max_tcp_force),
                            "timeout": place_press_timeout,
                            "use_actual_tcp_z_direction": False,    # 是否使用实际的tcp_pose方向
                            "dwell_after_stop": float(place_press_dwell_after_stop),
                            "direction_source": "world_z_down",
                            # 力控段允许的软停原因（力超限/卡滞/侧偏/到位均视为正常结束，不中止计划）
                            "allowed_stop_reasons": ["distance", "max_tcp_force", "stalled", "lateral_deviation"],
                        },
                    )
                )
            # 7.松开手爪
            segments.append(
                RtdeMotionSegment(
                    name="open_gripper",
                    command="open_gripper",
                    start_index=open_index,
                    end_index=open_index,
                )
            )

    place_tcp_pos, _place_tcp_rotmat = _pose_tcp(place_pose, grasp)
    return RtdeExecutionPlan(
        segments=segments,
        metadata={
            "planner": planner_name,
            "selected_grasp": getattr(grasp, "name", None),
            "frame_count": len(jv_list),
            "grasp_jaw_width": grasp_jaw_width,
            "pick_approach_jaw_width": pick_approach_jaw_width,
            "pick_tcp_pos": pick_tcp_pos.tolist(),
            "pick_path_tcp_pos_world": pick_path_tcp_pos_world.tolist(),
            "pre_pick_tcp_pos": pre_pick_tcp_pos_world.tolist(),
            "pre_pick_tcp_pos_base": pre_pick_tcp_pos_base.tolist(),
            "pick_path_tcp_pos_base": pick_path_tcp_pos_base.tolist(),
            "approach_direction_base": approach_direction_base.tolist(),
            "approach_direction_world": approach_direction_world.tolist(),
            "robot_base_pos_world": base_pos_world.tolist(),
            "robot_base_rotmat_world": base_rotmat_world.tolist(),
            "place_tcp_pos": place_tcp_pos.tolist(),
            "pick_index": pick_index,
            "pre_pick_index": pre_pick_index,
            "close_index": close_index,
            "open_index": open_index,
            "path_type": str(getattr(mot_data, "path_type", "")),
            "force_final_open_gripper": bool(getattr(mot_data, "force_final_open_gripper", False)),
            "forced_open_index": forced_open_index,
        },
    )


def save_rtde_execution_plan(plan: RtdeExecutionPlan, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")


def load_rtde_execution_plan(path: Path) -> RtdeExecutionPlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return RtdeExecutionPlan(
        segments=[RtdeMotionSegment(**segment) for segment in raw["segments"]],
        metadata=raw.get("metadata", {}),
    )


def _rtde_control(rtde_robot):
    return getattr(rtde_robot, "rtde_c", getattr(rtde_robot, "_rtde_c", None))


def _rtde_receive(rtde_robot):
    return getattr(rtde_robot, "rtde_r", getattr(rtde_robot, "_rtde_r", None))


def _try_stop_rtde_motion(rtde_robot) -> None:
    controller = _rtde_control(rtde_robot)
    if controller is None:
        return
    for method_name in ("forceModeStop", "servoStop", "speedStop", "stopJ", "stopScript"):
        method = getattr(controller, method_name, None)
        if method is None:
            continue
        try:
            if method_name == "stopJ":
                method(1.5)
            else:
                method()
        except Exception:
            pass


def _assert_not_stopped(rtde_robot, segment: RtdeMotionSegment, log: list[dict[str, Any]]) -> None:
    receiver = _rtde_receive(rtde_robot)
    if receiver is None:
        return
    checks = (("protective_stop", "isProtectiveStopped"), ("emergency_stop", "isEmergencyStopped"))
    for reason, method_name in checks:
        method = getattr(receiver, method_name, None)
        if method is not None and method():
            _try_stop_rtde_motion(rtde_robot)
            raise RtdeExecutionError(f"RTDE safety state is active: {reason}", segment=segment, log=log)


def _check_start_joint_alignment(
    rtde_robot,
    segment: RtdeMotionSegment,
    planned_start: list[float],
    max_start_joint_error: float,
    log: list[dict[str, Any]],
) -> None:
    if not hasattr(rtde_robot, "get_jnt_values"):
        return
    actual = np.asarray(rtde_robot.get_jnt_values(), dtype=float).reshape(-1)
    planned = np.asarray(planned_start, dtype=float).reshape(-1)
    if actual.shape != planned.shape:
        raise RtdeExecutionError(
            f"Actual joint vector shape {actual.shape} does not match planned {planned.shape}.",
            segment=segment,
            log=log,
        )
    max_error = float(np.max(np.abs(actual - planned)))
    if max_error > float(max_start_joint_error):
        raise RtdeExecutionError(
            f"Refusing {segment.name}: actual joints differ from planned start by "
            f"{math.degrees(max_error):.2f} deg, limit is {math.degrees(max_start_joint_error):.2f} deg.",
            segment=segment,
            log=log,
        )


def _jntspace_kwargs_for_segment(
    segment: RtdeMotionSegment,
    base_kwargs: dict[str, Any],
    segment_kwargs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path_kwargs = dict(base_kwargs)
    for key in ("*", segment.command, segment.name):
        overrides = segment_kwargs.get(key)
        if overrides:
            path_kwargs.update(overrides)
    path_kwargs.setdefault("move_to_start", True)
    return path_kwargs


def execute_rtde_execution_plan(
    rtde_robot,
    plan: RtdeExecutionPlan,
    dry_run: bool = True,
    prepend_actual_joint: bool = False,
    jntspace_kwargs: Optional[dict[str, Any]] = None,
    jntspace_kwargs_by_segment: Optional[dict[str, dict[str, Any]]] = None,
    compliant_kwargs: Optional[dict[str, Any]] = None,
    allowed_compliant_stop_reasons: tuple[str, ...] = ("distance",),
    max_start_joint_error: float = math.radians(5.0),
    use_move_l_compliant: bool = False,
) -> list[dict[str, Any]]:
    """Execute or preview an RTDE plan with a UR7EDH76_RTDE-like object.
        机器人rtde执行
    Real execution is fail-fast.  Any exception, unexpected compliant stop
    reason, protective stop, emergency stop, or large start-pose mismatch stops
    the current RTDE motion and raises RtdeExecutionError; later segments are
    not executed. ``moveL_compliant`` segments are executed as ordinary joint
    paths unless ``use_move_l_compliant`` is True.
    """
    jntspace_kwargs = {} if jntspace_kwargs is None else dict(jntspace_kwargs)
    jntspace_kwargs_by_segment = (
        {}
        if jntspace_kwargs_by_segment is None
        else {str(name): dict(kwargs) for name, kwargs in jntspace_kwargs_by_segment.items()}
    )
    compliant_kwargs = {} if compliant_kwargs is None else dict(compliant_kwargs)
    log = []
    for segment in plan.segments:
        entry = {"name": segment.name, "command": segment.command}
        try:
            if not dry_run:
                _assert_not_stopped(rtde_robot, segment, log)
            if segment.command == "move_jntspace_path":
                path = [np.asarray(conf, dtype=float).reshape(-1) for conf in segment.path]
                entry["waypoints"] = len(path)
                entry["move_to_start"] = True
                if not dry_run and path:
                    _check_start_joint_alignment(rtde_robot, segment, path[0], max_start_joint_error, log)
                    if prepend_actual_joint:
                        path[0] = np.asarray(rtde_robot.get_jnt_values(), dtype=float).reshape(-1)
                    path_kwargs = _jntspace_kwargs_for_segment(segment, jntspace_kwargs, jntspace_kwargs_by_segment)
                    entry["jntspace_kwargs"] = dict(path_kwargs)
                    rtde_robot.move_jntspace_path(path, **path_kwargs)
                    rtde_robot.move_jnts(path[-1], 0.5, 0.5)    # 避免到不了目标点
                    _assert_not_stopped(rtde_robot, segment, log)
            elif segment.command == "moveL_compliant":  # 力控，现在还是位控
                entry["direction"] = segment.direction
                entry["distance"] = segment.distance
                entry["metadata"] = dict(segment.metadata)
                if use_move_l_compliant:    # 使用力控
                    kwargs = _move_l_compliant_kwargs(segment.metadata, compliant_kwargs)
                    entry["executed_as"] = "moveL_compliant"
                    if not dry_run:
                        # 运行时默认改用实际 TCP 工具 +Z 作为力控方向，规避模型/真实基座系不一致；
                        # 软结束标志触发后继续力控保压 2s 再结束。
                        # 若段元数据【显式】给出 use_actual_tcp_z_direction（如 push_to_center_compliant
                        # 已把方向换算到真实基系，设为 False），则尊重段设置，不强制覆盖。
                        kwargs = dict(kwargs)
                        kwargs.setdefault("use_actual_tcp_z_direction", True)
                        kwargs.setdefault("dwell_after_stop", 2.0)  # 检测到反力后仍运行2s
                        result = rtde_robot.moveL_compliant(segment.direction, segment.distance, **kwargs)
                        entry["result"] = result
                        stop_reason = str(result.get("stop_reason", "unknown"))
                        _segment_allowed = tuple(
                            segment.metadata.get("allowed_stop_reasons", allowed_compliant_stop_reasons)
                        )
                        if stop_reason not in _segment_allowed and stop_reason != "skipped_safety":
                            detail = (
                                f"travelled={float(result.get('travelled', 0.0)):.4f}/"
                                f"{float(result.get('target_distance', 0.0)):.4f}m, "
                                f"lateral_error={float(result.get('lateral_error', 0.0)):.4f}m, "
                                f"lateral_stop_tolerance={result.get('lateral_stop_tolerance')}, "
                                f"elapsed={float(result.get('elapsed', 0.0)):.2f}s"
                            )
                            raise RtdeExecutionError(
                                f"Compliant segment {segment.name} stopped by {stop_reason}; {detail}; aborting plan.",
                                segment=segment,
                                log=log + [entry],
                                result=result,
                            )
                        _assert_not_stopped(rtde_robot, segment, log)
                else:   # 使用位控
                    path = [np.asarray(conf, dtype=float).reshape(-1) for conf in segment.path]
                    entry["executed_as"] = "move_jntspace_path"
                    entry["waypoints"] = len(path)
                    entry["move_to_start"] = True
                    if not dry_run:
                        if not path:
                            raise ValueError(
                                f"Segment {segment.name} has no joint path; it requires "
                                f"use_move_l_compliant=True to run as force-controlled motion"
                            )
                        _check_start_joint_alignment(rtde_robot, segment, path[0], max_start_joint_error, log)
                        if prepend_actual_joint:
                            path[0] = np.asarray(rtde_robot.get_jnt_values(), dtype=float).reshape(-1)
                        path_kwargs = _jntspace_kwargs_for_segment(segment, jntspace_kwargs, jntspace_kwargs_by_segment)
                        entry["jntspace_kwargs"] = dict(path_kwargs)
                        rtde_robot.move_jntspace_path(path, **path_kwargs)
                        _assert_not_stopped(rtde_robot, segment, log)
            elif segment.command == "moveL":
                entry["pose"] = segment.pose    # type:ignore
                entry["vel"] = segment.vel  # type:ignore
                entry["acc"] = segment.acc  # type:ignore
                if not dry_run:
                    _assert_not_stopped(rtde_robot, segment, log)
                    if segment.pose is None:
                        raise ValueError(f"Segment {segment.name} (moveL) requires a target pose")
                    target_pose = np.asarray(segment.pose, dtype=float).reshape(6).tolist()
                    vel = float(segment.vel if segment.vel is not None else MOVE_L_DEFAULT_VEL)
                    acc = float(segment.acc if segment.acc is not None else MOVE_L_DEFAULT_ACC)
                    rtde_robot.moveL(target_pose, vel, acc, wait=True)
                    _assert_not_stopped(rtde_robot, segment, log)
            elif segment.command == "close_gripper_to":
                if segment.jaw_width is None:
                    raise ValueError(f"Segment {segment.name} requires jaw_width")
                entry["jaw_width"] = float(segment.jaw_width)
                if not dry_run:
                    rtde_robot.close_gripper_to(float(segment.jaw_width))
                    _assert_not_stopped(rtde_robot, segment, log)
            elif segment.command == "close_gripper":
                entry["jaw_width"] = segment.jaw_width
                if not dry_run:
                    rtde_robot.close_gripper()
                    _assert_not_stopped(rtde_robot, segment, log)
            elif segment.command == "open_gripper":
                if not dry_run:
                    rtde_robot.open_gripper()
                    _assert_not_stopped(rtde_robot, segment, log)
            else:
                raise ValueError(f"Unsupported RTDE command: {segment.command}")
        except RtdeExecutionError:
            _try_stop_rtde_motion(rtde_robot)
            raise
        except Exception as exc:
            _try_stop_rtde_motion(rtde_robot)
            raise RtdeExecutionError(
                f"Segment {segment.name} failed with {type(exc).__name__}: {exc}",
                segment=segment,
                log=log + [entry],
            ) from exc
        log.append(entry)
    return log




