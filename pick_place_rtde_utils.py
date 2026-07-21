"""Utilities for turning WRS pick-and-place motion into RTDE execution steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


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
}


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


def _find_pre_pick_index_by_tcp_path(tcp_positions: np.ndarray, pick_index: int, approach_distance: float) -> int:
    if pick_index <= 0:
        return 0
    target_distance = max(float(approach_distance), 0.0)
    if target_distance <= 1e-6:
        return max(0, pick_index - 1)
    accumulated = 0.0
    for index in range(pick_index, 0, -1):
        step = float(np.linalg.norm(tcp_positions[index] - tcp_positions[index - 1]))
        accumulated += step
        if accumulated >= target_distance:
            return index - 1
    return 0


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
) -> RtdeExecutionPlan:
    """Split WRS MotionData into RTDE-friendly execution segments.

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
    pre_pick_index = _find_pre_pick_index_by_tcp_path(tcp_positions_world, pick_index, approach_distance)
    if pre_pick_index > pick_index:
        pre_pick_index = pick_index

    pre_pick_tcp_pos_world = tcp_positions_world[pre_pick_index]
    pick_path_tcp_pos_world = tcp_positions_world[pick_index]
    pre_pick_tcp_pos_base = tcp_positions_base[pre_pick_index]
    pick_path_tcp_pos_base = tcp_positions_base[pick_index]
    approach_delta_base = pick_path_tcp_pos_base - pre_pick_tcp_pos_base
    compliant_distance = float(np.linalg.norm(approach_delta_base))
    if compliant_distance <= 1e-6:
        approach_direction_base = _unit_vector(base_rotmat_world.T @ np.asarray(pick_tcp_rotmat[:, 2], dtype=float))
        compliant_distance = approach_distance
    else:
        approach_direction_base = approach_delta_base / compliant_distance
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
    if is_push_path:
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

    if not is_push_path:
        segments.append(
            RtdeMotionSegment(
                name="set_pick_approach_jaw_width",
                command="close_gripper_to",
                start_index=pre_pick_index,
                end_index=pre_pick_index,
                jaw_width=pick_approach_jaw_width,
                metadata={"grasp_jaw_width": grasp_jaw_width},
            )
        )
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
                "direction_source": "joint_path_fk_robot_base",
                "planned_start_tcp": pre_pick_tcp_pos_base.tolist(),
                "planned_goal_tcp": pick_path_tcp_pos_base.tolist(),
                "planned_start_tcp_base": pre_pick_tcp_pos_base.tolist(),
                "planned_goal_tcp_base": pick_path_tcp_pos_base.tolist(),
                "planned_start_tcp_world": pre_pick_tcp_pos_world.tolist(),
                "planned_goal_tcp_world": pick_path_tcp_pos_world.tolist(),
                "approach_direction_world": approach_direction_world.tolist(),
                "robot_base_pos_world": base_pos_world.tolist(),
                "robot_base_rotmat_world": base_rotmat_world.tolist(),
            },
        )
    )

    if not is_push_path:
        segments.append(
            RtdeMotionSegment(
                name="close_gripper",
                command="close_gripper",
                start_index=close_index,
                end_index=close_index,
                jaw_width=grasp_jaw_width,
            )
        )

    post_pick_start = min(close_index, len(jv_list) - 1)
    if open_index is not None and open_index > post_pick_start:
        segments.append(
            RtdeMotionSegment(
                name="transfer_to_place",
                command="move_jntspace_path",
                start_index=post_pick_start,
                end_index=open_index,
                path=_slice_path(jv_list, post_pick_start, open_index),
            )
        )
        segments.append(
            RtdeMotionSegment(
                name="open_gripper",
                command="open_gripper",
                start_index=open_index,
                end_index=open_index,
            )
        )
        if open_index < len(jv_list) - 1:
            segments.append(
                RtdeMotionSegment(
                    name="depart_after_place",
                    command="move_jntspace_path",
                    start_index=open_index,
                    end_index=len(jv_list) - 1,
                    path=_slice_path(jv_list, open_index, len(jv_list) - 1),
                )
            )
    elif post_pick_start < len(jv_list) - 1:
        segments.append(
            RtdeMotionSegment(
                name="after_pick_motion",
                command="move_jntspace_path",
                start_index=post_pick_start,
                end_index=len(jv_list) - 1,
                path=_slice_path(jv_list, post_pick_start, len(jv_list) - 1),
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
                    _assert_not_stopped(rtde_robot, segment, log)
            elif segment.command == "moveL_compliant":
                entry["direction"] = segment.direction
                entry["distance"] = segment.distance
                entry["metadata"] = dict(segment.metadata)
                if use_move_l_compliant:
                    kwargs = _move_l_compliant_kwargs(segment.metadata, compliant_kwargs)
                    entry["executed_as"] = "moveL_compliant"
                    if not dry_run:
                        result = rtde_robot.moveL_compliant(segment.direction, segment.distance, **kwargs)
                        entry["result"] = result
                        stop_reason = str(result.get("stop_reason", "unknown"))
                        if stop_reason not in allowed_compliant_stop_reasons:
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
                else:
                    path = [np.asarray(conf, dtype=float).reshape(-1) for conf in segment.path]
                    entry["executed_as"] = "move_jntspace_path"
                    entry["waypoints"] = len(path)
                    entry["move_to_start"] = True
                    if not dry_run:
                        if not path:
                            raise ValueError(f"Segment {segment.name} has no joint path for non-compliant execution")
                        _check_start_joint_alignment(rtde_robot, segment, path[0], max_start_joint_error, log)
                        if prepend_actual_joint:
                            path[0] = np.asarray(rtde_robot.get_jnt_values(), dtype=float).reshape(-1)
                        path_kwargs = _jntspace_kwargs_for_segment(segment, jntspace_kwargs, jntspace_kwargs_by_segment)
                        entry["jntspace_kwargs"] = dict(path_kwargs)
                        rtde_robot.move_jntspace_path(path, **path_kwargs)
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




