from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Optional
import numpy as np
import open3d as o3d
from wrs import mcm, mgm, ppp

REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from yanjiuyuan.constants import (  # noqa: E402
    BOTTLE_ROBOT_SIDE_PLACE_POS,
    BOTTLE_ROBOT_SIDE_PLACE_POSE_pos,
    PICK_APPROACH_DEPART_DISTANCE,
    PICK_LIFT_MAX_Z,
    REAL_PIPELINE_CONFIG,
)
from yanjiuyuan import sim_pick_and_place as sim_pick  # noqa: E402
from yanjiuyuan import pick_place_rtde_utils as rtde_utils  # noqa: E402
from yanjiuyuan.box_collision import (  # noqa: E402
    make_concave_box_collision_obstacles,
    make_detected_box_visual_model,
)
from types1 import ObjectIcpResult, PlanningResult, RtdeObjectPose
import utils

# 真实机器人抓取路径规划

# 放置位置箱碰撞体
def make_robot_side_place_box_collision_obstacle(show_cdprim: bool = False):
    place_pos = np.asarray(BOTTLE_ROBOT_SIDE_PLACE_POS, dtype=float).reshape(3)
    box_sgm = mgm.gen_box(
        xyz_lengths=np.array([0.20, 0.2, 0.1], dtype=float),    # 箱体尺寸
        rgb=np.array([1.0, 0.62, 0.2], dtype=float),    # 颜色
        alpha=0.28, # 不透明度
    )
    place_box = mcm.CollisionModel(
        box_sgm,
        name="robot_side_place_box_collision",
        cdprim_type=mcm.const.CDPrimType.AABB,
        ex_radius=0.005,
        rgb=np.array([1.0, 0.62, 0.2], dtype=float), # 颜色
        alpha=0.28, # 不透明度
    )
    place_box._name = "robot_side_place_box_collision"
    place_box.pose = (place_pos, np.eye(3))
    place_box.show_cdprim()
    return place_box

# 障碍列表
def build_obstacle_lists(
    args: SimpleNamespace,
    box_homomat: np.ndarray,
    include_display: bool = True,
) -> tuple[list[object], list[object]]:
    planning_obstacles: list[object] = []
    display_obstacles: list[object] = []
    if not args.no_env:
        table = sim_pick.make_table_obstacle()
        planning_obstacles.append(table)
        if include_display:
            display_obstacles.append(table)
    # 设置箱子碰撞体
    detected_box_panels = make_concave_box_collision_obstacles(box_homomat, show_cdprim=include_display)
    if include_display:
        display_obstacles.append(make_detected_box_visual_model(box_homomat))
        display_obstacles.extend(detected_box_panels)
    if args.box_obstacle:
        planning_obstacles.extend(detected_box_panels)
    # 放置位置箱体
    robot_side_place_box = make_robot_side_place_box_collision_obstacle(show_cdprim=include_display)
    # robot_side_place_box.attach_to(base)
    planning_obstacles.append(robot_side_place_box)
    if include_display:
        display_obstacles.append(robot_side_place_box)
    return planning_obstacles, display_obstacles

#  碰撞预检
def precheck_start_collision(robot, obstacle_list: list[object]) -> None:
    if not obstacle_list:
        return
    robot.backup_state()
    try:
        collided = bool(robot.is_collided(obstacle_list=obstacle_list, toggle_contacts=False))
    finally:
        robot.restore_state()
    if collided:
        names = [getattr(obstacle, "name", type(obstacle).__name__) for obstacle in obstacle_list]
        sim_pick.debug_print(
            "WARNING: start configuration is already in collision with planning obstacles "
            f"{names}. If this offline default is not the real robot start, pass --start-conf-deg."
        )

# 规划器全局配置
def configure_pick_place(args: SimpleNamespace, object_model_path: Path) -> None:
    sim_pick.IK_SOLVER = "ikfast"
    sim_pick.OBJECT_MODEL_PATH = object_model_path
    sim_pick.GRASP_PICKLE_PATH = Path(args.grasp_pickle)
    sim_pick.USE_RRT = bool(args.use_rrt)
    sim_pick.VISUALIZE_RESULT = not bool(args.dry_run)
    sim_pick.VISUALIZE_FAILURE = bool(args.visualize_failure)
    sim_pick.PRINT_WRS_SUMMARY = True
    sim_pick.DEBUG_IKFAST_FRAME = False
    if args.open_jaw is not None:
        sim_pick.OPEN_JAW_WIDTH = float(args.open_jaw)
    if args.approach_distance is not None:
        sim_pick.APPROACH_DISTANCE = float(args.approach_distance)
    if args.start_conf_deg is not None:
        sim_pick.DEFAULT_HOME_CONF = np.radians(np.asarray(args.start_conf_deg, dtype=float))
# 夹角
def xy_angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    vector_a = np.asarray(vector_a, dtype=float).reshape(2)
    vector_b = np.asarray(vector_b, dtype=float).reshape(2)
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return float("inf")
    cosine = float(np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))

# 基座位置
def robot_arm_base_pos(robot) -> np.ndarray:
    arm = getattr(robot, "arm", getattr(robot, "manipulator", None))
    if arm is None:
        return np.zeros(3)
    return np.asarray(getattr(arm, "pos", np.zeros(3)), dtype=float).reshape(3)

# 关节范围
def first_joint_range(robot) -> tuple[float, float]:
    ranges = getattr(robot, "jnt_ranges", None)
    ranges = ranges() if callable(ranges) else ranges
    if ranges is None:
        arm = getattr(robot, "arm", getattr(robot, "manipulator", None))
        ranges = getattr(arm, "jnt_ranges", None) if arm is not None else None
    ranges = np.asarray(ranges, dtype=float)
    if ranges.ndim != 2 or ranges.shape[0] < 1 or ranges.shape[1] < 2:
        raise RuntimeError("Could not read first joint range for post-pick alignment.")
    lower, upper = float(ranges[0, 0]), float(ranges[0, 1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise RuntimeError(f"Invalid first joint range for post-pick alignment: {ranges[0].tolist()}")
    return lower, upper

# 给定构型下，TCP−基座 的 XY 方向与目标 XY 方向夹角。
def tcp_base_xy_angle_for_conf(robot, conf: np.ndarray, base_pos: np.ndarray, target_xy: np.ndarray) -> float:
    tcp_pos, _tcp_rotmat = robot.fk(jnt_values=np.asarray(conf, dtype=float))
    tcp_vector_xy = (np.asarray(tcp_pos, dtype=float).reshape(3) - base_pos)[:2]
    return xy_angle_deg(tcp_vector_xy, target_xy)

# 沿世界 -Z 下降一段（抓取后、开爪前的下沉，绕过碰撞）。
def make_world_z_lowering_path(robot, start_conf: np.ndarray) -> tuple[list[np.ndarray], float]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    lowering_distance = float(PICK_APPROACH_DEPART_DISTANCE) * 0.75
    step_count = max(1, int(np.ceil(lowering_distance / max(float(sim_pick.LINEAR_GRANULARITY), 1e-4))))
    start_tcp_pos, start_tcp_rotmat = robot.fk(jnt_values=start_conf)
    start_tcp_pos = np.asarray(start_tcp_pos, dtype=float).reshape(3)
    start_tcp_rotmat = np.asarray(start_tcp_rotmat, dtype=float).reshape(3, 3)
    lowering_path: list[np.ndarray] = []
    seed_conf = start_conf
    for step_id in range(1, step_count + 1):
        ratio = float(step_id) / float(step_count)
        target_pos = start_tcp_pos - sim_pick.rm.const.z_ax * lowering_distance * ratio
        # print("target_pos:", target_pos)
        jnt_values = robot.ik(
            tgt_pos=target_pos,
            tgt_rotmat=start_tcp_rotmat,
            seed_jnt_values=seed_conf,
        )
        if jnt_values is None:
            return [], lowering_distance
        jnt_values = np.asarray(jnt_values, dtype=float).reshape(-1)
        lowering_path.append(jnt_values)
        seed_conf = jnt_values
    return lowering_path, lowering_distance

# 沿指定世界方向做直线 TCP 插补（每步 IK，失败返回空）
def make_world_linear_tcp_path(
    robot,
    start_conf: np.ndarray,
    direction: np.ndarray,
    distance: float,
    linear_granularity: Optional[float] = None,
) -> tuple[list[np.ndarray], float]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-9:
        raise ValueError("World linear TCP direction cannot be zero.")
    direction = direction / direction_norm
    distance = float(distance)
    if distance <= 0:
        raise ValueError(f"World linear TCP distance must be positive, got {distance}.")
    granularity = float(sim_pick.LINEAR_GRANULARITY if linear_granularity is None else linear_granularity)
    step_count = max(1, int(np.ceil(distance / max(granularity, 1e-4))))
    start_tcp_pos, start_tcp_rotmat = robot.fk(jnt_values=start_conf)
    start_tcp_pos = np.asarray(start_tcp_pos, dtype=float).reshape(3)
    start_tcp_rotmat = np.asarray(start_tcp_rotmat, dtype=float).reshape(3, 3)
    path: list[np.ndarray] = []
    seed_conf = start_conf
    for step_id in range(1, step_count + 1):
        ratio = float(step_id) / float(step_count)
        target_pos = start_tcp_pos + direction * distance * ratio
        jnt_values = robot.ik(
            tgt_pos=target_pos,
            tgt_rotmat=start_tcp_rotmat,
            seed_jnt_values=seed_conf,
        )
        if jnt_values is None:
            return [], distance
        jnt_values = np.asarray(jnt_values, dtype=float).reshape(-1)
        path.append(jnt_values)
        seed_conf = jnt_values
    return path, distance

# 沿世界 +Z 竖直上抬一段（离开/抬升用），是上面直线路径的特例。
def make_world_z_lift_path(robot, start_conf: np.ndarray, distance: float) -> tuple[list[np.ndarray], float]:
    return make_world_linear_tcp_path(
        robot,
        start_conf=start_conf,
        direction=sim_pick.rm.const.z_ax,
        distance=distance,
    )

# 直线移动到指定 TCP 目标位置
def make_world_tcp_position_path(
    robot,
    start_conf: np.ndarray,
    target_pos: np.ndarray,
    linear_granularity: Optional[float] = None,
) -> tuple[list[np.ndarray], float]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    start_tcp_pos, _start_tcp_rotmat = robot.fk(jnt_values=start_conf)
    start_tcp_pos = np.asarray(start_tcp_pos, dtype=float).reshape(3)
    delta = target_pos - start_tcp_pos
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        return [], 0.0
    return make_world_linear_tcp_path(
        robot,
        start_conf=start_conf,
        direction=delta,
        distance=distance,
        linear_granularity=linear_granularity,
    )

# 直接在关节空间按步距插值（不做事检）
def make_joint_interpolation_path(
    start_conf: np.ndarray,
    goal_conf: np.ndarray,
    path_step_deg: float = 2.0,
) -> list[np.ndarray]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    goal_conf = np.asarray(goal_conf, dtype=float).reshape(-1)
    if start_conf.shape != goal_conf.shape:
        raise ValueError(f"Joint interpolation shape mismatch: {start_conf.shape} vs {goal_conf.shape}.")
    max_delta = float(np.max(np.abs(goal_conf - start_conf))) if start_conf.size else 0.0
    if max_delta < 1e-9:
        return []
    step = max(float(np.radians(path_step_deg)), 1e-4)
    step_count = max(1, int(np.ceil(max_delta / step)))
    return [np.asarray(conf, dtype=float).reshape(-1) for conf in np.linspace(start_conf, goal_conf, step_count + 1)[1:]]

# 抓取后把第 1 关节逆时针转到使「TCP-基座」XY 方向与「基座→放置箱」方向夹角 ≤ max_angle_deg（默认 5°），让机器人朝向放置侧。
def make_first_joint_alignment_path(
    robot,
    start_conf: np.ndarray,
    ee_value,
    obstacle_list: list[object],
    max_angle_deg: float = 5.0,
    sample_step_deg: float = 1.0,
    path_step_deg: float = 2.0,
) -> tuple[list[np.ndarray], float]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    base_pos = robot_arm_base_pos(robot)
    target_xy = (np.asarray(BOTTLE_ROBOT_SIDE_PLACE_POS, dtype=float).reshape(3) - base_pos)[:2]
    if np.linalg.norm(target_xy) < 1e-9:
        raise RuntimeError("robot-base-to-BOTTLE_ROBOT_SIDE_PLACE_POS xy vector is too small for post-pick alignment.")
    lower, upper = first_joint_range(robot)
    sample_step = max(float(np.radians(sample_step_deg)), 1e-4)
    sample_count = int(np.ceil((upper - lower) / sample_step)) + 1
    sample_count = max(2, min(sample_count, 2000))
    sampled_q0 = np.linspace(lower, upper, sample_count)
    sampled_q0 = np.unique(np.concatenate(([start_conf[0]], sampled_q0)))

    candidates: list[tuple[float, float, float]] = []
    robot.backup_state()
    try:
        for q0 in sampled_q0:
            counterclockwise_delta = float(q0) - float(start_conf[0])
            if counterclockwise_delta < -1e-9:
                continue
            conf = start_conf.copy()
            conf[0] = float(q0)
            angle_deg = tcp_base_xy_angle_for_conf(robot, conf, base_pos, target_xy)
            if angle_deg <= max_angle_deg:
                candidates.append((counterclockwise_delta, angle_deg, float(q0)))
    finally:
        robot.restore_state()

    if not candidates:
        raise RuntimeError(
            f"No counterclockwise first-joint angle makes TCP-base XY direction within {max_angle_deg:.1f} deg "
            f"of robot-base-to-BOTTLE_ROBOT_SIDE_PLACE_POS xy={target_xy.tolist()}."
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    path_step = max(float(np.radians(path_step_deg)), 1e-4)
    for _delta, angle_deg, goal_q0 in candidates:
        delta = float(goal_q0) - float(start_conf[0])
        if abs(delta) < 1e-6:
            return [], angle_deg
        step_count = max(1, int(np.ceil(abs(delta) / path_step)))
        q0_values = np.linspace(float(start_conf[0]), goal_q0, step_count + 1)[1:]
        path = []
        for q0 in q0_values:
            conf = start_conf.copy()
            conf[0] = float(q0)
            path.append(conf)
        return path, angle_deg

# 用抓取位姿重建物体碰撞模型
def make_fresh_pick_object_model(object_model_path: Path, pick_pose: tuple[np.ndarray, np.ndarray], name: str):
    return sim_pick.make_object_model(object_model_path, pick_pose, name=name)

# 抓取运动规划核心
def gen_pick_only_path(robot, object_model_path: Path, grasps, pick_pose, place_pose, obstacle_list, args: SimpleNamespace, remaining_pcd: Optional["o3d.geometry.PointCloud"] = None) -> tuple[Optional[int], Any]:
    """
        做周围点云筛选 → 筛选手爪姿态 → 调 PickPlacePlanner.gen_pick_approach_only
        生成"接近段"→ 接 下沉/第1轴对齐/抬升 等后处理，返回 (selected_grasp_index, mot_data)
    """
    _t_total_start = time.time()
    _t_reason: float = 0.0
    _t_pcd_filter: float = 0.0
    _t_path_plan: float = 0.0
    _t_post_pick: float = 0.0

    sim_pick.debug_print("Starting pick-only planner...")
    sim_pick.debug_print(f"  grasps={len(grasps)}, use_rrt={sim_pick.USE_RRT}")
    pick_approach_distance = float(sim_pick.APPROACH_DISTANCE)
    pick_approach_distance_source = "default"
    sim_pick.debug_print(
        f"  pick approach distance={pick_approach_distance:.3f} m ({pick_approach_distance_source}), "
        f"depart distance={PICK_APPROACH_DEPART_DISTANCE:.3f} m, "
        f"linear_granularity={sim_pick.LINEAR_GRANULARITY:.3f} m"
    )
    sim_pick.debug_print("  directions: pick approach=grasp TCP +Z, pick depart=+Z")

    planner = ppp.PickPlacePlanner(robot)   # 规划器

    grasp_collection = sim_pick.make_grasp_collection(robot, grasps)
    # ---- 获取 remaining point cloud 周围点云 ----
    # 优先使用调用方传入的内存点云（已按物体序号存于 ctx.remaining_pcd_by_obj），
    # 避免重新读盘；仅在未传入时才回退从磁盘读取 args.remaining_pointcloud_path。
    if remaining_pcd is None:
        remaining_ply_path = getattr(args, "remaining_pointcloud_path", None)
        if remaining_ply_path is None:
            # 从 output_dir 直接找
            output_dir = getattr(args, "output_dir", None) or getattr(args, "object_output_dir", None)
            if output_dir is not None:
                candidate = Path(output_dir) / "remaining_pointcloud.ply"
                if candidate.exists():
                    remaining_ply_path = str(candidate)
        if remaining_ply_path is None or not Path(remaining_ply_path).exists():
            sim_pick.debug_print(" 周围点云不存在（内存与磁盘均无）.")
            remaining_pcd = None
        else:   # 周围点云存在，可以筛选
            sim_pick.debug_print(" 周围点云存在（从磁盘读取）.")
            remaining_pcd = o3d.io.read_point_cloud(remaining_ply_path)
    else:
        sim_pick.debug_print(" 周围点云存在（使用调用方传入的内存点云，免读盘）.")
    if remaining_pcd is not None:
        _t0 = time.time()
        candidate_indices = planner.filter_grasps_sequence(
                            grasp_collection,
                [pick_pose],
                            obstacle_list,
                            remaining_pcd,  # 周围点云
                            robot,
                            robot.end_effector,
                            log_func=sim_pick.debug_print,
                            toggle_dbg=False,
        )
        _t_reason = time.time() - _t0
        sim_pick.debug_print(f"  手爪姿态筛选后pick grasps result: {len(candidate_indices)}/{len(grasps)} ({_t_reason:.3f}s)")
        if candidate_indices:
            preview = candidate_indices[:20]
            suffix = "..." if len(candidate_indices) > len(preview) else ""
            sim_pick.debug_print(f"  pick grasp indices: {preview}{suffix}")
        else:   # 没有合适的抓取
            sim_pick.debug_print(" 没有合适的 pick 抓取，返回失败信号（无可行抓取）。")
            return None, None
    else:
        candidate_indices = list(range(len(grasps)))

    pick_approach_jaw_width_source = f"default +0.015m grasp jaw"

    for grasp_index in candidate_indices:
        grasp = grasps[grasp_index]
        grasp_width = grasp_jaw_width(grasp)
        pick_approach_jaw_width = pick_approach_jaw_width_for_grasp(robot, grasp)
        jaw_text = f"pick approach jaw={pick_approach_jaw_width:.5f} m ({pick_approach_jaw_width_source})"
        robot.goto_given_conf(sim_pick.DEFAULT_HOME_CONF, ee_values=pick_approach_jaw_width)
        sim_pick.debug_print(
            f"  pickle_grasp_{grasp_index}: grasp jaw={grasp_width:.5f} m, "
            f"{jaw_text}"
        )
        _t0 = time.time()
        # 仅生成“从起始位姿接近到抓取点”的路径（不含夹爪闭合/抬起/搬运）
        mot_data = planner.gen_pick_approach_only(
            obj_cmodel=make_fresh_pick_object_model(
                object_model_path,
                pick_pose,
                name=f"icp_bottle_pick_place_grasp_{grasp_index}",
            ),
            grasp=grasp,
            start_jnt_values=sim_pick.DEFAULT_HOME_CONF,
            pick_approach_jaw_width=pick_approach_jaw_width, # 接近过程中夹爪张开宽度（默认张到最大）
            pick_approach_direction=None,
            pick_approach_distance=pick_approach_distance,
            linear_granularity=sim_pick.LINEAR_GRANULARITY,
            obstacle_list=obstacle_list,
            use_rrt=sim_pick.USE_RRT,   # 使用rrt避障
            toggle_dbg=False,
        )
        _t_path_plan += time.time() - _t0
        if mot_data is None:
            sim_pick.debug_print(f"  pickle_grasp_{grasp_index}: pick approach-only failed")
            continue
        pick_frame_count = len(mot_data.jv_list) if hasattr(mot_data, "jv_list") else len(mot_data)
        sim_pick.debug_print(f"Pick approach-only produced {pick_frame_count} frame(s).")

        _t0 = time.time()
        # ---- 抓取点闭合夹爪（拿起物体）----
        closed_width = sim_pick.clamp_jaw_width(robot, grasp_width)
        grasp_conf = np.asarray(mot_data.jv_list[-1], dtype=float).copy()
        mot_data.extend(
            [grasp_conf],
            ev_list=[closed_width],
            mesh_list=[],
        )
        sim_pick.debug_print(f"  Closed gripper at grasp point: jaw={closed_width:.5f} m")

        # ---- 三段搬运以原生 moveL 执行（规划侧不再生成关节直线轨迹）----
        # 段1: 竖直 +Z 抬起离开；段2: 水平移动到放置点正上方；段3: 竖直下落到放置点。
        # 规划侧只校验航点 IK 可达性并记录放置点关节角；RTDE 拆分器用
        # transfer_movel_waypoints_world 生成原生 moveL 段（rtde_robot.moveL）执行。
        start_tcp_pos, _start_tcp_rotmat = robot.fk(jnt_values=grasp_conf)
        start_tcp_pos = np.asarray(start_tcp_pos, dtype=float).reshape(3)
        depart_distance = float(PICK_APPROACH_DEPART_DISTANCE)
        lifted_pos = start_tcp_pos + sim_pick.rm.const.z_ax * depart_distance
        lifted_pos[2] = min(lifted_pos[2], PICK_LIFT_MAX_Z) # 限制高度，以免过高到不了
        place_pos = np.asarray(place_pose[0], dtype=float).reshape(3)
        above_place_pos = np.array([place_pos[0], place_pos[1], min(lifted_pos[2],0.63)], dtype=float)
        waypoints = [lifted_pos, above_place_pos, place_pos]
        seg_labels = ["lift +Z (depart)", "move above place (XY)", "lower to place"]
        cur_conf = grasp_conf
        seg_total = 0.0
        move_ok = True
        # ---- 不规划关节直线轨迹：仅校验三个航点（保持抓取点姿态）的 IK 可达性，
        #      并记录放置点关节角供拆分器定位 open 事件；笛卡尔直线由 RTDE 原生 moveL 执行。----
        _chk_start_pos, _chk_start_rotmat = robot.fk(jnt_values=grasp_conf)
        _chk_start_rotmat = np.asarray(_chk_start_rotmat, dtype=float).reshape(3, 3)
        _seed = np.asarray(grasp_conf, dtype=float).reshape(-1)
        place_conf = None
        for wp, label in zip(waypoints, seg_labels):
            _jnt = robot.ik(
                tgt_pos=np.asarray(wp, dtype=float).reshape(3),
                tgt_rotmat=_chk_start_rotmat,
                seed_jnt_values=_seed,
            )
            if _jnt is None:
                sim_pick.debug_print(
                    f"  pickle_grasp_{grasp_index}: moveL waypoint '{label}' IK not solvable"
                )
                move_ok = False
                break
            _jnt = np.asarray(_jnt, dtype=float).reshape(-1)
            _seed = _jnt
            if label == "lower to place":
                place_conf = _jnt.copy()
        if not move_ok or place_conf is None:
            sim_pick.debug_print(
                f"  pickle_grasp_{grasp_index}: moveL waypoints unreachable, try next grasp"
            )
            continue
        seg_total = float(
            sum(
                np.linalg.norm(np.asarray(waypoints[i + 1], dtype=float) - np.asarray(waypoints[i], dtype=float))
                for i in range(len(waypoints) - 1)
            )
        )
        sim_pick.debug_print(
            f"  moveL waypoints OK (no joint-path planning); total distance={seg_total:.4f} m, "
            "executed as native moveL downstream"
        )
        cur_conf = place_conf

        # ---- 记录三段搬运为显式 moveL 航点（供 RTDE 拆分器原生 moveL 执行）----
        # 这三个 world 系航点对应 "lift +Z (depart)" / "move above place (XY)" / "lower to place"。
        # 拆分器会优先采用，避免依赖路径方向自动检测，保证三段确实以 UR 原生 moveL 执行。
        setattr(
            mot_data,
            "transfer_movel_waypoints_world",
            [np.asarray(w, dtype=float).reshape(3).copy() for w in waypoints],
        )
        setattr(mot_data, "transfer_movel_use_move_l", True)
        # 上移距离（抓取点竖直抬离桌面的距离），供 RTDE 拆分器与执行端
        # 以【实际 TCP 位姿】为基准生成原生物料 moveL 时使用。
        setattr(mot_data, "transfer_movel_depart_distance", depart_distance)

        # ---- 三段全部成功：在放置点张爪释放 ----
        final_open_width = sim_pick.clamp_jaw_width(robot, sim_pick.OPEN_JAW_WIDTH)
        mot_data.extend(
            [np.asarray(cur_conf, dtype=float).copy()],
            ev_list=[final_open_width],
            mesh_list=[],
        )
        setattr(mot_data, "force_final_open_gripper", True)
        setattr(mot_data, "forced_open_index", len(mot_data.jv_list) - 1)
        sim_pick.debug_print(
            f"  Released gripper at place pose: jaw={final_open_width:.5f} m "
            f"(total moveL distance={seg_total:.4f} m)"
        )

        _t_post_pick += time.time() - _t0

        frame_count = len(mot_data.jv_list) if hasattr(mot_data, "jv_list") else len(mot_data)
        sim_pick.debug_print(
            f"Pick planner produced {frame_count} frame(s): approach + close + "
            "3-segment straight moveL to place + release."
        )
        sim_pick.debug_print("Pick planner succeeded.")
        sim_pick.debug_print(f"  selected grasp: pickle_grasp_{grasp_index}")
        _t_total = time.time() - _t_total_start
        sim_pick.debug_print(
            f"[planning_timing] filter_grasps_sequence={_t_reason:.3f}s, "
            f"gen_pick_and_place_linear={_t_path_plan:.3f}s, "
            f"post_pick={_t_post_pick:.3f}s, "
            f"total={_t_total:.3f}s"
        )
        return grasp_index, mot_data

    sim_pick.debug_print(
        "Pick planner failed: no grasp produced a valid pick approach/depart path. "
        "返回失败信号（无可行抓取）。"
    )
    return None, None

# 打印抓取规划结果
def print_pick_only_summary(mot_data, grasp_count: int, selected_grasp_index: int, pick_pose, obstacle_names: list[str]) -> None:
    sim_pick.debug_print("Pick-only planner path ready.")
    sim_pick.debug_print(f"Loaded grasps: {grasp_count}")
    if selected_grasp_index is None:
        sim_pick.debug_print("Selected grasp: unknown")
    else:
        sim_pick.debug_print(f"Selected grasp: pickle_grasp_{selected_grasp_index}")
    sim_pick.debug_print(f"Frames: {len(mot_data.jv_list)}")
    sim_pick.debug_print("Path type: pick_place_movel")
    if len(mot_data.jv_list) > 1:
        max_step = max(
            float(np.max(np.abs(np.asarray(mot_data.jv_list[i + 1]) - np.asarray(mot_data.jv_list[i]))))
            for i in range(len(mot_data.jv_list) - 1)
        )
        sim_pick.debug_print(f"Max adjacent joint step(deg): {np.degrees(max_step):.3f}")
    sim_pick.debug_print(f"Collision obstacles: {obstacle_names}")
    sim_pick.debug_print(f"Pick object pos(m): {sim_pick.format_vec(pick_pose[0], digits=6)}")
    sim_pick.debug_print(f"Start conf(deg): {sim_pick.format_jnts_deg(mot_data.jv_list[0], digits=3)}")
    sim_pick.debug_print(f"End conf(deg): {sim_pick.format_jnts_deg(mot_data.jv_list[-1], digits=3)}")

# 抓取规划总入口
def run_or_skip_plan(args: SimpleNamespace, icp: ObjectIcpResult, remaining_pcd: Optional["o3d.geometry.PointCloud"] = None) -> Optional[PlanningResult]:
    if args.skip_plan:
        return None

    object_model_path = args.object_model if args.object_model is not None else icp.bottle_model_path
    object_model_path = object_model_path.resolve()
    if not object_model_path.exists():
        raise FileNotFoundError(f"Object model not found: {object_model_path}")
    #
    bottle_homomat = bottle_homomat_for_icp(icp)
    box_homomat = utils.load_homomat(icp.box_transform_path, "box")
    pick_pose = utils.homomat_to_pose(bottle_homomat)
    place_pose, place_pos_source, place_rot_source = resolve_place_pose(args, pick_pose)

    configure_pick_place(args, object_model_path)
    if not sim_pick.GRASP_PICKLE_PATH.exists():
        raise FileNotFoundError(f"Grasp pickle not found: {sim_pick.GRASP_PICKLE_PATH}")
    action_sequence = getattr(args, "action_sequence", None)
    action_sequence = None if action_sequence is None else int(action_sequence)

    sim_pick.debug_print("=== real_bottle_pick_place_interactive: pick-only planner run ===")
    if action_sequence is None:
        sim_pick.debug_print("Action sequence: disabled")
    else:
        sim_pick.debug_print(f"Action sequence: {action_sequence}")
    sim_pick.debug_print(f"Object model: {object_model_path}")
    sim_pick.debug_print(f"Grasp pickle: {sim_pick.GRASP_PICKLE_PATH}")
    sim_pick.debug_print("Robot: UR7EDH76(enable_cc=True, ik_solver='ikfast')")
    sim_pick.debug_print(f"Pick pose from ICP: pos={sim_pick.format_vec(pick_pose[0], digits=6)}")
    sim_pick.debug_print(
        f"Place pose preview only: pos={sim_pick.format_vec(place_pose[0], digits=6)} "
        f"({place_pos_source}), rot={place_rot_source}; not used by pick-only planner"
    )

    robot = sim_pick.make_robot()
    sim_pick.debug_ikfast_frame_conversion(robot)
    grasps = sim_pick.load_grasps(robot, sim_pick.GRASP_PICKLE_PATH)
    sim_pick.debug_print(f"Loaded grasps: {len(grasps)}")

    obstacle_list, display_obstacle_list = build_obstacle_lists(args, box_homomat, include_display=True)
    obstacle_names = [getattr(obstacle, "name", type(obstacle).__name__) for obstacle in obstacle_list]
    display_obstacle_names = [getattr(obstacle, "name", type(obstacle).__name__) for obstacle in display_obstacle_list]
    sim_pick.debug_print(f"Collision obstacles: {obstacle_names}")
    sim_pick.debug_print(f"Visualization context: {display_obstacle_names}")
    precheck_start_collision(robot, obstacle_list)
    # 生成抓取
    selected_grasp_index, mot_data = gen_pick_only_path(
        robot,
        object_model_path,
        grasps,
        pick_pose,
        place_pose,
        obstacle_list,
        args,
        remaining_pcd=remaining_pcd,
    )
    if selected_grasp_index is None:
        if mot_data is None:
            # gen_pick_only_path 明确返回无可行抓取（未生成 mot_data）
            print("[real_pipeline] ⚠️ 未找到可行的 pick 抓取，跳过该物体规划")
            return None
        selected_grasp_index = infer_selected_grasp_index(robot, grasps, mot_data, pick_pose)
    print_pick_only_summary(mot_data, len(grasps), selected_grasp_index, pick_pose, obstacle_names)

    # 仿真预计算“抓取点真实法兰盘位姿”
    # 用规划得到的关节构型（抓取闭合帧）经仿真 FK + 标定推算真实位姿，确定性、不依赖读真实机器人。
    predicted_grasp_real_tcp = None
    try:
        _close_idx = find_close_index(mot_data)
        if _close_idx is not None:
            _grasp_conf = np.asarray(mot_data.jv_list[_close_idx], dtype=float).reshape(-1)
        else:
            # 无闭合事件时退回到起始构型
            _grasp_conf = np.asarray(mot_data.jv_list[0], dtype=float).reshape(-1)
        predicted_grasp_real_tcp = robot.get_real_tcp_pose_from_conf(_grasp_conf)
        print(f"[real_pipeline] 预计算抓取点真实法兰盘位姿: {np.round(predicted_grasp_real_tcp, 5).tolist()}")
    except Exception as _exc:
        print(f"[real_pipeline] ⚠️ 预计算抓取点真实位姿失败，运行时回退仿真航点: {_exc}")

    planning = PlanningResult(
        selected_grasp_index=selected_grasp_index,
        action_sequence=action_sequence,
        mot_data=mot_data,
        pick_pose=pick_pose,
        place_pose=place_pose,
        object_model_path=object_model_path,
        obstacle_names=obstacle_names,
        place_pos_source=place_pos_source,
        place_rot_source=place_rot_source,
        predicted_grasp_real_tcp=predicted_grasp_real_tcp,  # 抓取点实际法兰盘位姿
    )
    planning.rtde_plan, planning.rtde_plan_path = build_and_save_rtde_plan(args, planning, icp.output_dir)

    if sim_pick.VISUALIZE_RESULT:
        sim_pick.visualize_result(mot_data, pick_pose, place_pose, display_obstacle_list)

    return planning

# 推开规划
def plan_push(
    args: SimpleNamespace,
    push_pose: tuple[np.ndarray, np.ndarray],
    jaw_width: float,
    object_model_path: Optional[Any] = None,
    box_homomat: Optional[np.ndarray] = None,
) -> Optional[rtde_utils.RtdeExecutionPlan]:
    """为单个 push 位姿（来自 bottle_dh76_push.pickle）生成可直接执行的 push RTDE 计划。

    运动剖面（与抓取一致的 RRT 接近 + 力控接触，但改成“推开”而非“搬运放置”）：
        1. set_pick_approach_jaw_width  （接近前把夹爪设到接近宽度，与 RRT 避障假设一致）
        2. move_to_pre_pick            （RRT 路径规划接近）
        3. approach_pick_compliant     （力控接触，推开）
        4. close_gripper               （闭合手爪）
        5. push_to_center_compliant    （水平力控推向箱子中心；方向经 get_real_tcp_pose_sim
                                        换算到真实基系；box_homomat 为 None 或配置关闭时省略）
        6. open_gripper                （张开）
        7. push_leave_movel            （moveL 竖直上抬离开；存在第 5 段时执行端改用
                                        运行时实际 TCP 位姿锚定）

    Args:
        args: 运行时配置（no_env / box_obstacle 等字段会被读取；push 力控参数改从
            constants.py 的 REAL_PIPELINE_CONFIG 读取，不再读 args.compliant_*）。
        push_pose: (pos, rotmat)，机器人基坐标系下的 push TCP 位姿（来自 pickle）。
        jaw_width: push 位姿对应的夹爪宽度。
        object_model_path: 瓶子模型路径；缺省使用 sim_pick.OBJECT_MODEL_PATH。
        box_homomat: 箱子位姿（用于构建箱子碰撞体）；为 None 时退化为“桌面 + 放置箱体”。

    Returns:
        成功返回 RtdeExecutionPlan；RRT 接近路径生成失败返回 None。
    """
    from wrs import gg
    from wrs.motion.motion_data import MotionData

    object_model_path = Path(object_model_path) if object_model_path is not None else Path(sim_pick.OBJECT_MODEL_PATH)
    if not object_model_path.exists():
        raise FileNotFoundError(f"Object model not found: {object_model_path}")

    # 障碍列表：与抓取规划（run_or_skip_plan）完全一致 —— 复用 build_obstacle_lists：
    #   桌子(table) + 箱子壁(detected_box_panels, 受 args.box_obstacle 控制, 配置默认 True) + 放置箱体(robot_side_place_box)。
    # box_homomat 来自实时检测到的箱子位姿（run_push_phase 传入 ctx.box_transform）。
    # 目标瓶子（被推物体）则以 obj_cmodel 形式传给 gen_pick_approach_only（与抓取规划一致），不额外加入 obstacle_list。
    if box_homomat is not None:
        obstacle_list, _ = build_obstacle_lists(args, box_homomat, include_display=False)
    else:
        obstacle_list = []
        if not getattr(args, "no_env", False):
            obstacle_list.append(sim_pick.make_table_obstacle())
        obstacle_list.append(make_robot_side_place_box_collision_obstacle(show_cdprim=False))
    print(f"[real_pipeline] push 规划障碍数: {len(obstacle_list)}")

    robot = sim_pick.make_robot()
    precheck_start_collision(robot, obstacle_list)

    # 用 push 位姿构造抓取（GGrasp），与抓取规划同一套接口
    push_pos = np.asarray(push_pose[0], dtype=float).reshape(3)
    push_rotmat = np.asarray(push_pose[1], dtype=float).reshape(3, 3)
    grasp = gg.Grasp(ee_values=float(jaw_width), ac_pos=np.zeros(3), ac_rotmat=np.eye(3))
    decorate_grasp_for_rtde(robot, grasp, 0, args)
    pick_approach_jaw_width = pick_approach_jaw_width_for_grasp(robot, grasp)

    planner = ppp.PickPlacePlanner(robot)
    obj_cmodel = make_fresh_pick_object_model(object_model_path, (push_pos, push_rotmat), name="push_obj_model")

    # ===== 生成 push 路径：先 RRT 关节空间规划到“接近起点(standoff)”，再由力控段到位 =====
    # 与抓取路径完全同构（build_pick_place_rtde_plan 会把 standoff→contact 转成 moveL_compliant 力控到位），
    # 但不再依赖 gen_pick_approach_only 内部的“直线接近”——其逐点 IK 严格检查对部分 push 位姿会失败
    # （IK not solvable in gen_linear_motion）。这里仅对 standoff / contact 做单次 IK（通常可解），
    # RRT 段只要求起止两点可解即可，更鲁棒。
    approach_distance = float(sim_pick.APPROACH_DISTANCE)
    approach_dir = np.asarray(push_rotmat[:, 2], dtype=float).reshape(3)  # 工具 +Z，即沿推开方向
    standoff_pos = push_pos - approach_dir * approach_distance
    standoff_rotmat = push_rotmat

    robot.backup_state()
    try:
        robot.goto_given_conf(jnt_values=sim_pick.DEFAULT_HOME_CONF)
        standoff_jv = robot.ik(standoff_pos, standoff_rotmat,
                               seed_jnt_values=sim_pick.DEFAULT_HOME_CONF, toggle_dbg=False)
        if standoff_jv is None:
            print("[real_pipeline] ⚠️ push 规划失败：standoff 接近起点 IK 不可解（跳过该物体）")
            return None
        contact_jv = robot.ik(push_pos, push_rotmat,
                              seed_jnt_values=np.asarray(standoff_jv, dtype=float).reshape(-1), toggle_dbg=False)
        if contact_jv is None:
            print("[real_pipeline] ⚠️ push 规划警告：contact 单次 IK 不可解，退回以 standoff 作为接触标记"
                  "（力控段仍按 方向+距离 推进到实际接触点）")
            contact_jv = np.asarray(standoff_jv, dtype=float).reshape(-1)
        # RRT：起始 → standoff（与抓取规划一致，use_rrt）
        if sim_pick.USE_RRT:
            robot.change_ee_values(ee_values=pick_approach_jaw_width)
            start2standoff = planner.rrtc_planner.plan(
                start_conf=sim_pick.DEFAULT_HOME_CONF,
                goal_conf=np.asarray(standoff_jv, dtype=float).reshape(-1),
                obstacle_list=obstacle_list + [obj_cmodel],
                ext_dist=0.3,
                max_time=100,
                toggle_dbg=False,
            )
        else:
            start2standoff = planner.im_planner.gen_interplated_between_given_conf(
                start_jnt_values=sim_pick.DEFAULT_HOME_CONF,
                end_jnt_values=np.asarray(standoff_jv, dtype=float).reshape(-1),
                obstacle_list=obstacle_list + [obj_cmodel],
                ee_values=pick_approach_jaw_width,
                toggle_dbg=False,
            )
        if start2standoff is None:
            print("[real_pipeline] ⚠️ push 规划失败：RRT/插值 接近路径（start→standoff）未生成（跳过该物体）")
            return None
    finally:
        robot.restore_state()

    # 组装与 gen_approach 等价的 MotionData：RRT 段 + [contact] 帧。
    # 这样 build_pick_place_rtde_plan 会：move_to_pre_pick = RRT(start→standoff)，
    # approach_pick_compliant = 力控(standoff→contact，按 方向+距离 执行)，与抓取路径同构。
    mot_data = MotionData(robot)
    mot_data._jv_list = [np.asarray(q, dtype=float).reshape(-1) for q in start2standoff.jv_list] + \
                        [np.asarray(contact_jv, dtype=float).reshape(-1)]
    mot_data._ev_list = list(start2standoff.ev_list) + [float(pick_approach_jaw_width)]
    # 标记为 push 路径：生成“力控接触→闭合→竖直离开→张开”
    mot_data.path_type = "push"
    # 让 build_pick_place_rtde_plan 据此反推 standoff 起点与接触方向（与抓取规划一致）
    grasp.approach_distance = approach_distance
    # 力控接触→闭合→竖直离开→张开
    rtde_plan = rtde_utils.build_pick_place_rtde_plan(
        robot=robot,
        mot_data=mot_data,
        pick_pose=RtdeObjectPose(pos=push_pos, rotmat=push_rotmat),
        place_pose=RtdeObjectPose(pos=push_pos, rotmat=push_rotmat),
        grasp=grasp,
        planner_name="WRS push planner + first-joint alignment",
        # push 力控参数统一从 constants.py 的 REAL_PIPELINE_CONFIG（push_compliant_* 段）读取，
        # 不再依赖 args.compliant_*（那批键本未定义，运行期会 AttributeError）。
        compliant_force=float(REAL_PIPELINE_CONFIG.get("push_compliant_force", 40.0)),
        compliant_vel=float(REAL_PIPELINE_CONFIG.get("push_compliant_vel", 0.06)),
        compliant_lateral_tolerance=float(REAL_PIPELINE_CONFIG.get("push_compliant_lateral_tolerance", 0.01)),
        compliant_lateral_stop_tolerance=REAL_PIPELINE_CONFIG.get("push_compliant_lateral_stop_tolerance", "auto"),
        compliant_force_frame=REAL_PIPELINE_CONFIG.get("push_compliant_force_frame", "direction"),
        compliant_axes=REAL_PIPELINE_CONFIG.get("push_compliant_axes", None),
        compliant_zero_ft_sensor=bool(REAL_PIPELINE_CONFIG.get("push_compliant_zero_ft_sensor", True)),
        compliant_max_tcp_force=float(REAL_PIPELINE_CONFIG.get("push_compliant_max_tcp_force", 20.0)),
        compliant_timeout=REAL_PIPELINE_CONFIG.get("push_compliant_timeout", 15.0),
        compliant_dwell_after_stop=float(REAL_PIPELINE_CONFIG.get("push_compliant_dwell_after_stop", 4.0)),
        push_depart_distance=float(REAL_PIPELINE_CONFIG.get("push_depart_distance", 0.35)),
        push_leave_vel=float(REAL_PIPELINE_CONFIG.get("push_leave_vel", 0.1)),
        push_leave_acc=float(REAL_PIPELINE_CONFIG.get("push_leave_acc", 0.3)),
        # 推开后追加“往箱子中心的水平力控推”：中心取检测到的箱子位姿平移分量（世界系）。
        # build_pick_place_rtde_plan 内部会用 robot.get_real_tcp_pose_sim 把
        # “接触点→中心”的水平方向换算成【真实机器人基系】方向后写入该力控段。
        push_center_pos_world=(
            np.asarray(box_homomat, dtype=float)[:3, 3].tolist()
            if (box_homomat is not None and REAL_PIPELINE_CONFIG.get("push_center_enabled", True))
            else None
        ),
        push_center_distance=float(REAL_PIPELINE_CONFIG.get("push_center_distance", 0.10)),
        push_center_force=float(REAL_PIPELINE_CONFIG.get("push_center_force", 30.0)),
        push_center_vel=float(REAL_PIPELINE_CONFIG.get("push_center_vel", 0.06)),
        push_center_lateral_tolerance=float(REAL_PIPELINE_CONFIG.get("push_center_lateral_tolerance", 0.02)),
        push_center_lateral_stop_tolerance=REAL_PIPELINE_CONFIG.get("push_center_lateral_stop_tolerance", "auto"),
        push_center_zero_ft_sensor=bool(REAL_PIPELINE_CONFIG.get("push_center_zero_ft_sensor", True)),
        push_center_max_tcp_force=float(REAL_PIPELINE_CONFIG.get("push_center_max_tcp_force", 25.0)),
        push_center_timeout=REAL_PIPELINE_CONFIG.get("push_center_timeout", 10.0),
        push_center_dwell_after_stop=float(REAL_PIPELINE_CONFIG.get("push_center_dwell_after_stop", 2.0)),
    )
    rtde_plan.metadata["path_type"] = "push"
    # 仿真预计算“push 末端（contact）真实法兰盘位姿”，供执行时锚定 push_leave（替代运行时 getActualTCPPose）。
    # push_leave 段开始时机械臂已在 contact（push_pos）处，故用 push_pos/push_rotmat 推算真实位姿。
    try:
        # 用规划得到的 contact 关节构型推算真实法兰盘位姿（与抓取规划一致，改用 conf 输入）
        _pred_push = robot.get_real_tcp_pose_from_conf(np.asarray(contact_jv, dtype=float).reshape(-1))
        _attached = False
        for _s in rtde_plan.segments:
            if _s.metadata.get("path_type") == "push_leave":
                _s.metadata["predicted_real_tcp"] = _pred_push.tolist()
                _attached = True
        if _attached:
            print(f"[real_pipeline] 预计算 push 末端真实法兰盘位姿: {np.round(_pred_push, 5).tolist()}")
        else:
            print("[real_pipeline] ⚠️ push 计划中无 push_leave 段，未能附加预测真实位姿（运行时回退仿真航点）")
    except Exception as _exc:
        print(f"[real_pipeline] ⚠️ 预计算 push 末端真实位姿失败，运行时回退仿真航点: {_exc}")
    print(f"[real_pipeline] ✅ push 计划生成完成（{len(rtde_plan.segments)} 段）")
    return rtde_plan


# ============================================================================
# 规划辅助函数（原 dual 模块中的路径规划逻辑，统一集中到本文件）
# ============================================================================
# 在mot_data.ev_list里找抓取即将发生帧
def find_close_index(mot_data) -> Optional[int]:
    values = [None if value is None else float(np.asarray(value).reshape(-1)[0]) for value in mot_data.ev_list]
    for index in range(1, len(values)):
        prev = values[index - 1]
        current = values[index]
        if prev is not None and current is not None and current < prev - 1e-5:
            return index
    return None

# 
def infer_selected_grasp_index(robot, grasps, mot_data, pick_pose: tuple[np.ndarray, np.ndarray]) -> Optional[int]:
    if not grasps or len(mot_data.jv_list) == 0:
        return None
    close_index = find_close_index(mot_data)
    if close_index is None:
        close_index = min(range(len(mot_data.jv_list)), key=lambda idx: abs(idx - len(mot_data.jv_list) // 3))
    tcp_pos, _tcp_rotmat = robot.fk(jnt_values=np.asarray(mot_data.jv_list[close_index], dtype=float))
    tcp_pos = np.asarray(tcp_pos, dtype=float)
    distances = []
    for grasp in grasps:
        grasp_tcp_pos, _grasp_tcp_rotmat = sim_pick.tcp_pose_from_object_pose(pick_pose, grasp)
        distances.append(float(np.linalg.norm(np.asarray(grasp_tcp_pos, dtype=float) - tcp_pos)))
    selected_index = int(np.argmin(distances))
    sim_pick.debug_print(
        f"Inferred selected grasp from close frame {close_index}: "
        f"pickle_grasp_{selected_index}, tcp_error={distances[selected_index]:.6f} m"
    )
    return selected_index


def grasp_jaw_width(grasp) -> float:
    return float(np.asarray(grasp.ee_values, dtype=float).reshape(-1)[0])


def pick_approach_jaw_width_for_grasp(robot, grasp) -> float:
    jaw_width = min(grasp_jaw_width(grasp) + 0.015, 0.118)
    return sim_pick.clamp_jaw_width(robot, jaw_width)


def decorate_grasp_for_rtde(robot, grasp, grasp_index: int, args: Optional[SimpleNamespace] = None):
    jaw_width = grasp_jaw_width(grasp)
    approach_distance = float(sim_pick.APPROACH_DISTANCE)
    pick_approach_jaw_width = pick_approach_jaw_width_for_grasp(robot, grasp)
    setattr(grasp, "name", f"pickle_grasp_{grasp_index}")
    setattr(grasp, "jaw_width", jaw_width)
    setattr(grasp, "pick_approach_jaw_width", pick_approach_jaw_width)
    setattr(grasp, "approach_distance", approach_distance)
    return grasp


def build_and_save_rtde_plan(args: SimpleNamespace, planning: PlanningResult, default_dir: Path) -> tuple[rtde_utils.RtdeExecutionPlan, Path]:
    # 把"运动规划结果"翻译成"真实 UR7e 机器人可执行的 RTDE 计划
    robot = sim_pick.make_robot()
    grasps = sim_pick.load_grasps(robot, sim_pick.GRASP_PICKLE_PATH)
    grasp_index = planning.selected_grasp_index
    if grasp_index is None:
        grasp_index = infer_selected_grasp_index(robot, grasps, planning.mot_data, planning.pick_pose)
    if grasp_index is None or grasp_index < 0 or grasp_index >= len(grasps):
        raise RuntimeError("Could not infer selected grasp for RTDE execution plan.")
    planning.selected_grasp_index = grasp_index
    grasp = decorate_grasp_for_rtde(robot, grasps[grasp_index], grasp_index, args)
    planner_name = "WRS pick-only planner + first-joint alignment"
    # 构建 RTDE 计划
    rtde_plan = rtde_utils.build_pick_place_rtde_plan(
        robot=robot,
        mot_data=planning.mot_data,
        pick_pose=RtdeObjectPose(pos=planning.pick_pose[0], rotmat=planning.pick_pose[1]),
        place_pose=RtdeObjectPose(pos=planning.place_pose[0], rotmat=planning.place_pose[1]),
        grasp=grasp,
        planner_name=planner_name,
        compliant_force=float(args.compliant_force),
        compliant_vel=float(args.compliant_vel),
        compliant_lateral_tolerance=float(args.compliant_lateral_tolerance),
        compliant_lateral_stop_tolerance=args.compliant_lateral_stop_tolerance,
        compliant_force_frame=args.compliant_force_frame,
        compliant_axes=args.compliant_axes,
        compliant_zero_ft_sensor=bool(args.compliant_zero_ft_sensor),
        compliant_max_tcp_force=float(args.compliant_max_tcp_force),
        compliant_timeout=args.compliant_timeout,
        compliant_dwell_after_stop=float(REAL_PIPELINE_CONFIG.get("compliant_dwell_after_stop", 2.0)),
        # 放置力控下压参数（来自 constants.py 的 REAL_PIPELINE_CONFIG）
        place_press_enabled=bool(REAL_PIPELINE_CONFIG.get("place_press_enabled", True)),
        place_press_distance=float(REAL_PIPELINE_CONFIG.get("place_press_distance", 0.03)),
        place_press_force=float(REAL_PIPELINE_CONFIG.get("place_press_force", 10.0)),
        place_press_vel=float(REAL_PIPELINE_CONFIG.get("place_press_vel", 0.02)),
        place_press_lateral_tolerance=float(REAL_PIPELINE_CONFIG.get("place_press_lateral_tolerance", 0.01)),
        place_press_lateral_stop_tolerance=REAL_PIPELINE_CONFIG.get("place_press_lateral_stop_tolerance", "auto"),
        place_press_zero_ft_sensor=bool(REAL_PIPELINE_CONFIG.get("place_press_zero_ft_sensor", True)),
        place_press_max_tcp_force=float(REAL_PIPELINE_CONFIG.get("place_press_max_tcp_force", 20.0)),
        place_press_timeout=REAL_PIPELINE_CONFIG.get("place_press_timeout", None),
        place_press_dwell_after_stop=float(REAL_PIPELINE_CONFIG.get("place_press_dwell_after_stop", 2.0)),
        transfer_movel_waypoints_world=getattr(planning.mot_data, "transfer_movel_waypoints_world", None),
        # moveL 放置目标：机器人基坐标系下的 TCP 位置（真实机器人标定值），
        # 不再使用仿真 world 放置点 BOTTLE_ROBOT_SIDE_PLACE_POS 的 world->base 换算结果。
        transfer_movel_place_pos_base=list(BOTTLE_ROBOT_SIDE_PLACE_POSE_pos),
    )
    rtde_plan.metadata["path_type"] = "pick_place_movel"
    rtde_plan_path = args.rtde_plan_out
    if rtde_plan_path is None:
        rtde_plan_path = default_dir / "pick_place_rtde_plan.json"
    rtde_utils.save_rtde_execution_plan(rtde_plan, rtde_plan_path)
    sim_pick.debug_print(f"Saved RTDE execution plan: {rtde_plan_path}")
    return rtde_plan, rtde_plan_path


def bottle_homomat_for_icp(icp: ObjectIcpResult) -> np.ndarray:
    if icp.bottle_homomat is not None:
        return np.asarray(icp.bottle_homomat, dtype=float)
    if icp.bottle_transform_path is None:
        raise RuntimeError("Bottle pose is missing; press D to estimate the bottle pose again.")
    return utils.load_homomat(icp.bottle_transform_path, "bottle ICP")


def resolve_place_pose(
    args: SimpleNamespace,
    pick_pose: tuple[np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray], str, str]:
    if args.place_pos is None:
        place_pos = np.asarray(BOTTLE_ROBOT_SIDE_PLACE_POS, dtype=float).copy()
        place_pos_source = f"constants.BOTTLE_ROBOT_SIDE_PLACE_POS {sim_pick.format_vec(BOTTLE_ROBOT_SIDE_PLACE_POS, digits=3)}"
    else:
        place_pos = np.asarray(args.place_pos, dtype=float)
        place_pos_source = "--place-pos"

    if args.place_rpy_deg is None:
        place_rotmat = pick_pose[1].copy()
        place_rot_source = "keep ICP pick orientation"
    else:
        _place_pos, place_rotmat = sim_pick.pose_from_pos_rpy(place_pos, args.place_rpy_deg)
        place_rot_source = f"--place-rpy-deg {sim_pick.format_vec(args.place_rpy_deg, digits=3)}"
    return (place_pos, place_rotmat), place_pos_source, place_rot_source


def write_summary(args: SimpleNamespace, icp: ObjectIcpResult, planning: Optional[PlanningResult]) -> Path:
    summary_path = args.summary_out if args.summary_out is not None else icp.output_dir / "real_bottle_pick_place_summary.json"
    summary = {
        "object_summary_path": str(icp.summary_path),
        "bottle_transform_path": None if icp.bottle_transform_path is None else str(icp.bottle_transform_path),
        "box_transform_path": str(icp.box_transform_path),
        "bottle_model_path": str(icp.bottle_model_path),
        "global_registered_path": None if icp.global_registered_path is None else str(icp.global_registered_path),
        "bottle_homomat": None if icp.bottle_homomat is None else np.asarray(icp.bottle_homomat, dtype=float).tolist(),
        "box_collision_obstacle": bool(args.box_obstacle),
        "table_obstacle": not bool(args.no_env),
        "ik_solver": "ikfast",
        "planned": planning is not None,
    }
    if planning is not None:
        summary.update(
            {
                "object_model_path": str(planning.object_model_path),
                "selected_grasp_index": planning.selected_grasp_index,
                "grasp_pickle_path": str(sim_pick.GRASP_PICKLE_PATH),
                "action_sequence": planning.action_sequence,
                "path_type": "pick_place_movel",
                "blocked_grasp_indices": [],
                "frame_count": len(planning.mot_data.jv_list),
                "collision_obstacle_names": planning.obstacle_names,
                "pick_pos": np.asarray(planning.pick_pose[0], dtype=float).tolist(),
                "pick_rotmat": np.asarray(planning.pick_pose[1], dtype=float).tolist(),
                "place_pos": np.asarray(planning.place_pose[0], dtype=float).tolist(),
                "place_rotmat": np.asarray(planning.place_pose[1], dtype=float).tolist(),
                "place_pos_source": planning.place_pos_source,
                "place_rot_source": planning.place_rot_source,
                "rtde_plan_path": None if planning.rtde_plan_path is None else str(planning.rtde_plan_path),
                "rtde_segment_count": None if planning.rtde_plan is None else len(planning.rtde_plan.segments),
                "use_move_l_compliant": bool(args.use_move_l_compliant),
                "visualized_result": not bool(args.dry_run),
            }
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path
