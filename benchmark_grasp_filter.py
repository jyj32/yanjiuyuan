"""
离线测速：reason_common_gids + delete_collision_gids
不需要连接真实机器人和相机，全部用仿真数据。

用法:
    conda activate wrsrobot3.11
    python yanjiuyuan/benchmark_grasp_filter.py
    python yanjiuyuan/benchmark_grasp_filter.py --repeat 10          # 跑 10 轮取平均
    python yanjiuyuan/benchmark_grasp_filter.py --ply path/to/xxx.ply # 指定 remaining_pointcloud.ply
    python yanjiuyuan/benchmark_grasp_filter.py --box-transform path/to/box_transform_used.txt
    python yanjiuyuan/benchmark_grasp_filter.py --no-box              # 不加箱子障碍物
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yanjiuyuan import sim_pick_and_place as sim_pick  # noqa: E402
from yanjiuyuan.box_collision import make_concave_box_collision_obstacles  # noqa: E402


def find_default_ply() -> str:
    """自动找最近一次 capture 的 remaining_pointcloud.ply"""
    captures_dir = REPO_ROOT / "yanjiuyuan" / "captures"
    if not captures_dir.exists():
        return ""
    candidates = sorted(captures_dir.glob("*/box_object_extraction/remaining_pointcloud.ply"))
    return str(candidates[-1]) if candidates else ""


def find_default_box_transform() -> str:
    """自动找最近一次 capture 的 box_transform_used.txt"""
    captures_dir = REPO_ROOT / "yanjiuyuan" / "captures"
    if not captures_dir.exists():
        return ""
    candidates = sorted(captures_dir.glob("*/box_object_extraction/box_transform_used.txt"))
    return str(candidates[-1]) if candidates else ""


def main():
    parser = argparse.ArgumentParser(description="离线测速 grasp filter")
    parser.add_argument("--repeat", type=int, default=5, help="重复次数（取平均）")
    parser.add_argument("--ply", type=str, default="", help="remaining_pointcloud.ply 路径")
    parser.add_argument("--box-transform", type=str, default="", help="box_transform_used.txt 路径")
    parser.add_argument("--no-box", action="store_true", help="不添加箱子障碍物")
    args = parser.parse_args()

    repeat = args.repeat
    ply_path = args.ply or find_default_ply()
    box_transform_path = args.box_transform or find_default_box_transform()

    # ---- 1. 构建仿真环境 ----
    print("=" * 60)
    print("[1/4] 构建仿真环境（无需连接真实机器人/相机）")
    print("=" * 60)

    robot = sim_pick.make_robot()
    print(f"  仿真机器人: {type(robot).__name__}")

    grasps = sim_pick.load_grasps(robot, sim_pick.GRASP_PICKLE_PATH)
    print(f"  加载 grasps: {len(grasps)} 个 (from {sim_pick.GRASP_PICKLE_PATH.name})")

    pick_pose = sim_pick.pose_from_pos_rpy(sim_pick.PICK_POS, sim_pick.PICK_RPY_DEG)
    print(f"  pick_pose: pos={pick_pose[0]}, rpy_deg={sim_pick.PICK_RPY_DEG}")

    table = sim_pick.make_table_obstacle()
    obstacle_list = [table]

    # 添加箱子障碍物（凹形碰撞模型：底面 + 4面壁）
    if not args.no_box:
        if box_transform_path and Path(box_transform_path).exists():
            box_homomat = np.loadtxt(box_transform_path).reshape(4, 4)
            box_obstacles = make_concave_box_collision_obstacles(box_homomat)
            obstacle_list.extend(box_obstacles)
            print(f"  箱子障碍物: {[o.name for o in box_obstacles]} (from {Path(box_transform_path).name})")
        else:
            print("  [警告] 未找到 box_transform_used.txt，跳过箱子障碍物（用 --box-transform 指定）")

    print(f"  障碍物: {[o.name for o in obstacle_list]}")

    if not ply_path or not Path(ply_path).exists():
        print("  [警告] 未找到 remaining_pointcloud.ply，delete_collision_gids 将使用随机点云")
        # 生成一个随机点云作为替代
        import open3d as o3d
        remaining_pcd = o3d.geometry.PointCloud()
        remaining_pcd.points = o3d.utility.Vector3dVector(
            np.random.uniform(-0.3, 0.3, size=(5000, 3))
        )
    else:
        import open3d as o3d
        remaining_pcd = o3d.io.read_point_cloud(ply_path)
        print(f"  加载 remaining_pointcloud: {ply_path} ({len(remaining_pcd.points)} points)")

    # ---- 2. 测速 reason_common_gids ----
    print()
    print("=" * 60)
    print(f"[2/4] 测速 reason_common_gids (repeat={repeat})")
    print("=" * 60)

    from wrs import ppp

    planner = ppp.PickPlacePlanner(robot)

    reason_times = []
    reason_results = []
    for i in range(repeat):
        grasp_collection = sim_pick.make_grasp_collection(robot, grasps)
        t0 = perf_counter()
        candidate_indices = list(
            planner.reason_common_gids(
                grasp_collection=grasp_collection,
                goal_pose_list=[pick_pose],
                obstacle_list=obstacle_list,
                toggle_dbg=False,
            )
        )
        elapsed = perf_counter() - t0
        reason_times.append(elapsed)
        reason_results.append(len(candidate_indices))
        print(f"  run {i+1}/{repeat}: {elapsed:.3f}s  |  剩余 {len(candidate_indices)}/{len(grasps)} 个候选")

    reason_avg = np.mean(reason_times)
    reason_std = np.std(reason_times)
    print(f"  >>> reason_common_gids 平均: {reason_avg:.3f}s  (std={reason_std:.3f}s)")

    # ---- 3. 测速 delete_collision_gids ----
    print()
    print("=" * 60)
    print(f"[3/4] 测速 delete_collision_gids (repeat={repeat})")
    print("=" * 60)

    # 两个函数都用最初的全部候选作为输入，独立测速
    grasp_collection = sim_pick.make_grasp_collection(robot, grasps)
    candidate_indices_input = list(range(len(grasps)))

    # 导入 delete_collision_gids（定义在主文件中）
    from yanjiuyuan.real_bottle_pick_place_interactive1_point_completion import delete_collision_gids

    collision_times = []
    collision_results = []
    for i in range(repeat):
        t0 = perf_counter() # 纳秒级测时
        filtered = delete_collision_gids(
            candidate_indices_0=list(candidate_indices_input),
            grasp_collection=grasp_collection,
            goal_pose_list=[pick_pose],
            gripper=robot.end_effector,
            surrounding_pcd=remaining_pcd,
            show=False,
        )
        elapsed = perf_counter() - t0
        collision_times.append(elapsed)
        collision_results.append(len(filtered))
        print(f"  run {i+1}/{repeat}: {elapsed:.3f}s  |  剩余 {len(filtered)}/{len(candidate_indices_input)} 个候选")

    collision_avg = np.mean(collision_times)
    collision_std = np.std(collision_times)
    print(f"  >>> delete_collision_gids 平均: {collision_avg:.3f}s  (std={collision_std:.3f}s)")

    # ---- 4. 对比两种顺序的总时间 ----
    print()
    print("=" * 60)
    print(f"[4/4] 对比两种顺序的总时间 (repeat={repeat})")
    print("=" * 60)

    all_indices = list(range(len(grasps)))

    # 方案A：先 reason → 再 delete_collision
    order_a_times = []
    order_a_reason_times = []
    order_a_collision_times = []
    for i in range(repeat):
        gc_a = sim_pick.make_grasp_collection(robot, grasps)
        t0 = perf_counter()
        step1 = list(
            planner.reason_common_gids(
                grasp_collection=gc_a,
                goal_pose_list=[pick_pose],
                obstacle_list=obstacle_list,
                toggle_dbg=False,
            )
        )
        t1 = perf_counter()
        step2 = delete_collision_gids(
            candidate_indices_0=step1,
            grasp_collection=gc_a,
            goal_pose_list=[pick_pose],
            gripper=robot.end_effector,
            surrounding_pcd=remaining_pcd,
            show=False,
        )
        t2 = perf_counter()
        reason_t = t1 - t0
        collision_t = t2 - t1
        total_t = t2 - t0
        order_a_times.append(total_t)
        order_a_reason_times.append(reason_t)
        order_a_collision_times.append(collision_t)
        print(f"  [A] run {i+1}/{repeat}: 总计 {total_t:.3f}s (reason {reason_t:.3f}s + collision {collision_t:.3f}s)  |  reason {len(all_indices)}→{len(step1)} → collision →{len(step1)}→{len(step2)}")

    # 方案B：先 delete_collision → 再 reason
    order_b_times = []
    order_b_collision_times = []
    order_b_reason_times = []
    for i in range(repeat):
        gc_b = sim_pick.make_grasp_collection(robot, grasps)
        t0 = perf_counter()
        step1 = delete_collision_gids(
            candidate_indices_0=list(all_indices),
            grasp_collection=gc_b,
            goal_pose_list=[pick_pose],
            gripper=robot.end_effector,
            surrounding_pcd=remaining_pcd,
            show=False,
        )
        t1 = perf_counter()
        step2 = list(
            planner.reason_common_gids_filtered(
                grasp_collection=gc_b,
                goal_pose_list=[pick_pose],
                previous_available_gids=step1,
                obstacle_list=obstacle_list,
                toggle_dbg=False,
            )
        )
        t2 = perf_counter()
        collision_t = t1 - t0
        reason_t = t2 - t1
        total_t = t2 - t0
        order_b_times.append(total_t)
        order_b_collision_times.append(collision_t)
        order_b_reason_times.append(reason_t)
        print(f"  [B] run {i+1}/{repeat}: 总计 {total_t:.3f}s (collision {collision_t:.3f}s + reason {reason_t:.3f}s)  |  collision {len(all_indices)}→{len(step1)} → reason {len(step1)}→{len(step2)}")

    # 去掉首次预热，取后续轮次平均
    def _rest_avg(lst):
        rest = lst[1:] if len(lst) > 1 else lst
        return np.mean(rest)

    a_total_avg = _rest_avg(order_a_times)
    a_reason_avg = _rest_avg(order_a_reason_times)
    a_collision_avg = _rest_avg(order_a_collision_times)
    b_total_avg = _rest_avg(order_b_times)
    b_reason_avg = _rest_avg(order_b_reason_times)
    b_collision_avg = _rest_avg(order_b_collision_times)

    print()
    print(f"  方案A（先reason后collision）:")
    print(f"    首次: 总计 {order_a_times[0]:.3f}s (reason {order_a_reason_times[0]:.3f}s + collision {order_a_collision_times[0]:.3f}s)")
    print(f"    预热后平均: 总计 {a_total_avg:.3f}s (reason {a_reason_avg:.3f}s + collision {a_collision_avg:.3f}s)")
    print(f"  方案B（先collision后reason）:")
    print(f"    首次: 总计 {order_b_times[0]:.3f}s (collision {order_b_collision_times[0]:.3f}s + reason {order_b_reason_times[0]:.3f}s)")
    print(f"    预热后平均: 总计 {b_total_avg:.3f}s (collision {b_collision_avg:.3f}s + reason {b_reason_avg:.3f}s)")
    print()
    faster = "A" if a_total_avg < b_total_avg else "B"
    diff = abs(a_total_avg - b_total_avg)
    print(f"  >>> 方案{faster} 更快，快 {diff:.3f}s ({diff/min(a_total_avg,b_total_avg)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
