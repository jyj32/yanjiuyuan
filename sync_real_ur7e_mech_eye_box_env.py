"""
Read the current UR7e state, capture/load the current Mech-Eye point cloud,
detect the blue box, and show everything together in WRS.

Default real setup:
    python yanjiuyuan/sync_real_ur7e_mech_eye_box_env.py

Offline robot test with a saved PLY:
    python yanjiuyuan/sync_real_ur7e_mech_eye_box_env.py --mock --ply yanjiuyuan/captures/xxx/colored_pointcloud.ply
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from yanjiuyuan.mech_eye_ur7e_pointcloud_env import (  # noqa: E402
    CAM_TO_WORLD,
    DEFAULT_OUTPUT_ROOT,
    apply_range_filter,
    attach_camera_z_axis,
    attach_detected_box,
    attach_obstacles,
    capture_mech_eye_pointcloud,
    downsample_points,
    estimate_box_pose_with_box_icp_from_arrays,
    load_ply_pointcloud,
    make_output_dir,
    make_rgba,
    numpy_to_open3d_pointcloud,
    open3d_to_numpy,
    parse_range,
    save_numpy_pointcloud,
    transform_points,
)
from yanjiuyuan.sync_real_ur7e_to_wrs import (  # noqa: E402
    DEFAULT_HOME_CONF,
    MockUR7EState,
    RealUR7ERTDEState,
    read_provider_jaw_width,
    SyncState,
    draw_synced_robot,
    make_sync_task,
    validate_jnt_values,
)


@dataclass
class CameraSceneData:
    points_world: np.ndarray
    colors: Optional[np.ndarray]
    raw_point_count: int
    frame: str
    rgb_path: Optional[Path]
    colored_ply_path: Optional[Path]
    world_ply_path: Optional[Path]
    output_dir: Optional[Path]
    depth_path: Optional[Path] = None
    pixel_indices: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize the real UR7e, capture/load Mech-Eye point cloud, detect box, and show all in WRS."
    )

    parser.add_argument("--robot-ip", default="192.168.125.30", help="UR controller IP address.")
    parser.add_argument("--gp-port", default="COM4", help="DH76 gripper serial port passed to UR7EDH76_RTDE.")
    parser.add_argument("--mock", action="store_true", help="Use a local moving pose instead of connecting to RTDE.")
    parser.add_argument("--period", type=float, default=0.05, help="Robot visualization update period in seconds.")
    parser.add_argument("--once", action="store_true", help="Read one robot pose and keep the WRS window open.")
    parser.add_argument("--print-interval", type=float, default=1.0, help="Console robot status print interval.")
    parser.add_argument("--max-failures", type=int, default=20, help="Stop robot sync after this many failures.")
    parser.add_argument("--show-joint-frames", action="store_true", help="Draw joint frames on the robot.")
    parser.add_argument("--hide-tcp-frame", action="store_true", help="Hide the TCP frame.")
    parser.add_argument("--hide-flange-frame", action="store_true", help="Hide the flange frame.")
    parser.add_argument("--alpha", type=float, default=0.85, help="Robot mesh alpha.")

    parser.add_argument("--ply", type=Path, default=None, help="Load an existing PLY instead of capturing.")
    parser.add_argument(
        "--ply-frame",
        choices=("auto", "camera", "world"),
        default="auto",
        help="Coordinate frame of --ply. Auto treats world_colored_pointcloud.ply as world, otherwise camera.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for captures.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Exact output directory for this run.")
    parser.add_argument("--ply-out", type=Path, default=None, help="Override colored PLY output path when capturing.")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Depth scale passed to Mech_camera.")
    parser.add_argument("--depth-trunc", type=float, default=3.0, help="Depth truncation distance in meters.")

    parser.add_argument("--max-points", type=int, default=150000, help="Maximum points drawn in Panda3D.")
    parser.add_argument("--point-size", type=float, default=0.002, help="Rendered point size in meters.")
    parser.add_argument("--x-range", type=parse_range, default=None, help="Optional world X display filter: min,max.")
    parser.add_argument("--y-range", type=parse_range, default=None, help="Optional world Y display filter: min,max.")
    parser.add_argument("--z-range", type=parse_range, default=None, help="Optional world Z display filter: min,max.")
    parser.add_argument("--no-env", action="store_true", help="Do not draw reference table/camera/box obstacles.")
    parser.add_argument("--show-obstacle-collision", action="store_true", help="Show obstacle collision primitives.")

    parser.add_argument("--no-detect-box", dest="detect_box", action="store_false", help="Skip box detection.")
    parser.set_defaults(detect_box=True)
    parser.add_argument("--box-transform-out", type=Path, default=None, help="Optional path to save detected box transform.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional path to save run summary text.")
    return parser.parse_args()


def resolve_run_output_dir(args: argparse.Namespace) -> Optional[Path]:
    if args.ply is None or args.output_dir is not None:
        return make_output_dir(args.output_root, args.output_dir)
    return None


def resolve_ply_frame(ply_path: Path, requested_frame: str) -> str:
    if requested_frame != "auto":
        return requested_frame
    if ply_path.name.lower() == "world_colored_pointcloud.ply":
        return "world"
    return "camera"


def make_robot_provider(args: argparse.Namespace):
    if args.mock:
        return MockUR7EState(DEFAULT_HOME_CONF)
    return RealUR7ERTDEState(robot_ip=args.robot_ip, gp_port=args.gp_port)


def read_robot_snapshot(provider, label: str) -> tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
    jnt_values = validate_jnt_values(provider.get_jnt_values())
    tcp_pos = provider.get_tcp_pos()
    jaw_width = read_provider_jaw_width(provider)
    print(f"{label} joints(deg): {np.round(np.degrees(jnt_values), 2).tolist()}")
    if tcp_pos is not None:
        print(f"{label} tcp_pos(m): {np.round(tcp_pos, 5).tolist()}")
    else:
        print(f"{label} tcp_pos(m): unavailable")
    if jaw_width is not None:
        print(f"{label} jaw_width(m): {jaw_width:.6f}")
    else:
        print(f"{label} jaw_width(m): unavailable")
    return jnt_values, tcp_pos, jaw_width

def print_camera_info(frame: str, points_world: np.ndarray, source_path: Optional[Path]) -> None:
    camera_pos = CAM_TO_WORLD[:3, 3]
    camera_view_dir = -CAM_TO_WORLD[:3, 2]
    print(f"Camera point cloud frame: {frame}")
    if source_path is not None:
        print(f"Camera point cloud source: {source_path}")
    print(f"Camera calibrated world position(m): {np.round(camera_pos, 6).tolist()}")
    print(f"Camera calibrated view direction(world): {np.round(camera_view_dir, 6).tolist()}")
    print("Camera-to-world matrix:\n" + np.array2string(CAM_TO_WORLD, precision=6))
    mins = points_world.min(axis=0)
    maxs = points_world.max(axis=0)
    print(
        "World point cloud bounds: "
        f"x=[{mins[0]:.4f}, {maxs[0]:.4f}], "
        f"y=[{mins[1]:.4f}, {maxs[1]:.4f}], "
        f"z=[{mins[2]:.4f}, {maxs[2]:.4f}]"
    )


def load_or_capture_camera_scene(args: argparse.Namespace, output_dir: Optional[Path]) -> CameraSceneData:
    rgb_path = None
    world_ply_path = None

    if args.ply is None:
        if output_dir is None:
            raise RuntimeError("Internal error: output_dir should exist when capturing from camera.")
        pcd, rgb_path, colored_ply_path, depth_path = capture_mech_eye_pointcloud(
            output_dir=output_dir,
            ply_out=args.ply_out,
            depth_scale=args.depth_scale,
            depth_trunc=args.depth_trunc,
        )
        frame = "camera"
        print(f"Saved RGB image to: {rgb_path}")
        print(f"Saved camera-frame colored point cloud to: {colored_ply_path}")
    else:
        colored_ply_path = args.ply
        pcd = load_ply_pointcloud(args.ply)
        frame = resolve_ply_frame(args.ply, args.ply_frame)
        print(f"Loaded point cloud from: {args.ply}")

    points, colors = open3d_to_numpy(pcd)
    raw_count = len(points)
    points_world = transform_points(points, CAM_TO_WORLD) if frame == "camera" else points.copy()

    if output_dir is not None:
        world_ply_path = output_dir / "world_colored_pointcloud.ply"
        save_numpy_pointcloud(points_world, colors, world_ply_path)
        print(f"Saved world-frame colored point cloud to: {world_ply_path}")

    print_camera_info(frame=frame, points_world=points_world, source_path=colored_ply_path)
    return CameraSceneData(
        points_world=points_world,
        colors=colors,
        raw_point_count=raw_count,
        frame=frame,
        rgb_path=rgb_path,
        colored_ply_path=colored_ply_path,
        world_ply_path=world_ply_path,
        output_dir=output_dir,
        depth_path=depth_path,
    )


def detect_box(args: argparse.Namespace, scene_data: CameraSceneData) -> Optional[dict]:
    if not args.detect_box:
        print("Box detection disabled.")
        return None

    print("Detecting blue box with NumPy prefilter + OBB-initialized local ICP...")
    box_detection = estimate_box_pose_with_box_icp_from_arrays(scene_data.points_world, scene_data.colors)
    transform = box_detection["transform"]
    box_pos = transform[:3, 3]
    print(f"Detected box position(m): {np.round(box_pos, 6).tolist()}")
    print(
        "Detected box ICP fitness/rmse: "
        f"{box_detection['icp_fitness']:.6f} / {box_detection['icp_rmse']:.6f}"
    )
    print(
        "Detected box OBB initial fitness/rmse: "
        f"{box_detection['obb_initial_fitness']:.6f} / {box_detection['obb_initial_rmse']:.6f}"
    )
    print(
        "Detected box target points: "
        f"segmented={box_detection['segmented_target_count']}, "
        f"downsampled={box_detection['downsampled_target_count']}, "
        f"clustered={box_detection['largest_cluster_count']}, "
        f"z_filtered={box_detection['clean_target_count']}"
    )
    print("Detected box transform:\n" + np.array2string(transform, precision=6))

    box_transform_out = args.box_transform_out
    if box_transform_out is None and scene_data.output_dir is not None:
        box_transform_out = scene_data.output_dir / "detected_box_transform.txt"
    if box_transform_out is not None:
        box_transform_out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(box_transform_out, transform, fmt="%.9f")
        print(f"Saved detected box transform to: {box_transform_out}")
    return box_detection


def write_summary(
    args: argparse.Namespace,
    scene_data: CameraSceneData,
    robot_jnt_values: np.ndarray,
    robot_tcp_pos: Optional[np.ndarray],
    robot_jaw_width: Optional[float],
    box_detection: Optional[dict],
) -> None:
    summary_out = args.summary_out
    if summary_out is None and scene_data.output_dir is not None:
        summary_out = scene_data.output_dir / "robot_camera_box_summary.txt"
    if summary_out is None:
        return

    lines = [
        f"robot_ip: {args.robot_ip}",
        f"mock_robot: {args.mock}",
        f"robot_joints_rad: {np.array2string(robot_jnt_values, precision=9)}",
        f"robot_joints_deg: {np.array2string(np.degrees(robot_jnt_values), precision=6)}",
        f"robot_tcp_pos: {None if robot_tcp_pos is None else np.array2string(robot_tcp_pos, precision=9)}",
        f"robot_jaw_width: {robot_jaw_width}",
        f"camera_frame: {scene_data.frame}",
        f"camera_rgb_path: {scene_data.rgb_path}",
        f"camera_colored_ply_path: {scene_data.colored_ply_path}",
        f"camera_world_ply_path: {scene_data.world_ply_path}",
        f"camera_position_world: {np.array2string(CAM_TO_WORLD[:3, 3], precision=9)}",
        "camera_to_world:",
        np.array2string(CAM_TO_WORLD, precision=9),
        f"raw_point_count: {scene_data.raw_point_count}",
    ]
    if box_detection is not None:
        lines.extend(
            [
                f"box_position_world: {np.array2string(box_detection['transform'][:3, 3], precision=9)}",
                f"box_icp_fitness: {box_detection['icp_fitness']:.9f}",
                f"box_icp_rmse: {box_detection['icp_rmse']:.9f}",
                "box_transform:",
                np.array2string(box_detection["transform"], precision=9),
            ]
        )
    else:
        lines.append("box_detection: None")

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved robot/camera/box summary to: {summary_out}")


def build_scene(
    args: argparse.Namespace,
    scene_data: CameraSceneData,
    provider,
    first_jnt_values: np.ndarray,
    first_jaw_width: Optional[float],
    box_detection: Optional[dict],
) -> None:
    from wrs import mgm, wd
    from wrs.robot_sim.robots.ur7e.ur7e_withouttable_dh76 import UR7EDH76

    base = wd.World(cam_pos=[2.0, -1.6, 1.2], lookat_pos=[0.4, -0.25, 0.3])
    mgm.gen_frame(ax_length=0.25, ax_radius=0.004).attach_to(base)

    if not args.no_env:
        attach_obstacles(base, show_collision=args.show_obstacle_collision)
    attach_camera_z_axis(base, mgm)

    points, colors = apply_range_filter(
        scene_data.points_world,
        scene_data.colors,
        args.x_range,
        args.y_range,
        args.z_range,
    )
    points, colors = downsample_points(points, colors, args.max_points)
    rgba = make_rgba(colors, len(points))
    mgm.gen_pointcloud(points=points, rgba=rgba, point_size=args.point_size).attach_to(base)
    print(f"Displayed {len(points)} / {scene_data.raw_point_count} camera points in WRS.")

    if box_detection is not None:
        box_model = attach_detected_box(
            base,
            box_detection["transform"],
            show_collision=args.show_obstacle_collision,
        )
        if box_model is not None:
            print("Displayed detected box.STL model in light blue.")

    robot_s = UR7EDH76(enable_cc=True)
    state = SyncState(robot=robot_s, provider=provider)
    draw_synced_robot(base, state, first_jnt_values, args, jaw_width=first_jaw_width)

    if args.once:
        print("One-shot robot pose drawn. The WRS window will stay open.")
        provider.close()
    else:
        base.taskMgr.doMethodLater(
            max(args.period, 0.001),
            make_sync_task(base, state, args),
            "sync_real_ur7e_mech_eye_box_env",
            appendTask=True,
        )
    base.run()


def main() -> None:
    args = parse_args()
    output_dir = resolve_run_output_dir(args)

    provider = make_robot_provider(args)
    atexit.register(provider.close)

    _initial_jnt_values, _initial_tcp_pos, _initial_jaw_width = read_robot_snapshot(provider, "Initial robot")
    scene_data = load_or_capture_camera_scene(args, output_dir)
    try:
        box_detection = detect_box(args, scene_data)
    except Exception as exc:
        box_detection = None
        print(
            "Warning: box detection failed; continuing without detected box model. "
            f"{type(exc).__name__}: {exc}"
        )
    current_jnt_values, current_tcp_pos, current_jaw_width = read_robot_snapshot(provider, "Scene robot")
    write_summary(args, scene_data, current_jnt_values, current_tcp_pos, current_jaw_width, box_detection)

    build_scene(
        args=args,
        scene_data=scene_data,
        provider=provider,
        first_jnt_values=current_jnt_values,
        first_jaw_width=current_jaw_width,
        box_detection=box_detection,
    )


if __name__ == "__main__":
    main()
