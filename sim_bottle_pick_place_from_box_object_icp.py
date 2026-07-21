"""
Offline simulation pipeline: saved box/object point cloud extraction -> bottle pick-and-place.

This script deliberately stays offline-only. It uses saved Mech-Eye capture data
or an existing box_object_extraction_summary.json, estimates the bottle start
pose with ICP, shows the start/place poses, and plans/animates the WRS
pick-and-place motion.

Interactive offline workflow:
    python yanjiuyuan/sim_bottle_pick_place_from_box_object_icp.py

    In the WRS viewer, press D to segment the object, choose a bottle template,
    and estimate the start pose. The estimated start bottle and fixed place goal
    are shown together. Press P to plan and animate the pick-and-place motion.

Batch path with an existing object extraction summary:
    python yanjiuyuan/sim_bottle_pick_place_from_box_object_icp.py ^
        --object-summary yanjiuyuan/captures/20260624-151047/box_object_extraction/box_object_extraction_summary.json ^
        --dry-run
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from yanjiuyuan.constants import BOX_CAPTURE_ROOT, BOTTLE_ROBOT_SIDE_PLACE_POS, MODEL_DIR  # noqa: E402
from yanjiuyuan import box_object_pointcloud_from_saved_capture_bottle_icp as box_object_icp  # noqa: E402
from yanjiuyuan import connection_status as conn_status  # noqa: E402
from yanjiuyuan import pick_place_rtde_utils as rtde_utils  # noqa: E402
from yanjiuyuan import sim_pick_and_place as sim_pick  # noqa: E402
from yanjiuyuan import sync_real_ur7e_mech_eye_box_env as sync_scene  # noqa: E402
from yanjiuyuan.box_collision import (  # noqa: E402
    make_concave_box_collision_obstacles,
    make_detected_box_visual_model,
)


BOX_OBJECT_SCRIPT = Path(__file__).resolve().parent / "box_object_pointcloud_from_saved_capture_bottle_icp.py"
DEFAULT_OBJECT_MODEL_PATH = MODEL_DIR / "bottle.stl"
DEFAULT_ROBOT_SIDE_PLACE_POS = np.array([0.55, -0.43, 0.08], dtype=float)


@dataclass
class ObjectIcpResult:
    output_dir: Path
    summary_path: Path
    bottle_transform_path: Path
    box_transform_path: Path
    bottle_model_path: Path


@dataclass
class PlanningResult:
    selected_grasp_index: Optional[int]
    mot_data: Any
    pick_pose: tuple[np.ndarray, np.ndarray]
    place_pose: tuple[np.ndarray, np.ndarray]
    object_model_path: Path
    obstacle_names: list[str]
    place_pos_source: str
    place_rot_source: str
    rtde_plan: Optional[rtde_utils.RtdeExecutionPlan] = None
    rtde_plan_path: Optional[Path] = None


@dataclass
class RtdeObjectPose:
    pos: np.ndarray
    rotmat: np.ndarray


def parse_vec3(value: str) -> np.ndarray:
    items = [item.strip() for item in value.split(",")]
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    try:
        result = np.array([float(item) for item in items], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not np.all(np.isfinite(result)):
        raise argparse.ArgumentTypeError("vector contains NaN or inf")
    return result


def parse_vec6(value: str) -> np.ndarray:
    items = [item.strip() for item in value.split(",")]
    if len(items) != 6:
        raise argparse.ArgumentTypeError("expected six comma-separated values")
    try:
        result = np.array([float(item) for item in items], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not np.all(np.isfinite(result)):
        raise argparse.ArgumentTypeError("vector contains NaN or inf")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline bottle ICP from a saved box capture, then WRS pick-and-place simulation."
    )

    extraction = parser.add_argument_group("object extraction / bottle ICP")
    extraction.add_argument("--capture-root", type=Path, default=BOX_CAPTURE_ROOT)
    extraction.add_argument("--capture-dir", type=Path, default=None, help="Saved capture folder. Defaults to newest capture.")
    extraction.add_argument("--ply", type=Path, default=None, help="Specific colored PLY for extraction.")
    extraction.add_argument("--image", type=Path, default=None, help="RGB image for point prompt segmentation.")
    extraction.add_argument("--box-transform", type=Path, default=None, help="Detected box transform. Defaults to capture output.")
    extraction.add_argument("--object-output-dir", type=Path, default=None, help="Output directory for extraction/ICP.")
    extraction.add_argument(
        "--object-summary",
        type=Path,
        default=None,
        help="Reuse an existing box_object_extraction_summary.json and skip extraction.",
    )
    extraction.add_argument("--mask", type=Path, default=None, help="Existing 2D mask for the bottle.")
    extraction.add_argument("--point", action="append", default=[], metavar="X,Y,LABEL")
    extraction.add_argument("--segment-box", default=None, metavar="X1,Y1,X2,Y2")
    extraction.add_argument("--auto-segment-box", action=argparse.BooleanOptionalAction, default=False)
    extraction.add_argument("--backend", choices=("fastsam", "sam"), default="sam")
    extraction.add_argument("--model", default=None)
    extraction.add_argument("--keep", choices=("best", "all", "largest", "smallest", "combined"), default="best")
    extraction.add_argument("--imgsz", type=int, default=1024)
    extraction.add_argument("--conf", type=float, default=0.25)
    extraction.add_argument("--iou", type=float, default=0.9)
    extraction.add_argument("--device", default=None)
    extraction.add_argument("--no-gui", action="store_true")
    extraction.add_argument("--show-object-viewer", action="store_true")
    extraction.add_argument("--show-box-model", action="store_true")
    extraction.add_argument(
        "--bottle-template",
        choices=("prompt", "surface", "top", "front", "left", "right", "custom"),
        default="prompt",
    )
    extraction.add_argument("--bottle-template-ply", type=Path, default=None)
    extraction.add_argument("--bottle-template-prompt-gui", action=argparse.BooleanOptionalAction, default=True)
    extraction.add_argument("--bottle-stl", type=Path, default=DEFAULT_OBJECT_MODEL_PATH)
    extraction.add_argument("--bottle-voxel", type=float, default=None)
    extraction.add_argument("--bottle-icp-max-iteration", type=int, default=None)

    sync_group = parser.add_argument_group("offline visualization / RTDE plan export")
    sync_group.add_argument("--robot-ip", default="192.168.125.30", help=argparse.SUPPRESS)
    sync_group.add_argument("--gp-port", default="COM3", help=argparse.SUPPRESS)
    sync_group.add_argument("--mock", action="store_true", help=argparse.SUPPRESS)
    sync_group.add_argument("--depth-scale", type=float, default=0.001, help=argparse.SUPPRESS)
    sync_group.add_argument("--depth-trunc", type=float, default=3.0, help=argparse.SUPPRESS)
    sync_group.add_argument("--scene-max-points", type=int, default=150000, help="Maximum scene points drawn in WRS.")
    sync_group.add_argument("--scene-point-size", type=float, default=0.002, help="Rendered point size for scene points.")
    sync_group.add_argument("--rtde-plan-out", type=Path, default=None, help="Where to save the offline RTDE plan generated by P.")
    sync_group.add_argument("--execute-dry-run", action=argparse.BooleanOptionalAction, default=False, help=argparse.SUPPRESS)
    sync_group.add_argument("--max-start-joint-error-deg", type=float, default=5.0, help=argparse.SUPPRESS)

    planning = parser.add_argument_group("simulation planning")
    planning.add_argument("--skip-plan", action="store_true", help="Stop after extraction/ICP.")
    planning.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Open a WRS viewer where D runs segmentation/ICP and P plans pick-and-place. Defaults on for capture-driven GUI runs.",
    )
    planning.add_argument("--object-model", type=Path, default=None, help="Object mesh for simulation.")
    planning.add_argument(
        "--place-pos",
        type=parse_vec3,
        default=None,
        help="World XYZ for the placed bottle. Defaults to a fixed robot-side pose.",
    )
    planning.add_argument(
        "--place-rpy-deg",
        type=parse_vec3,
        default=None,
        help="World RPY for the placed bottle. Defaults to keeping the ICP pick orientation.",
    )
    planning.add_argument("--start-conf-deg", type=parse_vec6, default=None)
    planning.add_argument("--open-jaw", type=float, default=None)
    planning.add_argument("--approach-distance", type=float, default=None)
    planning.add_argument("--dry-run", action="store_true", help="Plan and print results without opening the WRS window.")
    planning.add_argument("--no-rrt", dest="use_rrt", action="store_false")
    planning.set_defaults(use_rrt=True)
    planning.add_argument("--no-collision-check", dest="collision_check", action="store_false")
    planning.set_defaults(collision_check=True)
    planning.add_argument("--no-env", action="store_true", help="Do not add the table obstacle.")
    planning.add_argument(
        "--box-obstacle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the detected concave box panels as planning collision obstacles. Use --no-box-obstacle only for diagnosis.",
    )

    planning.add_argument("--visualize-failure", action="store_true", help="Open the WRS failure-debug scene if planning fails.")
    planning.add_argument("--summary-out", type=Path, default=None)

    return parser.parse_args()


def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def normalize_paths(args: argparse.Namespace) -> None:
    for attr in (
        "capture_root",
        "capture_dir",
        "ply",
        "image",
        "box_transform",
        "object_output_dir",
        "object_summary",
        "mask",
        "bottle_template_ply",
        "bottle_stl",
        "object_model",
        "rtde_plan_out",
        "summary_out",
    ):
        setattr(args, attr, resolve_path(getattr(args, attr)))


def append_path(cmd: list[str], flag: str, path: Optional[Path]) -> None:
    if path is not None:
        cmd.extend([flag, str(path)])


def append_value(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def run_command(label: str, cmd: list[str]) -> None:
    print(f"[sim_pipeline] {label}:")
    print(f"  {subprocess.list2cmdline(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def resolve_object_output_dir(args: argparse.Namespace) -> Path:
    if args.object_output_dir is not None:
        return args.object_output_dir
    if args.capture_dir is not None:
        return args.capture_dir / "box_object_extraction"
    return Path.cwd() / "box_object_extraction"


def build_box_object_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        capture_root=args.capture_root,
        capture_dir=args.capture_dir,
        ply=args.ply,
        prefer_world_ply=True,
        image=args.image,
        box_transform=args.box_transform,
        output_dir=output_dir,
        mask=args.mask,
        point=list(args.point),
        segment_box=args.segment_box,
        auto_segment_box=args.auto_segment_box,
        backend=args.backend,
        model=args.model,
        keep=args.keep,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_display=1400,
        no_gui=args.no_gui,
        inner_xy_margin=box_object_icp.BOX_OBJECT_INNER_XY_MARGIN,
        top_overhang_xy_margin=box_object_icp.BOX_OBJECT_TOP_OVERHANG_XY_MARGIN,
        above_top_margin=box_object_icp.BOX_OBJECT_ABOVE_TOP_MARGIN,
        remove_blue_box_points=box_object_icp.BOX_OBJECT_REMOVE_BLUE_BOX_POINTS,
        show_removed_context=box_object_icp.BOX_OBJECT_SHOW_REMOVED_CONTEXT,
        removed_context_xy_margin=box_object_icp.BOX_OBJECT_REMOVED_CONTEXT_XY_MARGIN,
        removed_context_z_margin=box_object_icp.BOX_OBJECT_REMOVED_CONTEXT_Z_MARGIN,
        pixel_color_tolerance=box_object_icp.BOX_OBJECT_PIXEL_COLOR_TOLERANCE,
        pixel_mapping_min_ratio=box_object_icp.BOX_OBJECT_PIXEL_MAPPING_MIN_RATIO,
        candidate_voxel=box_object_icp.BOX_OBJECT_CANDIDATE_VOXEL_DOWNSAMPLE,
        removed_voxel=box_object_icp.BOX_OBJECT_REMOVED_VOXEL_DOWNSAMPLE,
        selected_voxel=box_object_icp.BOX_OBJECT_SELECTED_VOXEL_DOWNSAMPLE,
        show_viewer=True,
        show_box_model=args.show_box_model,
        bottle_icp=True,
        bottle_template=args.bottle_template,
        bottle_template_ply=args.bottle_template_ply,
        bottle_template_prompt_gui=args.bottle_template_prompt_gui,
        bottle_template_preview_size=260,
        bottle_stl=args.bottle_stl,
        bottle_voxel=(args.bottle_voxel if args.bottle_voxel is not None else box_object_icp.bottle_icp.VOXEL_SIZE),
        bottle_model_sample_count=box_object_icp.bottle_icp.MODEL_SAMPLE_COUNT,
        bottle_model_even_radius=box_object_icp.bottle_icp.MODEL_EVEN_RADIUS,
        bottle_icp_max_iteration=(
            args.bottle_icp_max_iteration
            if args.bottle_icp_max_iteration is not None
            else box_object_icp.bottle_icp.ICP_MAX_ITERATION
        ),
        point_size=box_object_icp.BOX_OBJECT_POINT_SIZE,
    )


def prepare_interactive_pipeline_context(args: argparse.Namespace) -> box_object_icp.PipelineContext:
    output_dir = resolve_object_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    return box_object_icp.prepare_pipeline_context(build_box_object_args(args, output_dir))


def make_sync_capture_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        robot_ip=args.robot_ip,
        gp_port=args.gp_port,
        mock=args.mock,
        ply=None,
        ply_frame="auto",
        output_root=args.capture_root,
        output_dir=output_dir,
        ply_out=None,
        depth_scale=args.depth_scale,
        depth_trunc=args.depth_trunc,
        detect_box=True,
        box_transform_out=output_dir / "detected_box_transform.txt",
        summary_out=output_dir / "robot_camera_box_summary.txt",
    )


def capture_synced_context(args: argparse.Namespace) -> tuple[box_object_icp.PipelineContext, dict]:
    output_dir = sync_scene.make_output_dir(args.capture_root, None)
    sync_args = make_sync_capture_args(args, output_dir)
    provider = sync_scene.make_robot_provider(sync_args)
    box_detection = None
    current_jnt_values = None
    current_tcp_pos = None
    current_jaw_width = None
    try:
        robot_status = conn_status.check_robot_provider(provider, mock=args.mock)
        conn_status.print_status(robot_status, prefix="[sim_pipeline]")
        if not robot_status.ok:
            raise ConnectionError(robot_status.line())

        sync_scene.read_robot_snapshot(provider, "Initial robot")
        pcd, rgb_path, colored_ply_path, camera_status = conn_status.capture_mech_eye_pointcloud_checked(
            output_dir=output_dir,
            ply_out=sync_args.ply_out,
            depth_scale=sync_args.depth_scale,
            depth_trunc=sync_args.depth_trunc,
        )
        conn_status.print_status(conn_status.LiveConnectionStatus([camera_status]), prefix="[sim_pipeline]")

        points, colors = sync_scene.open3d_to_numpy(pcd)
        raw_count = len(points)
        points_world = sync_scene.transform_points(points, sync_scene.CAM_TO_WORLD)
        world_ply_path = output_dir / "world_colored_pointcloud.ply"
        sync_scene.save_numpy_pointcloud(points_world, colors, world_ply_path)
        print(f"Saved RGB image to: {rgb_path}")
        print(f"Saved camera-frame colored point cloud to: {colored_ply_path}")
        print(f"Saved world-frame colored point cloud to: {world_ply_path}")
        sync_scene.print_camera_info(frame="camera", points_world=points_world, source_path=colored_ply_path)
        scene_data = sync_scene.CameraSceneData(
            points_world=points_world,
            colors=colors,
            raw_point_count=raw_count,
            frame="camera",
            rgb_path=rgb_path,
            colored_ply_path=colored_ply_path,
            world_ply_path=world_ply_path,
            output_dir=output_dir,
        )

        try:
            box_detection = sync_scene.detect_box(sync_args, scene_data)
        except Exception as exc:
            print(
                "[sim_pipeline] Warning: box detection during C sync failed; "
                f"object extraction may retry from the point cloud. {type(exc).__name__}: {exc}"
            )
        current_jnt_values, current_tcp_pos, current_jaw_width = sync_scene.read_robot_snapshot(provider, "Scene robot")
        sync_scene.write_summary(
            sync_args,
            scene_data,
            current_jnt_values,
            current_tcp_pos,
            current_jaw_width,
            box_detection,
        )
    finally:
        provider.close()

    args.capture_dir = output_dir
    args.image = scene_data.rgb_path if scene_data.rgb_path is not None else args.image
    args.ply = scene_data.world_ply_path if scene_data.world_ply_path is not None else scene_data.colored_ply_path
    detected_box_transform = output_dir / "detected_box_transform.txt"
    args.box_transform = detected_box_transform if detected_box_transform.exists() else None
    args.object_summary = None
    args.object_output_dir = output_dir / "box_object_extraction"
    if current_jnt_values is not None:
        args.start_conf_deg = np.degrees(np.asarray(current_jnt_values, dtype=float))

    ctx = prepare_interactive_pipeline_context(args)
    metadata = {
        "output_dir": output_dir,
        "scene_data": scene_data,
        "box_detection": box_detection,
        "current_jnt_values": current_jnt_values,
        "current_tcp_pos": current_tcp_pos,
        "current_jaw_width": current_jaw_width,
        "robot_status": robot_status,
        "camera_status": camera_status,
    }
    return ctx, metadata


def find_close_index(mot_data) -> Optional[int]:
    values = [None if value is None else float(np.asarray(value).reshape(-1)[0]) for value in mot_data.ev_list]
    for index in range(1, len(values)):
        prev = values[index - 1]
        current = values[index]
        if prev is not None and current is not None and current < prev - 1e-5:
            return index
    return None


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
    return sim_pick.clamp_jaw_width(robot, grasp_jaw_width(grasp) * 1.2)


def decorate_grasp_for_rtde(robot, grasp, grasp_index: int):
    jaw_width = grasp_jaw_width(grasp)
    setattr(grasp, "name", f"pickle_grasp_{grasp_index}")
    setattr(grasp, "jaw_width", jaw_width)
    setattr(grasp, "pick_approach_jaw_width", pick_approach_jaw_width_for_grasp(robot, grasp))
    setattr(grasp, "approach_distance", float(sim_pick.APPROACH_DISTANCE))
    return grasp

def build_and_save_rtde_plan(args: argparse.Namespace, planning: PlanningResult, default_dir: Path) -> tuple[rtde_utils.RtdeExecutionPlan, Path]:
    robot = sim_pick.make_robot()
    grasps = sim_pick.load_grasps(robot, sim_pick.GRASP_PICKLE_PATH)
    grasp_index = planning.selected_grasp_index
    if grasp_index is None:
        grasp_index = infer_selected_grasp_index(robot, grasps, planning.mot_data, planning.pick_pose)
    if grasp_index is None or grasp_index < 0 or grasp_index >= len(grasps):
        raise RuntimeError("Could not infer selected grasp for RTDE execution plan.")
    planning.selected_grasp_index = grasp_index
    grasp = decorate_grasp_for_rtde(robot, grasps[grasp_index], grasp_index)
    rtde_plan = rtde_utils.build_pick_place_rtde_plan(
        robot=robot,
        mot_data=planning.mot_data,
        pick_pose=RtdeObjectPose(pos=planning.pick_pose[0], rotmat=planning.pick_pose[1]),
        place_pose=RtdeObjectPose(pos=planning.place_pose[0], rotmat=planning.place_pose[1]),
        grasp=grasp,
    )
    rtde_plan_path = args.rtde_plan_out
    if rtde_plan_path is None:
        rtde_plan_path = default_dir / "pick_place_rtde_plan.json"
    rtde_utils.save_rtde_execution_plan(rtde_plan, rtde_plan_path)
    sim_pick.debug_print(f"Saved RTDE execution plan: {rtde_plan_path}")
    return rtde_plan, rtde_plan_path

def object_icp_result_from_summary(summary: dict) -> ObjectIcpResult:
    summary_path = Path(summary["summary_path"])
    bottle_summary = summary.get("bottle_icp")
    if bottle_summary is None:
        raise RuntimeError("Bottle ICP did not produce a result; press D again after selecting an object/template.")
    bottle_model_path = Path(bottle_summary.get("bottle_stl") or DEFAULT_OBJECT_MODEL_PATH)
    result = ObjectIcpResult(
        output_dir=summary_path.parent,
        summary_path=summary_path,
        bottle_transform_path=Path(bottle_summary["icp_transform_path"]),
        box_transform_path=Path(summary["box_transform_used"]),
        bottle_model_path=bottle_model_path,
    )
    for label, path in (
        ("summary", result.summary_path),
        ("bottle transform", result.bottle_transform_path),
        ("box transform", result.box_transform_path),
        ("bottle model", result.bottle_model_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    return result


def read_object_summary(summary_path: Path) -> ObjectIcpResult:
    if not summary_path.exists():
        raise FileNotFoundError(f"Object summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bottle_summary = summary.get("bottle_icp")
    if bottle_summary is None:
        raise RuntimeError(f"Summary has no bottle_icp result: {summary_path}")

    bottle_transform_path = Path(bottle_summary["icp_transform_path"])
    box_transform_path = Path(summary["box_transform_used"])
    bottle_model_path = Path(bottle_summary.get("bottle_stl") or DEFAULT_OBJECT_MODEL_PATH)

    for label, path in (
        ("bottle transform", bottle_transform_path),
        ("box transform", box_transform_path),
        ("bottle model", bottle_model_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    return ObjectIcpResult(
        output_dir=summary_path.parent,
        summary_path=summary_path,
        bottle_transform_path=bottle_transform_path,
        box_transform_path=box_transform_path,
        bottle_model_path=bottle_model_path,
    )


def build_extraction_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(BOX_OBJECT_SCRIPT),
        "--capture-root",
        str(args.capture_root),
        "--output-dir",
        str(output_dir),
        "--backend",
        args.backend,
        "--keep",
        args.keep,
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--bottle-template",
        args.bottle_template,
        "--bottle-stl",
        str(args.bottle_stl),
    ]
    append_path(cmd, "--capture-dir", args.capture_dir)
    append_path(cmd, "--ply", args.ply)
    append_path(cmd, "--image", args.image)
    append_path(cmd, "--box-transform", args.box_transform)
    append_path(cmd, "--mask", args.mask)
    append_path(cmd, "--bottle-template-ply", args.bottle_template_ply)
    append_value(cmd, "--segment-box", args.segment_box)
    append_value(cmd, "--model", args.model)
    append_value(cmd, "--device", args.device)
    append_value(cmd, "--bottle-voxel", args.bottle_voxel)
    append_value(cmd, "--bottle-icp-max-iteration", args.bottle_icp_max_iteration)
    for point in args.point:
        cmd.extend(["--point", point])
    cmd.append("--auto-segment-box" if args.auto_segment_box else "--no-auto-segment-box")
    cmd.append("--show-viewer" if args.show_object_viewer else "--no-show-viewer")
    cmd.append("--show-box-model" if args.show_box_model else "--no-show-box-model")
    if args.no_gui:
        cmd.append("--no-gui")
    if not args.bottle_template_prompt_gui:
        cmd.append("--no-bottle-template-prompt-gui")
    return cmd


def run_or_reuse_object_icp(args: argparse.Namespace) -> ObjectIcpResult:
    if args.object_summary is not None:
        return read_object_summary(args.object_summary)

    output_dir = resolve_object_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_extraction_command(args, output_dir)
    run_command("run box/object extraction and bottle ICP", cmd)
    return read_object_summary(output_dir / "box_object_extraction_summary.json")


def load_homomat(path: Path, label: str) -> np.ndarray:
    homomat = np.asarray(np.loadtxt(path), dtype=float)
    if homomat.shape != (4, 4):
        raise ValueError(f"{label} transform must be 4x4, got {homomat.shape}: {path}")
    if not np.all(np.isfinite(homomat)):
        raise ValueError(f"{label} transform contains NaN or inf: {path}")
    return homomat


def homomat_to_pose(homomat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return homomat[:3, 3].copy(), homomat[:3, :3].copy()


def resolve_place_pose(
    args: argparse.Namespace,
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



def build_obstacle_lists(args: argparse.Namespace, box_homomat: np.ndarray) -> tuple[list[object], list[object]]:
    planning_obstacles: list[object] = []
    display_obstacles: list[object] = []
    if not args.no_env:
        table = sim_pick.make_table_obstacle()
        planning_obstacles.append(table)
        display_obstacles.append(table)
    detected_box_visual = make_detected_box_visual_model(box_homomat)
    detected_box_panels = make_concave_box_collision_obstacles(box_homomat)
    display_obstacles.append(detected_box_visual)
    display_obstacles.extend(detected_box_panels)
    if args.box_obstacle:
        planning_obstacles.extend(detected_box_panels)
    return planning_obstacles, display_obstacles


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


def configure_pick_place(args: argparse.Namespace, object_model_path: Path) -> None:
    sim_pick.IK_SOLVER = "ikfast"
    sim_pick.OBJECT_MODEL_PATH = object_model_path
    sim_pick.USE_RRT = bool(args.use_rrt)
    sim_pick.VALIDATE_COLLISIONS = bool(args.collision_check)
    sim_pick.VISUALIZE_RESULT = not bool(args.dry_run)
    sim_pick.VISUALIZE_FAILURE = bool(args.visualize_failure)
    sim_pick.USE_REASONED_COMMON_GRASPS = True
    sim_pick.PRINT_WRS_SUMMARY = True
    sim_pick.DEBUG_IKFAST_FRAME = False
    if args.open_jaw is not None:
        sim_pick.OPEN_JAW_WIDTH = float(args.open_jaw)
    if args.approach_distance is not None:
        sim_pick.APPROACH_DISTANCE = float(args.approach_distance)
    if args.start_conf_deg is not None:
        sim_pick.DEFAULT_HOME_CONF = np.radians(np.asarray(args.start_conf_deg, dtype=float))


def xy_angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    vector_a = np.asarray(vector_a, dtype=float).reshape(2)
    vector_b = np.asarray(vector_b, dtype=float).reshape(2)
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return float("inf")
    cosine = float(np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def robot_arm_base_pos(robot) -> np.ndarray:
    arm = getattr(robot, "arm", getattr(robot, "manipulator", None))
    if arm is None:
        return np.zeros(3)
    return np.asarray(getattr(arm, "pos", np.zeros(3)), dtype=float).reshape(3)


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


def tcp_base_xy_angle_for_conf(robot, conf: np.ndarray, base_pos: np.ndarray, target_xy: np.ndarray) -> float:
    tcp_pos, _tcp_rotmat = robot.fk(jnt_values=np.asarray(conf, dtype=float))
    tcp_vector_xy = (np.asarray(tcp_pos, dtype=float).reshape(3) - base_pos)[:2]
    return xy_angle_deg(tcp_vector_xy, target_xy)


def make_world_z_lowering_path(robot, start_conf: np.ndarray) -> tuple[list[np.ndarray], float]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    lowering_distance = float(sim_pick.PICK_APPROACH_DEPART_DISTANCE) * 0.75
    step_count = max(1, int(np.ceil(lowering_distance / max(float(sim_pick.LINEAR_GRANULARITY), 1e-4))))
    start_tcp_pos, start_tcp_rotmat = robot.fk(jnt_values=start_conf)
    start_tcp_pos = np.asarray(start_tcp_pos, dtype=float).reshape(3)
    start_tcp_rotmat = np.asarray(start_tcp_rotmat, dtype=float).reshape(3, 3)
    lowering_path: list[np.ndarray] = []
    seed_conf = start_conf
    for step_id in range(1, step_count + 1):
        ratio = float(step_id) / float(step_count)
        target_pos = start_tcp_pos - sim_pick.rm.const.z_ax * lowering_distance * ratio
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

    raise RuntimeError("Could not build post-pick first-joint alignment path.")


def gen_pick_only_path(robot, obj_cmodel, grasps, pick_pose, obstacle_list) -> tuple[Optional[int], Any]:
    from wrs import ppp

    sim_pick.debug_print("Starting pick-only planner...")
    sim_pick.debug_print(f"  grasps={len(grasps)}, use_rrt={sim_pick.USE_RRT}, validate_collisions={sim_pick.VALIDATE_COLLISIONS}")
    sim_pick.debug_print(f"  reason_pick_grasps={sim_pick.USE_REASONED_COMMON_GRASPS}")
    sim_pick.debug_print(
        f"  approach/depart distance={sim_pick.APPROACH_DISTANCE:.3f} m, "
        f"linear_granularity={sim_pick.LINEAR_GRANULARITY:.3f} m"
    )
    sim_pick.debug_print("  directions: pick approach=grasp TCP +Z, pick depart=+Z, post-pick lowering=world -Z")

    planner = ppp.PickPlacePlanner(robot)
    candidate_indices = list(range(len(grasps)))
    if sim_pick.USE_REASONED_COMMON_GRASPS:
        grasp_collection = sim_pick.make_grasp_collection(robot, grasps)
        is_debug = True
        candidate_indices = list(
            planner.reason_common_gids(
                grasp_collection=grasp_collection,
                goal_pose_list=[pick_pose],
                obstacle_list=obstacle_list,
                toggle_dbg=is_debug,
            )
        )
        if is_debug:
            return
        sim_pick.debug_print(f"  reasoned pick grasps: {len(candidate_indices)}/{len(grasps)}")
        if candidate_indices:
            preview = candidate_indices[:20]
            suffix = "..." if len(candidate_indices) > len(preview) else ""
            sim_pick.debug_print(f"  pick grasp indices: {preview}{suffix}")
        else:
            raise RuntimeError("Pick planner failed: no pick-feasible grasps.")

    for grasp_index in candidate_indices:
        grasp = grasps[grasp_index]
        grasp_width = grasp_jaw_width(grasp)
        pick_approach_jaw_width = pick_approach_jaw_width_for_grasp(robot, grasp)
        robot.goto_given_conf(sim_pick.DEFAULT_HOME_CONF, ee_values=pick_approach_jaw_width)
        sim_pick.debug_print(
            f"  pickle_grasp_{grasp_index}: grasp jaw={grasp_width:.5f} m, "
            f"pick approach jaw={pick_approach_jaw_width:.5f} m"
        )
        mot_data = planner.gen_pick_and_moveto(
            obj_cmodel=obj_cmodel.copy(),
            grasp=grasp,
            moveto_pose_list=[],
            moveto_approach_direction_list=[],
            moveto_approach_distance_list=[],
            moveto_depart_direction_list=[],
            moveto_depart_distance_list=[],
            start_jnt_values=sim_pick.DEFAULT_HOME_CONF,
            pick_approach_jaw_width=pick_approach_jaw_width,
            pick_approach_direction=None,
            pick_approach_distance=sim_pick.APPROACH_DISTANCE,
            pick_depart_direction=sim_pick.rm.const.z_ax,
            pick_depart_distance=sim_pick.PICK_APPROACH_DEPART_DISTANCE,
            linear_granularity=sim_pick.LINEAR_GRANULARITY,
            obstacle_list=obstacle_list,
            use_rrt=sim_pick.USE_RRT,
            toggle_dbg=False,
        )
        if mot_data is None:
            sim_pick.debug_print(f"  pickle_grasp_{grasp_index}: pick approach/depart failed")
            continue

        pick_frame_count = len(mot_data.jv_list) if hasattr(mot_data, "jv_list") else len(mot_data)
        sim_pick.debug_print(f"Pick approach/depart produced {pick_frame_count} frame(s); validating collision...")
        collisions = sim_pick.validate_motion(sim_pick.make_robot(), mot_data, obstacle_list) if sim_pick.VALIDATE_COLLISIONS else []
        if collisions:
            first_collision = sim_pick.format_collision(collisions[0])
            sim_pick.debug_print(
                f"  pickle_grasp_{grasp_index}: pick approach/depart collision validation failed; "
                f"{len(collisions)} frame(s), first={first_collision}"
            )
            continue

        try:
            align_path, final_angle_deg = make_first_joint_alignment_path(
                robot,
                np.asarray(mot_data.jv_list[-1], dtype=float),
                mot_data.ev_list[-1],
                obstacle_list,
            )
        except Exception as exc:
            sim_pick.debug_print(f"  pickle_grasp_{grasp_index}: post-pick first-joint alignment failed: {exc}")
            continue
        if align_path:
            mot_data.extend(
                align_path,
                ev_list=[mot_data.ev_list[-1]] * len(align_path),
                mesh_list=[],
            )
            sim_pick.debug_print(
                f"  Post-pick first-joint alignment: {len(align_path)} frame(s), "
                f"final angle={final_angle_deg:.2f} deg"
            )
        else:
            sim_pick.debug_print(
                f"  Post-pick first-joint alignment already within threshold: "
                f"angle={final_angle_deg:.2f} deg"
            )

        pre_lowering_conf = np.asarray(mot_data.jv_list[-1], dtype=float).copy()
        lowering_path, lowering_distance = make_world_z_lowering_path(
            robot,
            pre_lowering_conf,
        )
        if not lowering_path:
            sim_pick.debug_print(
                f"  pickle_grasp_{grasp_index}: world -Z lowering before open failed; IK not solvable"
            )
            continue
        mot_data.extend(
            lowering_path,
            ev_list=[mot_data.ev_list[-1]] * len(lowering_path),
            mesh_list=[],
        )
        sim_pick.debug_print(
            f"  World -Z lowering before open without collision check: {len(lowering_path)} frame(s), "
            f"distance={lowering_distance:.4f} m"
        )

        final_open_width = sim_pick.clamp_jaw_width(robot, sim_pick.OPEN_JAW_WIDTH)
        mot_data.extend(
            [np.asarray(mot_data.jv_list[-1], dtype=float).copy()],
            ev_list=[final_open_width],
            mesh_list=[],
        )
        setattr(mot_data, "force_final_open_gripper", True)
        setattr(mot_data, "forced_open_index", len(mot_data.jv_list) - 1)
        sim_pick.debug_print(f"  Forced open gripper frame appended before lift: jaw={final_open_width:.5f} m")

        lift_path = [np.asarray(conf, dtype=float).copy() for conf in reversed(lowering_path[:-1])]
        lift_path.append(pre_lowering_conf)
        mot_data.extend(
            lift_path,
            ev_list=[final_open_width] * len(lift_path),
            mesh_list=[],
        )
        sim_pick.debug_print(
            f"  World +Z lift after open without collision check: {len(lift_path)} frame(s), "
            f"distance={lowering_distance:.4f} m; returned to pre-lowering pose"
        )

        frame_count = len(mot_data.jv_list) if hasattr(mot_data, "jv_list") else len(mot_data)
        sim_pick.debug_print(
            f"Pick planner produced {frame_count} frame(s), including post-pick first-joint alignment, "
            "world -Z lowering, forced open, and world +Z lift."
        )
        sim_pick.debug_print("Pick planner succeeded.")
        sim_pick.debug_print(f"  selected grasp: pickle_grasp_{grasp_index}")
        return grasp_index, mot_data

    raise RuntimeError("Pick planner failed: no grasp produced a valid pick approach/depart path.")


def print_pick_only_summary(mot_data, grasp_count: int, selected_grasp_index: int, pick_pose, obstacle_names: list[str]) -> None:
    sim_pick.debug_print("Pick-only planner path ready.")
    sim_pick.debug_print(f"Loaded grasps: {grasp_count}")
    if selected_grasp_index is None:
        sim_pick.debug_print("Selected grasp: unknown")
    else:
        sim_pick.debug_print(f"Selected grasp: pickle_grasp_{selected_grasp_index}")
    sim_pick.debug_print(f"Frames: {len(mot_data.jv_list)}")
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

def run_or_skip_plan(args: argparse.Namespace, icp: ObjectIcpResult) -> Optional[PlanningResult]:
    if args.skip_plan:
        return None

    object_model_path = args.object_model if args.object_model is not None else icp.bottle_model_path
    object_model_path = object_model_path.resolve()
    if not object_model_path.exists():
        raise FileNotFoundError(f"Object model not found: {object_model_path}")
    if not sim_pick.GRASP_PICKLE_PATH.exists():
        raise FileNotFoundError(f"Grasp pickle not found: {sim_pick.GRASP_PICKLE_PATH}")

    bottle_homomat = load_homomat(icp.bottle_transform_path, "bottle ICP")
    box_homomat = load_homomat(icp.box_transform_path, "box")
    pick_pose = homomat_to_pose(bottle_homomat)
    place_pose, place_pos_source, place_rot_source = resolve_place_pose(args, pick_pose)

    configure_pick_place(args, object_model_path)

    sim_pick.debug_print("=== sim_bottle_pick_place_from_box_object_icp: pick-only planner run ===")
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

    obstacle_list, display_obstacle_list = build_obstacle_lists(args, box_homomat)
    obstacle_names = [getattr(obstacle, "name", type(obstacle).__name__) for obstacle in obstacle_list]
    display_obstacle_names = [getattr(obstacle, "name", type(obstacle).__name__) for obstacle in display_obstacle_list]
    sim_pick.debug_print(f"Collision obstacles: {obstacle_names}")
    sim_pick.debug_print(f"Visualization context: {display_obstacle_names}")
    precheck_start_collision(robot, obstacle_list)

    obj_cmodel = sim_pick.make_object_model(object_model_path, pick_pose, name="icp_bottle_pick_place")
    selected_grasp_index, mot_data = gen_pick_only_path(
        robot,
        obj_cmodel,
        grasps,
        pick_pose,
        obstacle_list,
    )
    if selected_grasp_index is None:
        selected_grasp_index = infer_selected_grasp_index(robot, grasps, mot_data, pick_pose)
    print_pick_only_summary(mot_data, len(grasps), selected_grasp_index, pick_pose, obstacle_names)
    if sim_pick.VALIDATE_COLLISIONS:
        sim_pick.debug_print("Collision check: passed for pick approach/depart; post-pick J1 alignment, world -Z lowering, and world +Z lift are unchecked.")

    planning = PlanningResult(
        selected_grasp_index=selected_grasp_index,
        mot_data=mot_data,
        pick_pose=pick_pose,
        place_pose=place_pose,
        object_model_path=object_model_path,
        obstacle_names=obstacle_names,
        place_pos_source=place_pos_source,
        place_rot_source=place_rot_source,
    )
    planning.rtde_plan, planning.rtde_plan_path = build_and_save_rtde_plan(args, planning, icp.output_dir)

    if sim_pick.VISUALIZE_RESULT:
        sim_pick.visualize_result(mot_data, pick_pose, place_pose, display_obstacle_list)

    return planning


def write_summary(args: argparse.Namespace, icp: ObjectIcpResult, planning: Optional[PlanningResult]) -> Path:
    summary_path = args.summary_out if args.summary_out is not None else icp.output_dir / "sim_bottle_pick_place_summary.json"
    summary = {
        "object_summary_path": str(icp.summary_path),
        "bottle_transform_path": str(icp.bottle_transform_path),
        "box_transform_path": str(icp.box_transform_path),
        "bottle_model_path": str(icp.bottle_model_path),
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
                "visualized_result": not bool(args.dry_run),
            }
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


class InteractiveBottlePickPlaceApp:
    def __init__(
        self,
        args: argparse.Namespace,
        ctx: Optional[box_object_icp.PipelineContext] = None,
        initial_icp: Optional[ObjectIcpResult] = None,
    ):
        import wrs.modeling.geometric_model as mgm
        import wrs.visualization.panda.world as wd
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        self.args = args
        self.ctx = ctx
        self.icp_result = initial_icp
        self.planning_result: Optional[PlanningResult] = None
        self.environment_stale = False
        self.running = False
        self.detect_attempt_count = 0
        self.static_scene_models: list[object] = []
        self.scene_obstacle_models: list[object] = []
        self.robot_sync_models: list[object] = []
        self.detection_models: list[object] = []
        self.plan_models: list[object] = []
        self.animation_data = None
        self.animation_task_name = "sim_bottle_pick_place_interactive_animation"
        self.mgm = mgm

        scene_points = self.compute_scene_points()
        cam_pos, lookat_pos, extent = box_object_icp.compute_camera_from_points(scene_points)
        self.base = wd.World(cam_pos=cam_pos, lookat_pos=lookat_pos, w=1280, h=720)

        frame_length = max(extent * 0.25, 0.03)
        frame_radius = max(frame_length * 0.015, 0.0005)
        mgm.gen_frame(ax_length=frame_length, ax_radius=frame_radius).attach_to(self.base)
        self.attach_static_pointclouds()
        self.attach_scene_obstacles()

        initial_text = "Offline scene loaded. Press D for segmentation/template/ICP, then P to plan."
        self.status_text = OnscreenText(
            text=initial_text,
            pos=(-1.28, 0.92),
            align=TextNode.ALeft,
            scale=0.044,
            fg=(0.02, 0.02, 0.02, 1.0),
            mayChange=True,
        )
        self.connection_status_text = OnscreenText(
            text="Offline mode: saved capture only",
            pos=(-1.28, 0.86),
            align=TextNode.ALeft,
            scale=0.036,
            fg=(0.02, 0.02, 0.02, 1.0),
            mayChange=True,
        )
        self.base.accept("d", self.run_detection)
        self.base.accept("p", self.run_plan)

        if self.icp_result is not None:
            self.clear_detection_models()
            self.attach_start_and_place_models(self.icp_result)
        print("Offline WRS viewer is ready. Press D for segmentation/template/ICP, P to plan.")

    def compute_scene_points(self) -> np.ndarray:
        if self.ctx is not None:
            static_mask = self.ctx.candidate_mask | self.ctx.removed_mask
            scene_points = self.ctx.capture.points_world[static_mask]
            if len(scene_points) > 0:
                return scene_points
            return self.ctx.capture.points_world
        if self.icp_result is not None:
            try:
                pick_pose = homomat_to_pose(load_homomat(self.icp_result.bottle_transform_path, "bottle ICP"))
                place_pose, _place_pos_source, _place_rot_source = resolve_place_pose(self.args, pick_pose)
                return np.vstack([pick_pose[0], place_pose[0]])
            except Exception:
                pass
        return np.array([[0.58, -0.12, 0.28]], dtype=np.float64)

    @staticmethod
    def detach_models(models: list[object]) -> None:
        for model in models:
            for method_name in ("detach", "remove"):
                method = getattr(model, method_name, None)
                if method is None:
                    continue
                try:
                    method()
                    break
                except Exception:
                    continue
        models.clear()

    def clear_detection_models(self) -> None:
        self.detach_models(self.detection_models)

    def clear_plan_models(self) -> None:
        try:
            self.base.taskMgr.remove(self.animation_task_name)
        except Exception:
            pass
        if self.animation_data is not None:
            for attr in ("mesh_model", "obj_model"):
                model = getattr(self.animation_data, attr, None)
                if model is not None:
                    self.detach_models([model])
            self.animation_data = None
        self.detach_models(self.plan_models)

    def clear_static_scene_models(self) -> None:
        self.detach_models(self.static_scene_models)

    def clear_scene_obstacle_models(self) -> None:
        self.detach_models(self.scene_obstacle_models)

    def clear_robot_sync_models(self) -> None:
        self.detach_models(self.robot_sync_models)

    def clear_synced_scene_models(self) -> None:
        self.clear_plan_models()
        self.clear_detection_models()
        self.clear_static_scene_models()
        self.clear_scene_obstacle_models()
        self.clear_robot_sync_models()
    def attach_static_pointclouds(self) -> None:
        if self.ctx is None:
            return
        args = self.ctx.args
        candidate_points = self.ctx.capture.points_world[self.ctx.candidate_mask]
        removed_points = self.ctx.capture.points_world[self.ctx.removed_mask]
        candidate_points, _ = box_object_icp.voxel_downsample_arrays(candidate_points, None, args.candidate_voxel)
        removed_points, _ = box_object_icp.voxel_downsample_arrays(removed_points, None, args.removed_voxel)
        max_points = int(max(0, getattr(self.args, "scene_max_points", 0)))
        total_points = len(candidate_points) + len(removed_points)
        if max_points > 0 and total_points > max_points:
            candidate_limit = max(1, int(max_points * len(candidate_points) / max(1, total_points)))
            removed_limit = max(1, max_points - candidate_limit)
            candidate_points, _ = sync_scene.downsample_points(candidate_points, None, candidate_limit)
            removed_points, _ = sync_scene.downsample_points(removed_points, None, removed_limit)
        point_size = float(getattr(self.args, "scene_point_size", args.point_size))
        if len(removed_points) > 0:
            removed_model = self.mgm.gen_pointcloud(
                removed_points,
                rgba=np.array([0.55, 0.55, 0.55, 0.7]),
                point_size=point_size,
            )
            removed_model.attach_to(self.base)
            self.static_scene_models.append(removed_model)
        if len(candidate_points) > 0:
            candidate_model = self.mgm.gen_pointcloud(
                candidate_points,
                rgba=np.array([0.0, 0.85, 0.15, 0.78]),
                point_size=point_size,
            )
            candidate_model.attach_to(self.base)
            self.static_scene_models.append(candidate_model)

    def attach_scene_obstacles(self) -> None:
        try:
            if self.ctx is not None:
                box_homomat = self.ctx.box_transform
                print(f"[sim_pipeline] DEBUG attach_scene_obstacles: using ctx.box_transform, shape={np.asarray(box_homomat).shape}")
            elif self.icp_result is not None:
                box_homomat = load_homomat(self.icp_result.box_transform_path, "box")
                print(f"[sim_pipeline] DEBUG attach_scene_obstacles: using icp_result path, shape={np.asarray(box_homomat).shape}")
            else:
                print("[sim_pipeline] DEBUG attach_scene_obstacles: ctx and icp_result both None, returning")
                return
            _planning_obstacles, display_obstacles = build_obstacle_lists(self.args, box_homomat)
            print(f"[sim_pipeline] DEBUG attach_scene_obstacles: built {len(display_obstacles)} display obstacles")
            for obstacle in display_obstacles:
                obstacle.attach_to(self.base)
                self.scene_obstacle_models.append(obstacle)
            print(f"[sim_pipeline] DEBUG attach_scene_obstacles: attached {len(self.scene_obstacle_models)} scene obstacle models")
        except Exception as exc:
            import traceback
            print(f"[sim_pipeline] Warning: could not attach scene obstacles: {exc}")
            traceback.print_exc()

    def attach_synced_robot(self, jnt_values: Optional[np.ndarray], jaw_width: Optional[float]) -> None:
        if jnt_values is None:
            return
        from wrs.robot_sim.robots.ur7e.ur7e_withouttable_dh76 import UR7EDH76

        self.clear_robot_sync_models()
        robot = UR7EDH76(enable_cc=True, ik_solver="ikfast")
        robot.goto_given_conf(jnt_values=np.asarray(jnt_values, dtype=float))
        if jaw_width is not None:
            try:
                robot.jaw_to(jawwidth=sim_pick.clamp_jaw_width(robot, float(jaw_width)))
            except Exception as exc:
                print(f"[sim_pipeline] Warning: could not set synced jaw width: {exc}")
        robot_model = robot.gen_meshmodel(
            alpha=0.78,
            toggle_tcp_frame=True,
            toggle_flange_frame=False,
            toggle_jnt_frames=False,
        )
        robot_model.attach_to(self.base)
        self.robot_sync_models.append(robot_model)
        print(f"[sim_pipeline] synced robot display joints(deg): {sim_pick.format_jnts_deg(jnt_values, digits=2)}")

    def refresh_synced_scene_display(self, sync_metadata: dict) -> None:
        self.clear_synced_scene_models()
        self.attach_static_pointclouds()
        self.attach_scene_obstacles()
        self.attach_synced_robot(
            sync_metadata.get("current_jnt_values"),
            sync_metadata.get("current_jaw_width"),
        )

    def resolve_object_model_path(self, icp: ObjectIcpResult) -> Path:
        object_model_path = self.args.object_model if self.args.object_model is not None else icp.bottle_model_path
        object_model_path = object_model_path.resolve()
        if not object_model_path.exists():
            raise FileNotFoundError(f"Object model not found: {object_model_path}")
        return object_model_path

    def load_pick_and_place_poses(
        self,
        icp: ObjectIcpResult,
    ) -> tuple[Path, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], str, str]:
        object_model_path = self.resolve_object_model_path(icp)
        bottle_homomat = load_homomat(icp.bottle_transform_path, "bottle ICP")
        pick_pose = homomat_to_pose(bottle_homomat)
        place_pose, place_pos_source, place_rot_source = resolve_place_pose(self.args, pick_pose)
        return object_model_path, pick_pose, place_pose, place_pos_source, place_rot_source

    def attach_start_and_place_models(self, icp: ObjectIcpResult) -> None:
        object_model_path, pick_pose, place_pose, place_pos_source, place_rot_source = self.load_pick_and_place_poses(icp)
        start_model = sim_pick.make_object_model(
            object_model_path,
            pick_pose,
            name="estimated_pick_start_bottle",
            alpha=0.55,
            rgb=np.array([1.0, 0.76, 0.18]),
        )
        place_model = sim_pick.make_object_model(
            object_model_path,
            place_pose,
            name="planned_place_goal_bottle",
            alpha=0.32,
            rgb=np.array([0.2, 0.9, 0.45]),
        )
        start_model.attach_to(self.base)
        place_model.attach_to(self.base)
        self.detection_models.extend([start_model, place_model])

        frame_len = 0.085
        for pose, color in ((pick_pose, np.array([1.0, 0.76, 0.18])), (place_pose, np.array([0.2, 0.9, 0.45]))):
            frame = self.mgm.gen_frame(pos=pose[0], rotmat=pose[1], ax_length=frame_len, ax_radius=0.002)
            frame.attach_to(self.base)
            self.detection_models.append(frame)
            marker = self.mgm.gen_sphere(pos=pose[0], radius=0.01, rgb=color, alpha=0.85)
            marker.attach_to(self.base)
            self.detection_models.append(marker)
        if np.linalg.norm(place_pose[0] - pick_pose[0]) > 1e-8:
            arrow = self.mgm.gen_arrow(
                spos=pick_pose[0],
                epos=place_pose[0],
                rgb=np.array([0.2, 0.45, 1.0]),
                alpha=0.72,
                stick_radius=0.004,
            )
            arrow.attach_to(self.base)
            self.detection_models.append(arrow)
        print(
            "[sim_pipeline] start/place preview ready: "
            f"pick={sim_pick.format_vec(pick_pose[0], digits=6)}, "
            f"place={sim_pick.format_vec(place_pose[0], digits=6)} "
            f"({place_pos_source}, rot={place_rot_source})"
        )

    def attach_detection_result(self, summary: dict, selected_mask: np.ndarray) -> None:
        self.clear_detection_models()
        if self.ctx is not None:
            selected_points = self.ctx.capture.points_world[selected_mask]
            selected_points, _ = box_object_icp.voxel_downsample_arrays(
                selected_points,
                None,
                self.ctx.args.selected_voxel,
            )
            if len(selected_points) > 0:
                selected_model = self.mgm.gen_pointcloud(
                    selected_points,
                    rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                    point_size=max(self.ctx.args.point_size, 0.0025),
                )
                selected_model.attach_to(self.base)
                self.detection_models.append(selected_model)
        self.icp_result = object_icp_result_from_summary(summary)
        self.clear_scene_obstacle_models()
        self.attach_scene_obstacles()
        self.attach_start_and_place_models(self.icp_result)
        summary_path = write_summary(self.args, self.icp_result, None)
        print(f"[sim_pipeline] sim summary after ICP: {summary_path}")

    def update_connection_status_display(self, metadata: dict) -> None:
        items = []
        robot_status = metadata.get("robot_status")
        if robot_status is not None:
            for check in robot_status.checks:
                if check.name == "robot RTDE control":
                    items.append(f"RTDE-C:{'OK' if check.ok else 'FAIL'}")
                elif check.name == "robot RTDE receive":
                    items.append(f"RTDE-R:{'OK' if check.ok else 'FAIL'}")
                elif check.name == "DH76 gripper":
                    items.append(f"Gripper:{'OK' if check.ok else 'FAIL'}")
        camera_status = metadata.get("camera_status")
        if camera_status is not None:
            items.append(f"Camera:{'OK' if camera_status.ok else 'FAIL'}")
        if not items:
            items.append("unchecked")
        self.connection_status_text.setText("Connections: " + " | ".join(items))
    def run_sync_capture(self) -> None:
        self.status_text.setText(
            "This script is offline-only. Use real_bottle_pick_place_interactive.py for C sync/capture."
        )
        print("[sim_pipeline] C is disabled in offline sim. Use yanjiuyuan/real_bottle_pick_place_interactive.py.")
    def run_execute(self) -> None:
        self.status_text.setText(
            "This script is offline-only. Use real_bottle_pick_place_interactive.py for O execution."
        )
        print("[sim_pipeline] O is disabled in offline sim. Use yanjiuyuan/real_bottle_pick_place_interactive.py.")
    def run_detection(self) -> None:
        if self.ctx is None:
            self.status_text.setText("No saved capture context. Start with --capture-dir/--ply or the newest saved capture.")
            return
        if self.running:
            print("[sim_pipeline] Another operation is already running; ignoring D key.")
            return
        self.running = True
        self.detect_attempt_count += 1
        self.status_text.setText(f"Attempt {self.detect_attempt_count}: segmenting, choosing template, running ICP...")
        self.planning_result = None
        self.clear_plan_models()
        self.clear_detection_models()
        try:
            summary, _masks, selected_mask = box_object_icp.run_segmentation_and_bottle_icp_attempt(self.ctx)
            self.attach_detection_result(summary, selected_mask)
            bottle_summary = summary.get("bottle_icp") or {}
            if bottle_summary:
                self.status_text.setText(
                    f"Attempt {self.detect_attempt_count}: ICP done. "
                    f"Template={bottle_summary['template']} fitness={bottle_summary['icp_fitness']:.4f}. Press P to plan."
                )
            else:
                self.status_text.setText(f"Attempt {self.detect_attempt_count}: detection done. Press P to plan.")
        except Exception as exc:
            self.status_text.setText(f"Attempt {self.detect_attempt_count}: failed: {exc}. Press D to retry.")
            print(f"[sim_pipeline] Detection attempt {self.detect_attempt_count} failed: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False

    def run_plan(self) -> None:
        if self.running:
            print("[sim_pipeline] Another operation is already running; ignoring P key.")
            return
        if self.args.skip_plan:
            self.status_text.setText("Planning is disabled by --skip-plan.")
            return
        if self.icp_result is None:
            self.status_text.setText("No estimated start pose yet. Press D first.")
            return
        self.running = True
        self.status_text.setText("Planning pick-only path...")
        self.clear_plan_models()
        try:
            plan_args = copy.copy(self.args)
            plan_args.dry_run = True
            planning = run_or_skip_plan(plan_args, self.icp_result)
            if planning is None:
                self.status_text.setText("Planning skipped.")
                return
            self.planning_result = planning
            self.attach_plan_result(planning)
            summary_path = write_summary(self.args, self.icp_result, planning)
            rtde_path = "" if planning.rtde_plan_path is None else f" RTDE={planning.rtde_plan_path}"
            self.status_text.setText(f"Planning done: {len(planning.mot_data.jv_list)} frames. Offline animation is running.")
            print(f"[sim_pipeline] sim summary after planning: {summary_path}{rtde_path}")
        except Exception as exc:
            self.status_text.setText(f"Planning failed: {exc}. Adjust pose/options, then press P again.")
            print(f"[sim_pipeline] Planning failed: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False

    def attach_plan_result(self, planning: PlanningResult) -> None:
        mot_data = planning.mot_data
        if len(mot_data) == 0:
            print("[sim_pipeline] No motion frames to visualize.")
            return

        marker_robot = sim_pick.make_robot()
        marker_robot.backup_state()
        try:
            for index in range(0, len(mot_data), max(1, sim_pick.RESULT_TRAIL_STRIDE)):
                jnt_values, ee_values, _obj_pose, _mesh = mot_data[index]
                marker_robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
                marker = self.mgm.gen_sphere(
                    pos=marker_robot.gl_tcp_pos,
                    radius=0.006,
                    rgb=np.array([1.0, 0.35, 0.05]),
                    alpha=0.75,
                )
                marker.attach_to(self.base)
                self.plan_models.append(marker)
        finally:
            marker_robot.restore_state()
        self.start_motion_animation(planning)

    def start_motion_animation(self, planning: PlanningResult) -> None:
        object_model_path = planning.object_model_path

        class AnimationData:
            def __init__(self, motion_data):
                self.counter = 0
                self.motion_data = motion_data
                self.robot = sim_pick.make_robot()
                self.mesh_model = None
                self.obj_model = None

        data = AnimationData(planning.mot_data)
        self.animation_data = data

        def update(task):
            if data.mesh_model is not None:
                self.detach_models([data.mesh_model])
                data.mesh_model = None
            if data.obj_model is not None:
                self.detach_models([data.obj_model])
                data.obj_model = None
            if data.counter >= len(data.motion_data):
                data.counter = 0

            jnt_values, ee_values, obj_pose, cached_mesh = data.motion_data[data.counter]
            if cached_mesh is not None:
                data.mesh_model = cached_mesh
            else:
                data.robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
                data.mesh_model = data.robot.gen_meshmodel(
                    alpha=0.72,
                    toggle_tcp_frame=True,
                    toggle_flange_frame=False,
                    toggle_jnt_frames=False,
                )
            if data.mesh_model is not None:
                data.mesh_model.attach_to(self.base)

            if obj_pose is not None and cached_mesh is None:
                data.obj_model = sim_pick.make_object_model(
                    object_model_path,
                    (np.asarray(obj_pose[0], dtype=float), np.asarray(obj_pose[1], dtype=float)),
                    name="interactive_animated_held_object",
                    alpha=0.65,
                    rgb=np.array([0.95, 0.72, 0.18]),
                )
                data.obj_model.attach_to(self.base)

            data.counter += 1
            return task.again

        self.base.taskMgr.doMethodLater(
            sim_pick.RESULT_ANIMATION_INTERVAL,
            update,
            self.animation_task_name,
            appendTask=True,
        )

    def run(self) -> None:
        self.base.run()


def should_run_interactive(args: argparse.Namespace) -> bool:
    if args.interactive is not None:
        return bool(args.interactive)
    return args.object_summary is None and not bool(args.dry_run)


def main() -> None:
    args = parse_args()
    normalize_paths(args)

    if should_run_interactive(args):
        if args.object_summary is not None:
            icp = read_object_summary(args.object_summary)
            app = InteractiveBottlePickPlaceApp(args, initial_icp=icp)
        else:
            ctx = prepare_interactive_pipeline_context(args)
            app = InteractiveBottlePickPlaceApp(args, ctx=ctx)
        app.run()
        return

    icp = run_or_reuse_object_icp(args)
    planning = run_or_skip_plan(args, icp)
    summary_path = write_summary(args, icp, planning)

    print("[sim_pipeline] done")
    print(f"[sim_pipeline] bottle transform: {icp.bottle_transform_path}")
    print(f"[sim_pipeline] box transform: {icp.box_transform_path}")
    if planning is not None:
        print(f"[sim_pipeline] planned frames: {len(planning.mot_data.jv_list)}")
        print(f"[sim_pipeline] place pos: {sim_pick.format_vec(planning.place_pose[0], digits=6)}")
    print(f"[sim_pipeline] summary: {summary_path}")


if __name__ == "__main__":
    main()



