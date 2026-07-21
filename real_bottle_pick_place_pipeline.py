"""
One-command pipeline for box-aware bottle pick-and-place.

The existing scripts still own the heavy work:
  - sync_real_ur7e_mech_eye_box_env.py captures/loads Mech-Eye data and detects the box.
  - box_object_pointcloud_from_saved_capture_bottle_icp.py segments the bottle and estimates its pose.
  - sim_bottle_pick_place_from_box_object_icp.py plans the bottle move while treating the detected box as an obstacle.

This file only coordinates those stages through the saved 4x4 transform files.

Examples:
    python yanjiuyuan/real_bottle_pick_place_pipeline.py --mock --point 950,560,fg --dry-run

    python yanjiuyuan/real_bottle_pick_place_pipeline.py ^
        --capture-dir yanjiuyuan/captures/20260101-120000 ^
        --point 950,560,fg ^
        --place-pos 0.45,-0.05,0.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from yanjiuyuan.mech_eye_ur7e_pointcloud_env import DEFAULT_OUTPUT_ROOT  # noqa: E402
from yanjiuyuan import sync_real_ur7e_mech_eye_box_env as sync_scene  # noqa: E402


OBJECT_EXTRACTION_SCRIPT = Path(__file__).resolve().parent / "box_object_pointcloud_from_saved_capture_bottle_icp.py"
PICK_PLACE_SCRIPT = Path(__file__).resolve().parent / "sim_bottle_pick_place_from_box_object_icp.py"
DEFAULT_PLACE_POS = np.array([0.45, -0.05, 0.0], dtype=float)
DEFAULT_RPY_DEG = np.array([0.0, 0.0, 0.0], dtype=float)


@dataclass
class CaptureStageResult:
    capture_dir: Path
    box_transform_path: Optional[Path]
    world_ply_path: Optional[Path]
    rgb_path: Optional[Path]
    start_conf_deg: Optional[np.ndarray]


@dataclass
class ObjectStageResult:
    output_dir: Path
    summary_path: Path
    box_transform_path: Path
    bottle_transform_path: Path


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
        description="Capture/load a Mech-Eye scene, estimate box and bottle poses, then plan bottle pick-and-place."
    )

    scene_group = parser.add_argument_group("scene input")
    scene_group.add_argument("--capture-dir", type=Path, default=None, help="Use an existing capture directory.")
    scene_group.add_argument("--ply", type=Path, default=None, help="Load this PLY instead of capturing from Mech-Eye.")
    scene_group.add_argument(
        "--ply-frame",
        choices=("auto", "camera", "world"),
        default="auto",
        help="Coordinate frame of --ply.",
    )
    scene_group.add_argument("--image", type=Path, default=None, help="RGB image for object segmentation.")
    scene_group.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    scene_group.add_argument("--output-dir", type=Path, default=None, help="Output directory for a new capture/load run.")
    scene_group.add_argument("--ply-out", type=Path, default=None, help="Camera-frame PLY path when capturing.")
    scene_group.add_argument("--depth-scale", type=float, default=0.001)
    scene_group.add_argument("--depth-trunc", type=float, default=3.0)

    robot_group = parser.add_argument_group("robot state")
    robot_group.add_argument("--robot-ip", default="192.168.125.30")
    robot_group.add_argument("--gp-port", default="COM3")
    robot_group.add_argument("--mock", action="store_true", help="Use mock robot state instead of RTDE.")
    robot_group.add_argument(
        "--read-robot-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Read the current robot state and pass it as planner start config. Defaults to on for live capture.",
    )
    robot_group.add_argument("--start-conf-deg", type=parse_vec6, default=None, help="Override planner start joints.")

    object_group = parser.add_argument_group("box/object extraction")
    object_group.add_argument("--object-output-dir", type=Path, default=None)
    object_group.add_argument("--object-summary", type=Path, default=None, help="Reuse an existing extraction summary JSON.")
    object_group.add_argument("--mask", type=Path, default=None, help="Existing 2D bottle mask.")
    object_group.add_argument("--point", action="append", default=[], metavar="X,Y,LABEL", help="Point prompt for bottle segmentation.")
    object_group.add_argument("--segment-box", default=None, metavar="X1,Y1,X2,Y2")
    object_group.add_argument("--auto-segment-box", action=argparse.BooleanOptionalAction, default=False)
    object_group.add_argument("--backend", choices=("fastsam", "sam"), default="sam")
    object_group.add_argument("--model", default=None)
    object_group.add_argument("--keep", choices=("best", "all", "largest", "smallest", "combined"), default="best")
    object_group.add_argument("--imgsz", type=int, default=1024)
    object_group.add_argument("--conf", type=float, default=0.25)
    object_group.add_argument("--iou", type=float, default=0.9)
    object_group.add_argument("--device", default=None)
    object_group.add_argument("--no-gui", action="store_true")
    object_group.add_argument("--show-object-viewer", action="store_true", help="Open the Open3D object/ICP viewer.")
    object_group.add_argument("--show-box-model", action="store_true")
    object_group.add_argument(
        "--bottle-template",
        choices=("prompt", "surface", "top", "front", "left", "right", "custom"),
        default="surface",
    )
    object_group.add_argument("--bottle-template-ply", type=Path, default=None)
    object_group.add_argument("--bottle-template-prompt-gui", action=argparse.BooleanOptionalAction, default=True)
    object_group.add_argument("--bottle-stl", type=Path, default=None)
    object_group.add_argument("--bottle-voxel", type=float, default=None)
    object_group.add_argument("--bottle-icp-max-iteration", type=int, default=None)

    plan_group = parser.add_argument_group("pick-and-place planning")
    plan_group.add_argument("--skip-plan", action="store_true", help="Stop after bottle pose estimation.")
    plan_group.add_argument("--planner", choices=("wrs", "manual"), default="wrs")
    plan_group.add_argument("--object-model", type=Path, default=None, help="Planner object mesh. Defaults to bottle STL.")
    plan_group.add_argument("--place-pos", type=parse_vec3, default=DEFAULT_PLACE_POS)
    plan_group.add_argument("--place-rpy-deg", type=parse_vec3, default=DEFAULT_RPY_DEG)
    plan_group.add_argument("--grasp-json", type=Path, default=None)
    plan_group.add_argument("--approach-distance", type=float, default=None)
    plan_group.add_argument("--open-jaw", type=float, default=None)
    plan_group.add_argument("--closed-jaw", type=float, default=None)
    plan_group.add_argument(
        "--ik-solver",
        choices=("auto", "ikfast", "fast", "n", "numeric", "none"),
        default="auto",
    )
    plan_group.add_argument("--rtde-plan-out", type=Path, default=None)
    plan_group.add_argument("--dry-run", action="store_true")
    plan_group.add_argument("--manual", action="store_true")
    plan_group.add_argument("--fps", type=float, default=24.0)
    plan_group.add_argument("--return-home", action="store_true")
    plan_group.add_argument("--no-rrt", dest="use_rrt", action="store_false")
    plan_group.set_defaults(use_rrt=True)
    plan_group.add_argument("--no-collision-check", dest="collision_check", action="store_false")
    plan_group.set_defaults(collision_check=True)
    plan_group.add_argument("--allow-collision", action="store_true")
    plan_group.add_argument("--no-env", action="store_true")
    plan_group.add_argument(
        "--no-box-obstacle",
        dest="box_obstacle",
        action="store_false",
        help="Do not add the detected box as a planner obstacle.",
    )
    plan_group.set_defaults(box_obstacle=True)
    plan_group.add_argument("--show-obstacle-collision", action="store_true")
    plan_group.add_argument("--show-robot-collision", action="store_true")
    plan_group.add_argument("--summary-out", type=Path, default=None, help="Integrated pipeline summary JSON.")

    return parser.parse_args()


def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def normalize_paths(args: argparse.Namespace) -> None:
    for attr in (
        "capture_dir",
        "ply",
        "image",
        "output_root",
        "output_dir",
        "ply_out",
        "object_output_dir",
        "object_summary",
        "mask",
        "bottle_template_ply",
        "bottle_stl",
        "object_model",
        "grasp_json",
        "rtde_plan_out",
        "summary_out",
    ):
        setattr(args, attr, resolve_path(getattr(args, attr)))


def vec_to_cli(vector: np.ndarray) -> str:
    return ",".join(f"{float(value):.9g}" for value in np.asarray(vector, dtype=float))


def should_read_robot_start(args: argparse.Namespace) -> bool:
    if args.read_robot_start is not None:
        return bool(args.read_robot_start)
    return args.capture_dir is None and args.ply is None


def make_sync_args(args: argparse.Namespace, output_dir: Optional[Path]) -> argparse.Namespace:
    return argparse.Namespace(
        robot_ip=args.robot_ip,
        gp_port=args.gp_port,
        mock=args.mock,
        ply=args.ply,
        ply_frame=args.ply_frame,
        output_root=args.output_root,
        output_dir=output_dir,
        ply_out=args.ply_out,
        depth_scale=args.depth_scale,
        depth_trunc=args.depth_trunc,
        detect_box=True,
        box_transform_out=None,
        summary_out=None,
    )


def run_capture_stage(args: argparse.Namespace) -> CaptureStageResult:
    if args.capture_dir is not None:
        start_conf_deg = read_robot_start_conf(args) if should_read_robot_start(args) else None
        return CaptureStageResult(
            capture_dir=args.capture_dir,
            box_transform_path=None,
            world_ply_path=args.capture_dir / "world_colored_pointcloud.ply",
            rgb_path=args.image if args.image is not None else args.capture_dir / "rgb.png",
            start_conf_deg=start_conf_deg,
        )

    output_dir = sync_scene.make_output_dir(args.output_root, args.output_dir)
    sync_args = make_sync_args(args, output_dir)
    provider = sync_scene.make_robot_provider(sync_args) if should_read_robot_start(args) else None
    initial_jnt_values = None
    current_jnt_values = None
    current_tcp_pos = None
    current_jaw_width = None

    try:
        if provider is not None:
            initial_jnt_values, _initial_tcp_pos, _initial_jaw_width = sync_scene.read_robot_snapshot(
                provider, "Initial robot"
            )
        scene_data = sync_scene.load_or_capture_camera_scene(sync_args, output_dir)
        box_detection = sync_scene.detect_box(sync_args, scene_data)
        if provider is not None:
            current_jnt_values, current_tcp_pos, current_jaw_width = sync_scene.read_robot_snapshot(
                provider, "Scene robot"
            )
            sync_scene.write_summary(
                sync_args,
                scene_data,
                current_jnt_values,
                current_tcp_pos,
                current_jaw_width,
                box_detection,
            )
    finally:
        if provider is not None:
            provider.close()

    start_conf = current_jnt_values if current_jnt_values is not None else initial_jnt_values
    start_conf_deg = None if start_conf is None else np.degrees(start_conf)
    return CaptureStageResult(
        capture_dir=output_dir,
        box_transform_path=output_dir / "detected_box_transform.txt",
        world_ply_path=scene_data.world_ply_path,
        rgb_path=scene_data.rgb_path if scene_data.rgb_path is not None else args.image,
        start_conf_deg=start_conf_deg,
    )


def read_robot_start_conf(args: argparse.Namespace) -> Optional[np.ndarray]:
    sync_args = make_sync_args(args, output_dir=None)
    provider = sync_scene.make_robot_provider(sync_args)
    try:
        jnt_values, _tcp_pos, _jaw_width = sync_scene.read_robot_snapshot(provider, "Planner start robot")
    finally:
        provider.close()
    return np.degrees(jnt_values)


def append_optional_path(cmd: list[str], flag: str, path: Optional[Path]) -> None:
    if path is not None:
        cmd.extend([flag, str(path)])


def append_optional_value(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def print_command(label: str, cmd: list[str]) -> None:
    print(f"[pipeline] {label}:")
    print(f"  {subprocess.list2cmdline(cmd)}")


def run_subprocess(label: str, cmd: list[str]) -> None:
    print_command(label, cmd)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def build_object_stage_command(
    args: argparse.Namespace,
    capture: CaptureStageResult,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(OBJECT_EXTRACTION_SCRIPT),
        "--capture-dir",
        str(capture.capture_dir),
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
    ]
    if capture.box_transform_path is not None and capture.box_transform_path.exists():
        cmd.extend(["--box-transform", str(capture.box_transform_path)])
    if capture.world_ply_path is not None and capture.world_ply_path.exists():
        cmd.extend(["--ply", str(capture.world_ply_path)])
    image_path = args.image if args.image is not None else capture.rgb_path
    append_optional_path(cmd, "--image", image_path)
    append_optional_path(cmd, "--mask", args.mask)
    append_optional_value(cmd, "--segment-box", args.segment_box)
    append_optional_value(cmd, "--model", args.model)
    append_optional_value(cmd, "--device", args.device)
    append_optional_path(cmd, "--bottle-template-ply", args.bottle_template_ply)
    append_optional_path(cmd, "--bottle-stl", args.bottle_stl)
    append_optional_value(cmd, "--bottle-voxel", args.bottle_voxel)
    append_optional_value(cmd, "--bottle-icp-max-iteration", args.bottle_icp_max_iteration)
    for point in args.point:
        cmd.extend(["--point", point])
    if args.auto_segment_box:
        cmd.append("--auto-segment-box")
    else:
        cmd.append("--no-auto-segment-box")
    if args.no_gui:
        cmd.append("--no-gui")
    cmd.append("--show-viewer" if args.show_object_viewer else "--no-show-viewer")
    cmd.append("--show-box-model" if args.show_box_model else "--no-show-box-model")
    if not args.bottle_template_prompt_gui:
        cmd.append("--no-bottle-template-prompt-gui")
    return cmd


def load_object_stage_result(summary_path: Path, output_dir: Path) -> ObjectStageResult:
    if not summary_path.exists():
        raise FileNotFoundError(f"Object extraction summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    box_transform_path = Path(summary["box_transform_used"])
    bottle_summary = summary.get("bottle_icp")
    if not bottle_summary:
        raise RuntimeError("Object extraction summary does not contain bottle_icp results.")
    bottle_transform_path = Path(bottle_summary["icp_transform_path"])
    if not box_transform_path.exists():
        raise FileNotFoundError(f"Box transform not found: {box_transform_path}")
    if not bottle_transform_path.exists():
        raise FileNotFoundError(f"Bottle transform not found: {bottle_transform_path}")
    return ObjectStageResult(
        output_dir=output_dir,
        summary_path=summary_path,
        box_transform_path=box_transform_path,
        bottle_transform_path=bottle_transform_path,
    )


def run_object_stage(args: argparse.Namespace, capture: CaptureStageResult) -> ObjectStageResult:
    if args.object_summary is not None:
        output_dir = args.object_summary.parent
        return load_object_stage_result(args.object_summary, output_dir)

    output_dir = args.object_output_dir if args.object_output_dir is not None else capture.capture_dir / "box_object_extraction"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_object_stage_command(args, capture, output_dir)
    run_subprocess("segment bottle and run bottle ICP", cmd)
    return load_object_stage_result(output_dir / "box_object_extraction_summary.json", output_dir)


def build_plan_command(
    args: argparse.Namespace,
    capture: CaptureStageResult,
    object_result: ObjectStageResult,
) -> list[str]:
    rtde_plan_out = args.rtde_plan_out
    if rtde_plan_out is None:
        rtde_plan_out = object_result.output_dir / "pick_place_rtde_plan.json"

    cmd = [
        sys.executable,
        str(PICK_PLACE_SCRIPT),
        "--object-summary",
        str(object_result.summary_path),
        "--place-pos",
        vec_to_cli(args.place_pos),
        "--place-rpy-deg",
        vec_to_cli(args.place_rpy_deg),
        "--rtde-plan-out",
        str(rtde_plan_out),
    ]
    planner_object_model = args.object_model if args.object_model is not None else args.bottle_stl
    append_optional_path(cmd, "--object-model", planner_object_model)
    append_optional_value(cmd, "--approach-distance", args.approach_distance)
    append_optional_value(cmd, "--open-jaw", args.open_jaw)

    start_conf_deg = args.start_conf_deg
    if start_conf_deg is None:
        start_conf_deg = capture.start_conf_deg
    if start_conf_deg is not None:
        cmd.extend(["--start-conf-deg", vec_to_cli(start_conf_deg)])

    cmd.append("--box-obstacle" if args.box_obstacle else "--no-box-obstacle")
    if args.dry_run:
        cmd.append("--dry-run")
    if not args.use_rrt:
        cmd.append("--no-rrt")
    if not args.collision_check:
        cmd.append("--no-collision-check")
    if args.no_env:
        cmd.append("--no-env")
    return cmd


def run_plan_stage(
    args: argparse.Namespace,
    capture: CaptureStageResult,
    object_result: ObjectStageResult,
) -> Optional[Path]:
    if args.skip_plan:
        return None
    cmd = build_plan_command(args, capture, object_result)
    run_subprocess("plan bottle pick-and-place", cmd)
    if args.rtde_plan_out is not None:
        return args.rtde_plan_out
    return object_result.output_dir / "pick_place_rtde_plan.json"


def write_pipeline_summary(
    args: argparse.Namespace,
    capture: CaptureStageResult,
    object_result: ObjectStageResult,
    rtde_plan_path: Optional[Path],
) -> Path:
    summary_path = args.summary_out
    if summary_path is None:
        summary_path = object_result.output_dir / "real_bottle_pick_place_pipeline_summary.json"
    summary = {
        "capture_dir": str(capture.capture_dir),
        "capture_box_transform_path": None if capture.box_transform_path is None else str(capture.box_transform_path),
        "capture_world_ply_path": None if capture.world_ply_path is None else str(capture.world_ply_path),
        "capture_rgb_path": None if capture.rgb_path is None else str(capture.rgb_path),
        "object_output_dir": str(object_result.output_dir),
        "object_summary_path": str(object_result.summary_path),
        "box_obstacle_transform_path": str(object_result.box_transform_path),
        "bottle_object_transform_path": str(object_result.bottle_transform_path),
        "rtde_plan_path": None if rtde_plan_path is None else str(rtde_plan_path),
        "place_pos": np.asarray(args.place_pos, dtype=float).tolist(),
        "place_rpy_deg": np.asarray(args.place_rpy_deg, dtype=float).tolist(),
        "box_obstacle": bool(args.box_obstacle),
        "planner": args.planner,
        "planned": rtde_plan_path is not None,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    normalize_paths(args)

    capture = run_capture_stage(args)
    object_result = run_object_stage(args, capture)
    rtde_plan_path = run_plan_stage(args, capture, object_result)
    summary_path = write_pipeline_summary(args, capture, object_result, rtde_plan_path)

    print("[pipeline] done")
    print(f"[pipeline] bottle transform: {object_result.bottle_transform_path}")
    print(f"[pipeline] box obstacle transform: {object_result.box_transform_path}")
    if rtde_plan_path is not None:
        print(f"[pipeline] RTDE plan: {rtde_plan_path}")
    print(f"[pipeline] summary: {summary_path}")


if __name__ == "__main__":
    main()
