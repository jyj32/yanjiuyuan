"""
Extract object point clouds inside the detected blue box/bin from a saved Mech-Eye capture.

Steps:
  1. Reuse a detected box transform, or run box_icp_from_saved_capture.py logic to get it.
  2. Crop points in the box local frame, remove inner-wall/top-box points, and keep object points that sit above the box rim.
  3. Use point_hint_segment.py to pick one object on rgb.png.
  4. Save point clouds: selected object is red; points removed by the box crop/wall filter are gray.

Examples:
    python yanjiuyuan/box_object_pointcloud_from_saved_capture.py
    python yanjiuyuan/box_object_pointcloud_from_saved_capture.py --point 950,560,fg --no-gui
    python yanjiuyuan/box_object_pointcloud_from_saved_capture.py --mask yanjiuyuan/runs/point_hint_segment/.../mask.png
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yanjiuyuan import box_icp_from_saved_capture as box_icp  # noqa: E402
from yanjiuyuan import bottle_icp_from_saved_capture as bottle_icp  # noqa: E402
from yanjiuyuan import bottle_icp_pose_estimation as bottle_pose  # noqa: E402
from yanjiuyuan import sam_completion_template_matching as completion_matching  # noqa: E402
from yanjiuyuan.constants import (  # noqa: E402
    BOX_CAPTURE_ROOT,
    BOX_MODEL_PATH,
    BOX_OBJECT_AUTO_SEGMENT_BOX,
    BOX_OBJECT_CONCAVE_X_RANGE,
    BOX_OBJECT_CONCAVE_Y_RANGE,
    BOX_OBJECT_CONCAVE_Z_RANGE,
    BOX_OBJECT_CONCAVE_REGION_RGB,
    BOX_OBJECT_ABOVE_TOP_MARGIN,
    BOX_OBJECT_TOP_OVERHANG_XY_MARGIN,
    BOX_OBJECT_CANDIDATE_VOXEL_DOWNSAMPLE,
    BOX_OBJECT_INNER_XY_MARGIN,
    BOX_OBJECT_OUTPUT_DIR,
    BOX_OBJECT_PIXEL_COLOR_TOLERANCE,
    BOX_OBJECT_PIXEL_MAPPING_MIN_RATIO,
    BOX_OBJECT_POINT_SIZE,
    BOX_OBJECT_REMOVE_BLUE_BOX_POINTS,
    BOX_OBJECT_REMOVED_CONTEXT_XY_MARGIN,
    BOX_OBJECT_REMOVED_CONTEXT_Z_MARGIN,
    BOX_OBJECT_SHOW_REMOVED_CONTEXT,
    BOX_OBJECT_REMOVED_VOXEL_DOWNSAMPLE,
    BOX_OBJECT_SELECTED_VOXEL_DOWNSAMPLE,
    BOX_OBJECT_SHOW_BOX_MODEL,
    BOX_OBJECT_SHOW_VIEWER,
)
BOTTLE_TEMPLATE_PATHS = bottle_pose.BOTTLE_TEMPLATE_PATHS
BOTTLE_PROMPT_TEMPLATE_CHOICES = bottle_pose.BOTTLE_PROMPT_TEMPLATE_CHOICES
BOTTLE_TEMPLATE_CHOICES = bottle_pose.BOTTLE_TEMPLATE_CHOICES
GLOBAL_REGISTERED_POINTS_RGB = bottle_pose.GLOBAL_REGISTERED_POINTS_RGB
BOTTLE_GLOBAL_RANSAC_ATTEMPTS = bottle_pose.BOTTLE_GLOBAL_RANSAC_ATTEMPTS
NETWORK_INPUT_POINTS_RGB = (1.0, 0.0, 0.0)
COMPLETED_POINTS_RGB = (0.0, 0.85, 0.15)
COMPLETION_GLOBAL_REGISTERED_POINTS_RGB = (0.62, 0.14, 1.0)
COMPLETION_REGISTERED_POINTS_RGB = (0.95, 0.35, 0.05)

from yanjiuyuan.mech_eye_ur7e_pointcloud_env import (  # noqa: E402
    CAM_TO_WORLD,
    load_ply_pointcloud,
    numpy_to_open3d_pointcloud,
    transform_points,
)


@dataclass
class PipelineContext:
    args: argparse.Namespace
    capture: object
    output_dir: Path
    box_transform: np.ndarray
    box_transform_path: Path
    detected_box_now: bool
    bounds: object
    box_region_mask: np.ndarray
    candidate_mask: np.ndarray
    removed_mask: np.ndarray
    image_rgb: np.ndarray
    pixel_indices: np.ndarray
    mapped_points: np.ndarray
    auto_box: Optional[tuple[int, int, int, int]]


@dataclass
class CaptureData:
    pcd_world: Optional[object]
    points_world: np.ndarray
    colors: Optional[np.ndarray]
    target_ply: Optional[Path]
    camera_ply: Optional[Path]
    rgb_path: Optional[Path]
    capture_dir: Path
    frame: str
    pixel_indices: Optional[np.ndarray] = None
    rgb_image_bgr: Optional[np.ndarray] = None


@dataclass
class BoxLocalBounds:
    xy_min: np.ndarray
    xy_max: np.ndarray
    z_min: float
    z_max: float


@dataclass
class ExtractionMasks:
    box_region: np.ndarray
    candidate: np.ndarray
    removed: np.ndarray
    selected: np.ndarray
    mask_projected: np.ndarray
    mapped_points: np.ndarray




BOTTLE_ICP_CONFIG_FIELDS = {
    "bottle_icp": "enabled",
    "bottle_template": "template",
    "bottle_template_ply": "template_ply",
    "bottle_template_prompt_gui": "template_prompt_gui",
    "bottle_template_preview_size": "template_preview_size",
    "bottle_stl": "stl",
    "bottle_voxel": "voxel",
    "bottle_template_voxel": "template_voxel",
    "bottle_global_ransac_n": "global_ransac_n",
    "bottle_global_ransac_attempts": "global_ransac_attempts",
    "bottle_model_sample_count": "model_sample_count",
    "bottle_model_even_radius": "model_even_radius",
    "bottle_icp_max_iteration": "icp_max_iteration",
}
BOTTLE_ICP_ARG_NAMES = tuple(BOTTLE_ICP_CONFIG_FIELDS)
BOTTLE_ICP_CLI_FLAGS = {
    "bottle_icp": "--bottle-icp",
    "bottle_template": "--bottle-template",
    "bottle_template_ply": "--bottle-template-ply",
    "bottle_template_prompt_gui": "--bottle-template-prompt-gui",
    "bottle_template_preview_size": "--bottle-template-preview-size",
    "bottle_stl": "--bottle-stl",
    "bottle_voxel": "--bottle-voxel",
    "bottle_template_voxel": "--bottle-template-voxel",
    "bottle_global_ransac_n": "--bottle-global-ransac-n",
    "bottle_global_ransac_attempts": "--bottle-global-ransac-attempts",
    "bottle_model_sample_count": "--bottle-model-sample-count",
    "bottle_model_even_radius": "--bottle-model-even-radius",
    "bottle_icp_max_iteration": "--bottle-icp-max-iteration",
}


def default_bottle_icp_values() -> dict[str, object]:
    # 返回ICP配置的默认值字典，用于填充解析器参数。

    config = bottle_pose.default_bottle_icp_config()
    return {
        option_name: getattr(config, config_field)
        for option_name, config_field in BOTTLE_ICP_CONFIG_FIELDS.items()
    }


def bottle_icp_config_from_runtime_options(source: object, overrides: Optional[dict[str, object]] = None) -> bottle_pose.BottleIcpPoseConfig:
    config = bottle_pose.default_bottle_icp_config()
    for option_name, config_field in BOTTLE_ICP_CONFIG_FIELDS.items():
        if hasattr(source, option_name):
            value = getattr(source, option_name)
            if value is not None:
                setattr(config, config_field, value)
    if overrides:
        for name, value in overrides.items():
            if value is None:
                continue
            config_field = BOTTLE_ICP_CONFIG_FIELDS.get(name, name)
            if not hasattr(config, config_field):
                raise KeyError(f"Unknown bottle ICP option: {name}")
            setattr(config, config_field, value)
    return config


def add_bottle_icp_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = default_bottle_icp_values()
    parser.add_argument("--bottle-icp", action=argparse.BooleanOptionalAction, default=defaults["bottle_icp"], help="Register bottle.stl to the selected object points and draw the result.")
    parser.add_argument("--bottle-template", choices=BOTTLE_TEMPLATE_CHOICES, default=defaults["bottle_template"], help="Bottle template point cloud for registration. Use prompt to choose after segmentation.")
    parser.add_argument("--bottle-template-ply", type=Path, default=defaults["bottle_template_ply"], help="Custom bottle template PLY. Required when --bottle-template custom.")
    parser.add_argument("--bottle-template-prompt-gui", action=argparse.BooleanOptionalAction, default=defaults["bottle_template_prompt_gui"], help="Show a clickable template preview prompt when --bottle-template prompt.")
    parser.add_argument("--bottle-template-preview-size", type=int, default=defaults["bottle_template_preview_size"], help="Pixel size of each template preview tile.")
    parser.add_argument("--bottle-stl", type=Path, default=defaults["bottle_stl"])
    parser.add_argument("--bottle-voxel", type=float, default=defaults["bottle_voxel"])
    parser.add_argument("--bottle-template-voxel", type=float, default=defaults["bottle_template_voxel"], help="Voxel size used to downsample the bottle template before registration; <=0 disables.")
    parser.add_argument("--bottle-global-ransac-n", type=int, default=defaults["bottle_global_ransac_n"], help="RANSAC sample size for global feature registration.")
    parser.add_argument("--bottle-global-ransac-attempts", type=int, default=defaults["bottle_global_ransac_attempts"], help="Number of global RANSAC attempts; the best sane result is used.")
    parser.add_argument("--bottle-model-sample-count", type=int, default=defaults["bottle_model_sample_count"])
    parser.add_argument("--bottle-model-even-radius", type=float, default=defaults["bottle_model_even_radius"])
    parser.add_argument("--bottle-icp-max-iteration", type=int, default=defaults["bottle_icp_max_iteration"])


def append_bottle_icp_cli_args(cmd: list[str], source: object) -> None:
    for option_name in BOTTLE_ICP_ARG_NAMES:
        if not hasattr(source, option_name):
            continue
        value = getattr(source, option_name)
        if value is None:
            continue
        flag = BOTTLE_ICP_CLI_FLAGS[option_name]
        if isinstance(value, bool):
            cmd.append(flag if value else f"--no-{flag[2:]}")
        else:
            cmd.extend([flag, str(value)])


def add_completion_matching_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--completion-matching", action=argparse.BooleanOptionalAction, default=True, help="After SAM point selection, complete the selected point cloud with AdaPoinTr, then run surface template ICP.")
    parser.add_argument("--completion-template", choices=("surface", "custom"), default="surface", help="Template used after completion. surface uses yanjiuyuan/models/bottle_surface_points.ply.")
    parser.add_argument("--completion-template-ply", type=Path, default=None, help="Custom complete template PLY when --completion-template custom.")
    parser.add_argument("--completion-adapointr-script", type=Path, default=completion_matching.DEFAULT_ADAPOINTR_SCRIPT, help="Path to infer_AdaPoinTr.py.")
    parser.add_argument("--completion-adapointr-checkpoint", type=Path, default=completion_matching.DEFAULT_ADAPOINTR_CHECKPOINT, help="AdaPoinTr checkpoint.")
    parser.add_argument("--completion-output-prefix", default="sam_completion_surface")
    parser.add_argument("--completion-device", default="cuda:0")
    parser.add_argument("--completion-global-scale", type=float, default=0.4)
    parser.add_argument("--completion-num-points", type=int, default=1024)
    parser.add_argument("--completion-num-query", type=int, default=128)
    parser.add_argument("--completion-voxel-size", type=float, default=0.005)
    parser.add_argument("--completion-template-voxel-size", type=float, default=0.005)
    parser.add_argument("--completion-ransac-n", type=int, default=3)
    parser.add_argument("--completion-ransac-attempts", type=int, default=5)
    parser.add_argument("--completion-icp-max-iteration", type=int, default=80)
    parser.add_argument("--completion-network-input-points", type=int, default=2048)
    parser.add_argument("--completion-selected-outlier-nb-neighbors", type=int, default=24)
    parser.add_argument("--completion-selected-outlier-std-ratio", type=float, default=1.8)
    parser.add_argument("--completion-selected-outlier-min-keep-ratio", type=float, default=0.65)


def add_completion_bottle_icp_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--completion-bottle-icp", action=argparse.BooleanOptionalAction, default=True, help="After AdaPoinTr completion, use selected+completed for global registration and selected-only for ICP.")
    parser.add_argument("--completion-bottle-template", choices=("surface", "top", "front", "left", "right", "custom"), default="surface", help="Bottle template for selected+completed pose estimation. Defaults to the full surface template.")
    parser.add_argument("--completion-bottle-template-ply", type=Path, default=None, help="Custom template PLY when --completion-bottle-template custom.")
    parser.add_argument("--completion-bottle-target-voxel-size", type=float, default=0.003, help="Voxel size for downsampling selected/completed target points before bottle ICP; <=0 disables.")
    parser.add_argument("--completion-bottle-template-voxel-size", type=float, default=0.003, help="Voxel size for bottle template downsampling before ICP; keep small to preserve geometry; <=0 disables.")


def completion_template_path_from_runtime_options(source: object) -> Path:
    template_name = getattr(source, "completion_template", "surface")
    if template_name == "surface":
        return completion_matching.DEFAULT_FULL_TEMPLATE_PLY
    if template_name == "custom":
        custom_path = getattr(source, "completion_template_ply", None)
        if custom_path is None:
            raise ValueError("--completion-template custom requires --completion-template-ply.")
        return Path(custom_path)
    raise ValueError(f"Unknown completion template: {template_name}")


def completion_matching_config_from_runtime_options(source: object, output_dir: Path) -> completion_matching.CompletionMatchingConfig:
    return completion_matching.CompletionMatchingConfig(
        adapointr_script=resolve_path(Path(source.completion_adapointr_script)),
        adapointr_checkpoint=resolve_path(Path(source.completion_adapointr_checkpoint)),
        full_template_ply=resolve_path(completion_template_path_from_runtime_options(source)),
        output_dir=resolve_path(output_dir) / "sam_completion_template_matching",
        cam_to_world=completion_matching.normalize_homomat(CAM_TO_WORLD, "CAM_TO_WORLD"),
        device=str(source.completion_device),
        global_scale=float(source.completion_global_scale),
        num_points=int(source.completion_num_points),
        num_query=int(source.completion_num_query),
        voxel_size=float(source.completion_voxel_size),
        template_voxel_size=float(source.completion_template_voxel_size),
        ransac_n=int(source.completion_ransac_n),
        ransac_attempts=int(source.completion_ransac_attempts),
        icp_max_iteration=int(source.completion_icp_max_iteration),
        network_input_points=int(source.completion_network_input_points),
        selected_outlier_nb_neighbors=int(source.completion_selected_outlier_nb_neighbors),
        selected_outlier_std_ratio=float(source.completion_selected_outlier_std_ratio),
        selected_outlier_min_keep_ratio=float(source.completion_selected_outlier_min_keep_ratio),
    )


def completion_bottle_icp_config_from_runtime_options(source: object) -> bottle_pose.BottleIcpPoseConfig:
    template_name = getattr(source, "completion_bottle_template", "surface")
    template_ply = getattr(source, "completion_bottle_template_ply", None)
    if template_name == "custom" and template_ply is None:
        template_ply = getattr(source, "bottle_template_ply", None)
    config = bottle_icp_config_from_runtime_options(
        source,
        overrides={
            "enabled": True,
            "template": template_name,
            "template_ply": template_ply,
            "template_voxel": getattr(source, "completion_bottle_template_voxel_size", None),
        },
    )
    if config.template == "custom" and config.template_ply is None:
        raise ValueError("--completion-bottle-template custom requires --completion-bottle-template-ply.")
    return config


def log(message: str) -> None:
    print(message, flush=True)


@contextmanager
def timed_step(name: str):
    start_time = perf_counter()
    log(f"[box_object] START {name}")
    try:
        yield
    except Exception:
        log(f"[box_object] FAILED {name} after {perf_counter() - start_time:.2f}s")
        raise
    log(f"[box_object] DONE {name} in {perf_counter() - start_time:.2f}s")


# 使用 argparse 解析命令行参数，涵盖捕获目录、点云文件、提示点、掩膜、盒体变换、裁剪参数、瓶体 ICP、AdaPoinTr 补全等所有可配置项。
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Extract and select object point clouds inside the detected box.")
    parser.add_argument("--capture-root", type=Path, default=BOX_CAPTURE_ROOT)
    parser.add_argument("--capture-dir", type=Path, default=None, help="Capture folder. Defaults to newest capture.")
    parser.add_argument("--ply", type=Path, default=None, help="Specific colored PLY. Defaults to capture PLY.")
    parser.add_argument("--prefer-world-ply", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image", type=Path, default=None, help="RGB image used by point_hint_segment.py.")
    parser.add_argument("--box-transform", type=Path, default=None, help="Optional 4x4 detected box transform file.")
    parser.add_argument("--output-dir", type=Path, default=BOX_OBJECT_OUTPUT_DIR)
    parser.add_argument("--mask", type=Path, default=None, help="Existing 2D object mask. If omitted, point_hint_segment.py is used.")
    parser.add_argument("--point", action="append", default=[], metavar="X,Y,LABEL", help="Point prompt for point_hint_segment.py.")
    parser.add_argument("--segment-box", default=None, metavar="X1,Y1,X2,Y2", help="Optional 2D box prompt.")
    parser.add_argument("--auto-segment-box", action=argparse.BooleanOptionalAction, default=BOX_OBJECT_AUTO_SEGMENT_BOX, help="Pass an automatically projected candidate bbox to point_hint_segment.py. Disabled by default because it can bias SAM toward the whole box.")
    parser.add_argument("--backend", choices=("fastsam", "sam"), default="sam")
    parser.add_argument("--model", default=None)
    parser.add_argument("--keep", choices=("best", "all", "largest", "smallest", "combined"), default="best")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-display", type=int, default=1400)
    parser.add_argument("--no-gui", action="store_true", help="Do not open the interactive point prompt window.")
    parser.add_argument(
        "--show-gui-with-points",
        action="store_true",
        help="When --point is provided, show those points in the interactive point prompt window.",
    )
    parser.add_argument("--inner-xy-margin", type=float, default=BOX_OBJECT_INNER_XY_MARGIN)
    parser.add_argument("--top-overhang-xy-margin", type=float, default=BOX_OBJECT_TOP_OVERHANG_XY_MARGIN, help="Extra local XY margin around the concave region for objects resting on the box top edge.")
    parser.add_argument("--above-top-margin", type=float, default=BOX_OBJECT_ABOVE_TOP_MARGIN, help="How far above concave z_max to preserve top-overhang points.")
    parser.add_argument("--remove-blue-box-points", action=argparse.BooleanOptionalAction, default=BOX_OBJECT_REMOVE_BLUE_BOX_POINTS)
    parser.add_argument("--show-removed-context", action=argparse.BooleanOptionalAction, default=BOX_OBJECT_SHOW_REMOVED_CONTEXT, help="Show nearby points rejected by the box-local crop as gray context.")
    parser.add_argument("--removed-context-xy-margin", type=float, default=BOX_OBJECT_REMOVED_CONTEXT_XY_MARGIN, help="Extra local XY margin for gray rejected context around the concave region.")
    parser.add_argument("--removed-context-z-margin", type=float, default=BOX_OBJECT_REMOVED_CONTEXT_Z_MARGIN, help="Extra local Z margin below concave z_min for gray rejected context.")
    parser.add_argument("--pixel-color-tolerance", type=int, default=BOX_OBJECT_PIXEL_COLOR_TOLERANCE)
    parser.add_argument("--pixel-mapping-min-ratio", type=float, default=BOX_OBJECT_PIXEL_MAPPING_MIN_RATIO)
    parser.add_argument("--candidate-voxel", type=float, default=BOX_OBJECT_CANDIDATE_VOXEL_DOWNSAMPLE)
    parser.add_argument("--removed-voxel", type=float, default=BOX_OBJECT_REMOVED_VOXEL_DOWNSAMPLE)
    parser.add_argument("--selected-voxel", type=float, default=BOX_OBJECT_SELECTED_VOXEL_DOWNSAMPLE)
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=BOX_OBJECT_SHOW_VIEWER)
    parser.add_argument("--show-box-model", action=argparse.BooleanOptionalAction, default=BOX_OBJECT_SHOW_BOX_MODEL, help="Show and save the registered box.STL model. Disabled by default so it does not hide inner points.")
    add_bottle_icp_arguments(parser)
    add_completion_matching_arguments(parser)
    add_completion_bottle_icp_arguments(parser)
    parser.set_defaults(bottle_icp=False)
    parser.add_argument("--point-size", type=float, default=BOX_OBJECT_POINT_SIZE)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    # 把相对路径转为绝对路径
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def transform_open3d_pointcloud(pcd, homomat: np.ndarray):
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(homomat)
    return pcd_copy


def open3d_to_numpy_keep_order(pcd) -> tuple[np.ndarray, Optional[np.ndarray]]:
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float64)
        if len(colors) != len(points):
            colors = None
    if len(points) == 0:
        raise RuntimeError("Point cloud is empty.")
    return points, colors


def resolve_capture(args: argparse.Namespace) -> CaptureData:
    # 解析命令行指定的捕获目录或 PLY 文件，加载点云（世界坐标系），确定帧类型（camera/world），返回 CaptureData 对象。
    capture_dir_arg = args.capture_dir
    if args.ply is None and capture_dir_arg is None:
        capture_dir_arg = box_icp.find_latest_capture_dir(args.capture_root)
    ply_path, capture_dir, frame = box_icp.resolve_capture_ply(
        capture_dir=capture_dir_arg,
        capture_ply=args.ply,
        prefer_world_frame=args.prefer_world_ply,
    )
    rgb_path = resolve_path(args.image) if args.image is not None else capture_dir / "rgb.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    pcd = load_ply_pointcloud(ply_path)
    points, colors = open3d_to_numpy_keep_order(pcd)
    if frame == "camera":
        points_world = transform_points(points, CAM_TO_WORLD)
        pcd_world = transform_open3d_pointcloud(pcd, CAM_TO_WORLD)
    else:
        points_world = points.copy()
        pcd_world = pcd
    camera_ply = capture_dir / "colored_pointcloud.ply"
    if not camera_ply.exists():
        camera_ply = None
    log(f"[box_object] capture_dir: {capture_dir}")
    log(f"[box_object] target_ply: {ply_path} ({frame})")
    log(f"[box_object] rgb_path: {rgb_path}")
    log(f"[box_object] raw points: {len(points_world)}")
    return CaptureData(pcd_world, points_world, colors, ply_path, camera_ply, rgb_path, capture_dir, frame)


def default_output_dir(capture_dir: Path) -> Path:
    # 基于捕获目录生成默认输出目录
    return resolve_path(Path(BOX_OBJECT_OUTPUT_DIR)) if BOX_OBJECT_OUTPUT_DIR is not None else capture_dir / "box_object_extraction"


def candidate_transform_paths(capture_dir: Path) -> list[Path]:
    # 返回可能存放盒体变换文件的路径列表
    return [
        capture_dir / "detected_box_transform.txt",
        capture_dir / "box_icp" / "box_obb_icp_transform.txt",
        capture_dir / "box_icp" / "box_icp_transform.txt",
    ]


def load_box_transform(args: argparse.Namespace, capture: CaptureData, output_dir: Path) -> tuple[np.ndarray, Path, bool]:
    # 尝试从指定文件或候选路径加载盒体变换矩阵
    if args.box_transform is not None:
        path = resolve_path(args.box_transform)
        if not path.exists():
            raise FileNotFoundError(f"Box transform does not exist: {path}")
        return np.loadtxt(path), path, False
    for path in candidate_transform_paths(capture.capture_dir):
        if path.exists():
            return np.loadtxt(path), path, False
    with timed_step("detect box pose because no transform file was found"):
        world_pcd = capture.pcd_world
        if world_pcd is None:
            log("[box_object] building Open3D world point cloud for box-transform fallback only")
            world_pcd = numpy_to_open3d_pointcloud(capture.points_world, capture.colors)
        transform = detect_box_transform_from_pointcloud(world_pcd)
    path = output_dir / "detected_box_transform.txt"
    np.savetxt(path, transform, fmt="%.9f")
    return transform, path, True


def detect_box_transform_from_pointcloud(world_pcd) -> np.ndarray:
    # 从点云中检测蓝盒姿态：裁剪、滤波、聚类、下采样，然后通过 OBB+ICP 匹配盒体模板，返回变换矩阵。
    target_pcd = box_icp.crop_pointcloud_by_range(world_pcd)    # 裁剪，通过xyz范围
    target_pcd, _ = box_icp.segment_blue_box_points(target_pcd) # 滤波，通过颜色分割出蓝箱点云
    target_pcd = box_icp.voxel_downsample_target(target_pcd)    # 下采样
    target_pcd, _ = box_icp.keep_largest_cluster(target_pcd)    # 聚类
    target_pcd = box_icp.remove_target_outliers(target_pcd) # 滤波，去除稀疏的、不连续的噪点
    target_pcd = box_icp.filter_pointcloud_by_z_range(
        target_pcd, box_icp.MATCH_TARGET_Z_MIN, box_icp.MATCH_TARGET_Z_MAX, "Box target point cloud"
    )   # 裁剪，通过z范围
    source_pcd, _ = box_icp.load_box_template_pointcloud()  # 箱体点云，只保留顶部平面上的点
    obb_initial_transform, obb_initial_result, icp_result, *_ = box_icp.run_obb_initialized_icp(
        source_pcd,
        target_pcd,
        voxel_size=box_icp.VOXEL_SIZE,
    )
    icp_fitness = float(icp_result.fitness)
    obb_fitness = float(obb_initial_result.fitness)
    if icp_fitness < obb_fitness:
        log(
            "[box_object] Warning: box ICP reduced fitness "
            f"from {obb_fitness:.6f} to {icp_fitness:.6f}; using OBB initial transform."
        )
        return np.asarray(obb_initial_transform, dtype=np.float64)
    return np.asarray(icp_result.transformation, dtype=np.float64)


def compute_box_local_bounds() -> BoxLocalBounds:
    x_min, x_max = BOX_OBJECT_CONCAVE_X_RANGE
    y_min, y_max = BOX_OBJECT_CONCAVE_Y_RANGE
    z_min, z_max = BOX_OBJECT_CONCAVE_Z_RANGE
    bounds = BoxLocalBounds(
        xy_min=np.array([x_min, y_min], dtype=np.float64),
        xy_max=np.array([x_max, y_max], dtype=np.float64),
        z_min=float(z_min),
        z_max=float(z_max),
    )
    log(
        "[box_object] configured concave local region: "
        f"x=[{x_min:.5f}, {x_max:.5f}], "
        f"y=[{y_min:.5f}, {y_max:.5f}], "
        f"z=[{z_min:.5f}, {z_max:.5f}]"
    )
    return bounds


def transform_world_to_box_local(points_world: np.ndarray, box_transform: np.ndarray) -> np.ndarray:
    inv_transform = np.linalg.inv(box_transform)
    points_h = np.column_stack((points_world, np.ones(len(points_world))))
    return (inv_transform @ points_h.T).T[:, :3]


def colors_to_u8(colors: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if colors is None:
        return None
    return np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)


def blue_box_color_mask(colors: Optional[np.ndarray]) -> np.ndarray:
    if colors is None:
        return np.zeros(0, dtype=bool)
    return np.asarray(box_icp.blue_color_mask(np.clip(colors, 0.0, 1.0)), dtype=bool)


def compute_geometric_candidate_masks(
    points_world: np.ndarray,
    colors: Optional[np.ndarray],
    box_transform: np.ndarray,
    bounds: BoxLocalBounds,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points_local = transform_world_to_box_local(points_world, box_transform)
    x, y, z = points_local[:, 0], points_local[:, 1], points_local[:, 2]
    region_min = bounds.xy_min + args.inner_xy_margin
    region_max = bounds.xy_max - args.inner_xy_margin
    z_min = bounds.z_min
    z_max = bounds.z_max
    if np.any(region_min >= region_max):
        raise RuntimeError(
            "Concave crop is empty. Reduce --inner-xy-margin; "
            f"region_min={region_min}, region_max={region_max}."
        )

    full_concave_region = (
        (x >= bounds.xy_min[0]) & (x <= bounds.xy_max[0]) &
        (y >= bounds.xy_min[1]) & (y <= bounds.xy_max[1]) &
        (z >= z_min) & (z <= z_max)
    )
    concave_search_region = (
        (x >= region_min[0]) & (x <= region_max[0]) &
        (y >= region_min[1]) & (y <= region_max[1]) &
        (z >= z_min) & (z <= z_max)
    )
    overhang_min = bounds.xy_min - args.top_overhang_xy_margin
    overhang_max = bounds.xy_max + args.top_overhang_xy_margin
    top_overhang_z_min = z_max
    top_overhang_z_max = z_max + args.above_top_margin
    top_overhang_region = (
        (x >= overhang_min[0]) & (x <= overhang_max[0]) &
        (y >= overhang_min[1]) & (y <= overhang_max[1]) &
        (z >= top_overhang_z_min) &
        (z <= top_overhang_z_max)
    )
    search_region = concave_search_region | top_overhang_region
    box_region = full_concave_region | top_overhang_region

    blue_removed = np.zeros(len(points_world), dtype=bool)
    if args.remove_blue_box_points and colors is not None:
        blue_mask = blue_box_color_mask(colors)
        if len(blue_mask) == len(search_region):
            blue_removed = blue_mask & box_region
            log(f"[box_object] blue box-color points inside concave/overhang region: {int(blue_removed.sum())}")

    candidate = search_region & ~blue_removed
    excluded_by_margin = box_region & ~search_region & ~blue_removed
    if args.show_removed_context:
        context_min = bounds.xy_min - args.removed_context_xy_margin
        context_max = bounds.xy_max + args.removed_context_xy_margin
        context_region = (
            (x >= context_min[0]) & (x <= context_max[0]) &
            (y >= context_min[1]) & (y <= context_max[1]) &
            (z >= z_min - args.removed_context_z_margin) &
            (z <= z_max + args.above_top_margin)
        )
        context_removed = context_region & ~candidate
    else:
        context_region = box_region
        context_removed = excluded_by_margin
    removed = blue_removed | context_removed
    log(
        "[box_object] concave crop masks: "
        f"concave_region={int(full_concave_region.sum())}, "
        f"top_overhang_region={int(top_overhang_region.sum())}, "
        f"search_region={int(search_region.sum())}, "
        f"candidate/kept={int(candidate.sum())}, removed_gray={int(removed.sum())}, "
        f"blue_removed={int(blue_removed.sum())}, gray_context_region={int(context_region.sum())}, "
        f"excluded_by_margin={int(excluded_by_margin.sum())}, "
        f"concave_x=[{bounds.xy_min[0]:.4f}, {bounds.xy_max[0]:.4f}], "
        f"concave_y=[{bounds.xy_min[1]:.4f}, {bounds.xy_max[1]:.4f}], "
        f"concave_z=[{z_min:.4f}, {z_max:.4f}], "
        f"overhang_xy_margin={args.top_overhang_xy_margin:.4f}, "
        f"overhang_z=[{top_overhang_z_min:.4f}, {top_overhang_z_max:.4f}], "
        f"inner_xy_margin={args.inner_xy_margin:.4f}"
    )
    return points_local, box_region, candidate, removed


def load_rgb_image_rgb(image_path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def load_mask_image(mask_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    mask = Image.open(mask_path).convert("L")
    if mask.size != (image_shape[1], image_shape[0]):
        mask = mask.resize((image_shape[1], image_shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def build_pixel_to_point_indices_by_color_sequence(image_rgb: np.ndarray, point_colors_u8: np.ndarray, tolerance: int) -> np.ndarray:
    flat_rgb = image_rgb.reshape(-1, 3).astype(np.int16)
    point_colors = point_colors_u8.astype(np.int16)
    pixel_indices = np.full(len(point_colors), -1, dtype=np.int64)
    cursor = 0
    tol = int(max(0, tolerance))
    for point_idx, color in enumerate(point_colors):
        while cursor < len(flat_rgb):
            if np.max(np.abs(flat_rgb[cursor] - color)) <= tol:
                pixel_indices[point_idx] = cursor
                cursor += 1
                break
            cursor += 1
        if cursor >= len(flat_rgb) and point_idx < len(point_colors) - 1:
            break
    return pixel_indices


def resolve_pixel_indices_for_capture(
    capture: CaptureData,
    image_rgb: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    pixel_count = int(image_rgb.shape[0] * image_rgb.shape[1])
    if capture.pixel_indices is not None:
        pixel_indices = np.asarray(capture.pixel_indices, dtype=np.int64).reshape(-1)
        if len(pixel_indices) != len(capture.points_world):
            raise RuntimeError(
                "CaptureData.pixel_indices length does not match point cloud length: "
                f"{len(pixel_indices)} != {len(capture.points_world)}."
            )
        source = "direct captured pixel indices"
    else:
        colors_u8 = colors_to_u8(capture.colors)
        if colors_u8 is None:
            raise RuntimeError(
                "CaptureData.pixel_indices is missing and point cloud has no colors; "
                "cannot project the SAM mask back to 3D points."
            )
        pixel_indices = build_pixel_to_point_indices_by_color_sequence(
            image_rgb,
            colors_u8,
            args.pixel_color_tolerance,
        )
        source = "color sequence fallback"
    mapped_points = (pixel_indices >= 0) & (pixel_indices < pixel_count)
    mapped_ratio = float(mapped_points.sum()) / max(1, len(pixel_indices))
    log(
        f"[box_object] pixel mapping ({source}): "
        f"{int(mapped_points.sum())}/{len(pixel_indices)} ({mapped_ratio * 100.0:.1f}%)"
    )
    if mapped_ratio < args.pixel_mapping_min_ratio:
        raise RuntimeError(
            "Too few points could be mapped to RGB pixels. "
            f"source={source}, ratio={mapped_ratio:.3f}, required={args.pixel_mapping_min_ratio:.3f}."
        )
    return pixel_indices, mapped_points, mapped_ratio

def bbox_from_pixel_indices(pixel_indices: np.ndarray, mask: np.ndarray, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    valid = pixel_indices[mask]
    valid = valid[valid >= 0]
    if len(valid) == 0:
        return None
    ys = valid // width
    xs = valid % width
    pad = 8
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(width - 1, int(xs.max()) + pad),
        min(height - 1, int(ys.max()) + pad),
    )


def format_segment_box(box: Optional[tuple[int, int, int, int]]) -> Optional[str]:
    return None if box is None else ",".join(str(int(v)) for v in box)


def point_hint_result_to_mask_summary(
    phs,
    result,
    image_shape: tuple[int, int],
    logical_image_path: Path,
    segment_args: argparse.Namespace,
    points: list[tuple[int, int, int]],
    box: Optional[tuple[int, int, int, int]],
) -> tuple[np.ndarray, dict]:
    if result.masks is None:
        raw_masks: list[np.ndarray] = []
    else:
        mask_data = result.masks.data
        if hasattr(mask_data, "cpu"):
            mask_data = mask_data.cpu().numpy()
        raw_masks = [phs.ensure_mask(mask, image_shape) for mask in np.asarray(mask_data)]
    selected_masks = phs.select_masks(raw_masks, segment_args.keep, points) if raw_masks else []
    if not selected_masks:
        raise RuntimeError("point_hint_segment did not produce a usable mask.")

    prompts = [{"x": x, "y": y, "label": int(label)} for x, y, label in points]
    detections = []
    for idx, mask in enumerate(selected_masks):
        detections.append(
            {
                "index": idx,
                "area_px": int((mask > 0).sum()),
                "mask_path": None,
                "cutout_path": None,
            }
        )
    summary = {
        "image": str(logical_image_path),
        "backend": segment_args.backend,
        "model": segment_args.model,
        "keep": segment_args.keep,
        "box": None if box is None else list(box),
        "points": prompts,
        "num_raw_masks": len(raw_masks),
        "num_saved_masks": len(selected_masks),
        "outputs_saved": False,
        "overlay_path": None,
        "summary_path": None,
        "detections": detections,
    }
    return selected_masks[0] > 0, summary


def run_or_load_point_hint_mask(
    args: argparse.Namespace,
    image_path: Optional[Path],
    output_dir: Path,
    auto_box: Optional[tuple[int, int, int, int]],
    image_bgr: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[Path], Optional[dict]]:
    # SAM 2D 分割
    import cv2

    if image_bgr is not None:
        image_bgr = np.asarray(image_bgr, dtype=np.uint8)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        if image_path is None:
            raise RuntimeError("RGB image is only available in memory, but no image array was provided.")
        image_rgb = load_rgb_image_rgb(image_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read RGB image with cv2: {image_path}")
    image_shape = image_rgb.shape[:2]
    if args.mask is not None:
        mask_path = resolve_path(args.mask)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask image not found: {mask_path}")
        return load_mask_image(mask_path, image_shape), mask_path, None

    from yanjiuyuan import point_hint_segment as phs

    segment_box = args.segment_box or (format_segment_box(auto_box) if args.auto_segment_box else None)
    logical_image_path = image_path.resolve() if image_path is not None else (output_dir / "rgb_in_memory.png").resolve()
    segment_args = argparse.Namespace(
        image=logical_image_path,
        backend=args.backend,
        model=args.model or phs.default_model(args.backend),
        out_dir=(output_dir / "point_hint_segment").resolve(),
        point=list(args.point),
        box=segment_box,
        keep=args.keep,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_display=args.max_display,
        no_gui=args.no_gui,
        show_gui_with_points=bool(getattr(args, "show_gui_with_points", False)),
    )

    h, w = image_bgr.shape[:2]
    points = phs.clamp_points([phs.parse_point(point) for point in segment_args.point], w, h)
    box = phs.clamp_box(phs.parse_box(segment_args.box), w, h) if segment_args.box else None
    log(f"[box_object] point_hint backend={segment_args.backend}, model={segment_args.model}")
    if box is not None:
        log(f"[box_object] point_hint box prompt: {box}")

    model_key = (segment_args.backend, str(segment_args.model))
    model = getattr(args, "point_hint_model", None)
    if model is None or getattr(args, "point_hint_model_key", None) != model_key:
        model = phs.load_model(segment_args.backend, segment_args.model)
        setattr(args, "point_hint_model", model)
        setattr(args, "point_hint_model_key", model_key)
        log("[box_object] point_hint model loaded for this context")
    else:
        log("[box_object] reusing preloaded point_hint model")
    show_gui_with_points = bool(getattr(segment_args, "show_gui_with_points", False))
    if points:
        log("[box_object] point_hint points: " + " ".join(f"--point {phs.format_point_hint(point)}" for point in points))
    if show_gui_with_points and points and not segment_args.no_gui:
        log("[box_object] point_hint GUI will show the configured points before segmentation.")

    if segment_args.no_gui or (points and not show_gui_with_points):
        if not points:
            raise ValueError("--no-gui requires at least one --point, or use --mask.")
        result = phs.run_model(
            model,
            segment_args.backend,
            image_bgr,
            points,
            box,
            segment_args.imgsz,
            segment_args.conf,
            segment_args.iou,
            segment_args.device,
        )
        mask_image, summary = point_hint_result_to_mask_summary(
            phs,
            result,
            image_shape,
            logical_image_path,
            segment_args,
            points,
            box,
        )
        log("[box_object] point_hint mask kept in memory; no mask/cutout/overlay images written.")
        return mask_image, None, summary
    else:
        session = phs.InteractiveSession(segment_args, image_bgr, model)
        summary = session.run()

    if not summary or not summary.get("detections"):
        raise RuntimeError("point_hint_segment.py did not save any mask. Add foreground points, press 's', then press 'q'.")
    mask_path_value = summary["detections"][0].get("mask_path")
    if not mask_path_value:
        raise RuntimeError("point_hint_segment returned a summary without a saved mask path.")
    mask_path = Path(mask_path_value)
    return load_mask_image(mask_path, image_shape), mask_path, summary


def apply_2d_mask_to_points(
    mask_image: np.ndarray,
    pixel_indices: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # 2D mask 投影到 3D，选出瓶子点
    flat_mask = mask_image.reshape(-1)
    mapped_points = (pixel_indices >= 0) & (pixel_indices < len(flat_mask))
    mask_projected = np.zeros(len(pixel_indices), dtype=bool)
    mask_projected[mapped_points] = flat_mask[pixel_indices[mapped_points]]
    selected = candidate_mask & mask_projected
    log(
        "[box_object] mask selection: "
        f"mask_projected={int(mask_projected.sum())}, selected_after_box_candidate={int(selected.sum())}"
    )
    return selected, mask_projected


def voxel_downsample_arrays(points: np.ndarray, colors: Optional[np.ndarray], voxel_size: Optional[float]):
    if voxel_size is None or voxel_size <= 0 or len(points) == 0:
        return points, colors
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    pcd = pcd.voxel_down_sample(float(voxel_size))
    out_points = np.asarray(pcd.points, dtype=np.float64)
    out_colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
    return out_points, out_colors



def save_registered_bottle_template_pointclouds(
    bottle_summary: dict,
    config: bottle_pose.BottleIcpPoseConfig,
    output_dir: Path,
) -> dict:
    import open3d as o3d

    source_pcd, _template_path, _template_name = bottle_pose.load_bottle_template_pointcloud(config)
    source_pcd, _source_original_count = bottle_pose.downsample_bottle_template_pointcloud(
        source_pcd,
        config.template_voxel,
    )
    global_transform = np.asarray(bottle_summary["global_transform"], dtype=np.float64)
    icp_transform = np.asarray(bottle_summary["icp_transform"], dtype=np.float64)
    global_registered = transform_open3d_pointcloud(source_pcd, global_transform)
    icp_registered = transform_open3d_pointcloud(source_pcd, icp_transform)

    global_registered_path = output_dir / "completion_bottle_global_registered_template_world.ply"
    icp_registered_path = output_dir / "completion_bottle_icp_registered_template_world.ply"
    o3d.io.write_point_cloud(str(global_registered_path), global_registered, write_ascii=False)
    o3d.io.write_point_cloud(str(icp_registered_path), icp_registered, write_ascii=False)
    return {
        "global_registered_path": str(global_registered_path),
        "icp_registered_path": str(icp_registered_path),
        "registered_template_point_count": int(len(source_pcd.points)),
    }


def run_completion_bottle_icp_on_selected_and_completed(
    selected_points_world: np.ndarray,
    completed_points_world: np.ndarray,
    output_dir: Path,
    config: bottle_pose.BottleIcpPoseConfig,
    target_voxel_size: Optional[float],
) -> dict:
    # 把瓶子模板点云对齐到实际拍摄+补全的点云上，求出瓶子在世界坐标系下的位姿。RANSAC粗配准+icp精匹配
    # 1.准备目标点云
    selected_points_world = np.asarray(selected_points_world, dtype=np.float64).reshape(-1, 3)
    completed_points_world = np.asarray(completed_points_world, dtype=np.float64).reshape(-1, 3)
    if len(selected_points_world) == 0:
        raise RuntimeError("Selected point cloud is empty; cannot run completion bottle ICP.")
    if len(completed_points_world) == 0:
        raise RuntimeError("Completed point cloud is empty; cannot run completion bottle ICP.")

    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_downsampled, _ = voxel_downsample_arrays(selected_points_world, None, target_voxel_size)
    completed_downsampled, _ = voxel_downsample_arrays(completed_points_world, None, target_voxel_size)
    if len(selected_downsampled) == 0:
        selected_downsampled = selected_points_world
    if len(completed_downsampled) == 0:
        completed_downsampled = completed_points_world
    if len(selected_downsampled) < 20:
        raise RuntimeError(
            f"Selected target is too small for ICP after downsampling: {len(selected_downsampled)} points."
        )

    combined_before_final = np.vstack((selected_downsampled, completed_downsampled))
    combined_points, _ = voxel_downsample_arrays(combined_before_final, None, target_voxel_size)
    if len(combined_points) == 0:
        combined_points = combined_before_final
    if len(combined_points) < 20:
        raise RuntimeError(
            f"Selected+completed target is too small for global registration after downsampling: {len(combined_points)} points."
        )

    selected_target_path = output_dir / "completion_bottle_selected_target_downsampled_world.ply"
    completed_target_path = output_dir / "completion_bottle_completed_target_downsampled_world.ply"
    combined_target_path = output_dir / "completion_bottle_global_selected_plus_completed_target_world.ply"
    completion_matching.write_pointcloud(selected_downsampled, selected_target_path)
    completion_matching.write_pointcloud(completed_downsampled, completed_target_path)
    completion_matching.write_pointcloud(combined_points, combined_target_path)

    log(
        "[box_object] completion bottle registration targets: "
        f"selected={len(selected_points_world)}->{len(selected_downsampled)}, "
        f"completed={len(completed_points_world)}->{len(completed_downsampled)}, "
        f"global_combined={len(combined_before_final)}->{len(combined_points)}, "
        "icp_selected_only=True, "
        f"target_voxel={target_voxel_size}, template_voxel={config.template_voxel}"
    )

    bottle_output_dir = output_dir / "bottle_icp"
    bottle_output_dir.mkdir(parents=True, exist_ok=True)
    bottle_stl = resolve_path(config.stl)
    source_pcd, template_path, template_name = bottle_pose.load_bottle_template_pointcloud(config)
    source_pcd, source_original_point_count = bottle_pose.downsample_bottle_template_pointcloud(
        source_pcd,
        config.template_voxel,
    )
    global_target_pcd = bottle_icp.numpy_to_open3d_pointcloud(combined_points)
    icp_target_pcd = bottle_icp.numpy_to_open3d_pointcloud(selected_downsampled)
    ransac_n = max(3, int(config.global_ransac_n))

    old_icp_max_iteration = bottle_icp.ICP_MAX_ITERATION
    bottle_icp.ICP_MAX_ITERATION = int(config.icp_max_iteration)
    try:
        # 2.RANSAC粗配准（global registration），5次循环
        global_choice = bottle_pose.run_global_registration_with_retries(
            source_pcd,
            global_target_pcd,
            config,
            ransac_n,
        )
        global_result = global_choice.result
        # 3.ICP 配准前的预处理步骤
        icp_target_down, _icp_target_fpfh = bottle_icp.preprocess_point_cloud_for_registration(
            icp_target_pcd,
            float(config.voxel),
        )
        # 4.ICP 精配准
        icp_result = bottle_icp.run_point_to_plane_icp(
            global_choice.source_down,
            icp_target_down,
            global_result.transformation,
            voxel_size=float(config.voxel),
        )
    finally:
        bottle_icp.ICP_MAX_ITERATION = old_icp_max_iteration

    global_transform_path = bottle_output_dir / "completion_bottle_global_transform.txt"
    icp_transform_path = bottle_output_dir / "completion_bottle_icp_transform.txt"
    summary_path = bottle_output_dir / "completion_bottle_icp_summary.json"
    np.savetxt(global_transform_path, global_result.transformation, fmt="%.9f")
    np.savetxt(icp_transform_path, icp_result.transformation, fmt="%.9f")

    bottle_summary = {
        "bottle_stl": str(bottle_stl),
        "template": template_name,
        "template_path": str(template_path),
        "source_original_point_count": int(source_original_point_count),
        "source_point_count": int(len(source_pcd.points)),
        "voxel_size": float(config.voxel),
        "template_voxel_size": None if config.template_voxel <= 0 else float(config.template_voxel),
        "global_ransac_n": int(ransac_n),
        "global_ransac_attempts": int(max(1, int(config.global_ransac_attempts))),
        "global_ransac_selected_attempt": int(global_choice.attempt_count),
        "global_centroid_distance": float(global_choice.centroid_distance),
        "model_sample_count": int(config.model_sample_count),
        "model_even_radius": None if config.model_even_radius is None else float(config.model_even_radius),
        "icp_max_iteration": int(config.icp_max_iteration),
        "global_fitness": float(global_result.fitness),
        "global_inlier_rmse": float(global_result.inlier_rmse),
        "icp_fitness": float(icp_result.fitness),
        "icp_inlier_rmse": float(icp_result.inlier_rmse),
        "source_path": None,
        "registered_model_path": None,
        "pointcloud_outputs_saved": False,
        "global_transform_path": str(global_transform_path),
        "icp_transform_path": str(icp_transform_path),
        "global_transform": np.asarray(global_result.transformation, dtype=np.float64).tolist(),
        "icp_transform": np.asarray(icp_result.transformation, dtype=np.float64).tolist(),
        "summary_path": str(summary_path),
        "pipeline": "global_registration_target=selected_world+completed_world; icp_target=selected_world_only",
        "target_source": "global_selected_downsampled_plus_completed_downsampled__icp_selected_downsampled_only",
        "selected_original_point_count": int(len(selected_points_world)),
        "completed_original_point_count": int(len(completed_points_world)),
        "selected_downsampled_point_count": int(len(selected_downsampled)),
        "completed_downsampled_point_count": int(len(completed_downsampled)),
        "combined_before_final_downsample_point_count": int(len(combined_before_final)),
        "target_point_count": int(len(combined_points)),
        "global_target_point_count": int(len(combined_points)),
        "icp_target_point_count": int(len(selected_downsampled)),
        "target_voxel_size": None if target_voxel_size is None or target_voxel_size <= 0 else float(target_voxel_size),
        "target_selected_downsampled_path": str(selected_target_path),
        "target_completed_downsampled_path": str(completed_target_path),
        "target_path": str(combined_target_path),
        "global_target_path": str(combined_target_path),
        "icp_target_path": str(selected_target_path),
        "output_dir": str(output_dir),
    }
    bottle_summary.update(save_registered_bottle_template_pointclouds(bottle_summary, config, output_dir))
    summary_path.write_text(json.dumps(bottle_summary, indent=2), encoding="utf-8")
    log(
        "[box_object] completion bottle registration result: "
        f"global_target_points={len(combined_points)}, icp_target_points={len(selected_downsampled)}, "
        f"global_fitness={global_result.fitness:.6f}, global_rmse={global_result.inlier_rmse:.6f}, "
        f"icp_fitness={icp_result.fitness:.6f}, icp_rmse={icp_result.inlier_rmse:.6f}"
    )
    return bottle_summary

def local_points_to_world(points_local: np.ndarray, homomat: np.ndarray) -> np.ndarray:
    points_h = np.column_stack((points_local, np.ones(len(points_local))))
    return (homomat @ points_h.T).T[:, :3]


def make_concave_region_lineset(bounds: BoxLocalBounds, box_transform: np.ndarray):
    import open3d as o3d

    x0, y0, z0 = bounds.xy_min[0], bounds.xy_min[1], bounds.z_min
    x1, y1, z1 = bounds.xy_max[0], bounds.xy_max[1], bounds.z_max
    corners_local = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float64)
    corners_world = local_points_to_world(corners_local, box_transform)
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)
    color = np.asarray(BOX_OBJECT_CONCAVE_REGION_RGB, dtype=np.float64).reshape(1, 3)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return line_set, corners_world



def make_transformed_box_mesh(box_transform: np.ndarray):
    import open3d as o3d

    if not BOX_MODEL_PATH.exists():
        log(f"[box_object] Warning: missing box model {BOX_MODEL_PATH}")
        return None
    mesh = o3d.io.read_triangle_mesh(str(BOX_MODEL_PATH))
    if mesh.is_empty():
        log(f"[box_object] Warning: failed to load box model {BOX_MODEL_PATH}")
        return None
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.55, 0.82, 1.0])
    mesh.transform(box_transform)
    return mesh


def read_ascii_ply_points(ply_path: Path) -> np.ndarray:
    with ply_path.open("r", encoding="ascii") as file:
        vertex_count = None
        for line in file:
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex count: {ply_path}")
        points = []
        for _ in range(vertex_count):
            values = file.readline().split()
            if len(values) < 3:
                break
            points.append([float(values[0]), float(values[1]), float(values[2])])
    return np.asarray(points, dtype=np.float64)


def load_template_points_for_preview(ply_path: Path) -> np.ndarray:
    ply_path = resolve_path(ply_path)
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points, dtype=np.float64)
        if len(points) > 0:
            return points
    except ImportError:
        pass
    return read_ascii_ply_points(ply_path)


def project_template_points_for_preview(points: np.ndarray, template_name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centered = points - points.mean(axis=0)
    if template_name == "top":
        projected = centered[:, [0, 1]]
    elif template_name == "front":
        projected = centered[:, [0, 2]]
    elif template_name in ("left", "right"):
        projected = centered[:, [1, 2]]
    else:
        rz = np.deg2rad(-35.0)
        rx = np.deg2rad(62.0)
        rot_z = np.array([
            [np.cos(rz), -np.sin(rz), 0.0],
            [np.sin(rz), np.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ])
        rot_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx), np.cos(rx)],
        ])
        rotated = centered @ (rot_z @ rot_x).T
        projected = rotated[:, [0, 2]]
    return projected


def draw_preview_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.55) -> None:
    try:
        import cv2
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), 1, cv2.LINE_AA)
        return
    except ImportError:
        pass
    try:
        from PIL import Image, ImageDraw
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        draw.text(origin, text, fill=(35, 35, 35))
        image[:] = np.asarray(pil_image)
    except ImportError:
        return


def make_template_preview_tile(template_name: str, points: np.ndarray, tile_size: int, index: int) -> np.ndarray:
    tile_size = int(max(180, tile_size))
    image = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    projected = project_template_points_for_preview(points, template_name)
    if len(projected) > 0:
        pad = 22
        label_h = 48
        xy_min = projected.min(axis=0)
        xy_max = projected.max(axis=0)
        span = np.maximum(xy_max - xy_min, 1e-9)
        scale = min((tile_size - pad * 2) / span[0], (tile_size - label_h - pad * 2) / span[1])
        coords = (projected - (xy_min + xy_max) / 2.0) * scale
        xs = np.rint(coords[:, 0] + tile_size / 2.0).astype(np.int64)
        ys = np.rint(tile_size - label_h / 2.0 - (coords[:, 1] + (tile_size - label_h) / 2.0)).astype(np.int64)
        valid = (xs >= 2) & (xs < tile_size - 2) & (ys >= 2) & (ys < tile_size - label_h - 2)
        xs = xs[valid]
        ys = ys[valid]
        color = {
            "surface": np.array([230, 35, 25], dtype=np.uint8),
            "top": np.array([0, 180, 230], dtype=np.uint8),
            "front": np.array([0, 190, 70], dtype=np.uint8),
            "left": np.array([235, 155, 0], dtype=np.uint8),
            "right": np.array([170, 60, 220], dtype=np.uint8),
        }.get(template_name, np.array([230, 35, 25], dtype=np.uint8))
        image[ys, xs] = color
        image[np.clip(ys + 1, 0, tile_size - 1), xs] = color
        image[ys, np.clip(xs + 1, 0, tile_size - 1)] = color
    image[-44:, :] = 245
    draw_preview_text(image, f"{index}. {template_name}", (12, tile_size - 24), scale=0.62)
    draw_preview_text(image, f"{len(points)} pts", (12, tile_size - 8), scale=0.43)
    try:
        import cv2
        cv2.rectangle(image, (0, 0), (tile_size - 1, tile_size - 1), (120, 120, 120), 1)
    except ImportError:
        image[0, :, :] = 120
        image[-1, :, :] = 120
        image[:, 0, :] = 120
        image[:, -1, :] = 120
    return image


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        cv2.imwrite(str(path), image[:, :, ::-1])
        return
    except ImportError:
        pass
    from PIL import Image
    Image.fromarray(image).save(path)


def build_bottle_template_prompt_image(output_dir: Path, config: bottle_pose.BottleIcpPoseConfig):
    preview_dir = output_dir / "bottle_template_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    templates = []
    tiles = []
    tile_size = int(config.template_preview_size)
    for idx, template_name in enumerate(BOTTLE_PROMPT_TEMPLATE_CHOICES, start=1):
        template_path = resolve_path(BOTTLE_TEMPLATE_PATHS[template_name])
        if not template_path.exists():
            log(f"[box_object] template preview missing: {template_name} ({template_path})")
            continue
        points = load_template_points_for_preview(template_path)
        tile = make_template_preview_tile(template_name, points, tile_size, idx)
        save_rgb_image(preview_dir / f"bottle_template_{template_name}_preview.png", tile)
        templates.append({"name": template_name, "path": template_path, "points": len(points), "index": idx})
        tiles.append(tile)
    if not templates:
        raise FileNotFoundError("No bottle template PLY files found. Run python yanjiuyuan/sample_bottle_surface.py first.")

    cols = min(3, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    gap = 14
    header_h = 60
    grid_w = cols * tile_size + (cols + 1) * gap
    grid_h = header_h + rows * tile_size + (rows + 1) * gap
    canvas = np.full((grid_h, grid_w, 3), 250, dtype=np.uint8)
    draw_preview_text(canvas, "Choose bottle template: click a tile or press 1-5", (18, 28), scale=0.72)
    draw_preview_text(canvas, "Enter=surface, Esc/Q=cancel to surface", (18, 52), scale=0.50)
    rects = []
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x0 = gap + col * (tile_size + gap)
        y0 = header_h + gap + row * (tile_size + gap)
        canvas[y0:y0 + tile_size, x0:x0 + tile_size] = tile
        rects.append((x0, y0, x0 + tile_size, y0 + tile_size))
    grid_path = preview_dir / "bottle_template_prompt_grid.png"
    save_rgb_image(grid_path, canvas)
    log(f"[box_object] saved bottle template prompt image: {grid_path}")
    return canvas, templates, rects, grid_path


def choose_bottle_template_from_console(templates: list[dict], grid_path: Path) -> str:
    print(f"Bottle template preview image: {grid_path}")
    for display_idx, item in enumerate(templates, start=1):
        print(f"  {display_idx}. {item['name']}  ({item['points']} pts)  {item['path']}")
    try:
        choice = input("Select bottle template [1=surface]: ").strip()
    except EOFError:
        choice = ""
    if not choice:
        return templates[0]["name"]
    try:
        selected_idx = int(choice) - 1
    except ValueError:
        selected_idx = 0
    if selected_idx < 0 or selected_idx >= len(templates):
        selected_idx = 0
    return templates[selected_idx]["name"]


def choose_bottle_template_with_prompt(output_dir: Path, config: bottle_pose.BottleIcpPoseConfig) -> str:
    image, templates, rects, grid_path = build_bottle_template_prompt_image(output_dir, config)
    if not config.template_prompt_gui:
        selected = choose_bottle_template_from_console(templates, grid_path)
        log(f"[box_object] selected bottle template: {selected}")
        return selected
    try:
        import cv2
    except ImportError:
        selected = choose_bottle_template_from_console(templates, grid_path)
        log(f"[box_object] selected bottle template: {selected}")
        return selected

    selected = {"name": None}

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for item, rect in zip(templates, rects):
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                selected["name"] = item["name"]
                break

    window_name = "choose bottle template"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, image[:, :, ::-1])
    log("[box_object] choose bottle template: click a tile, or press 1-5. Enter/Esc defaults to surface.")
    while selected["name"] is None:
        key = cv2.waitKey(40) & 0xFF
        if key in (13, 10, 27, ord("q"), ord("Q")):
            selected["name"] = templates[0]["name"]
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(templates):
                selected["name"] = templates[idx]["name"]
    cv2.destroyWindow(window_name)
    log(f"[box_object] selected bottle template: {selected['name']}")
    return str(selected["name"])

def write_extraction_outputs(
    output_dir: Path,
    capture: CaptureData,
    box_transform: np.ndarray,
    box_transform_path: Path,
    detected_box_now: bool,
    bounds: BoxLocalBounds,
    masks: ExtractionMasks,
    mask_path: Optional[Path],
    point_hint_summary: Optional[dict],
    args: argparse.Namespace,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = capture.points_world

    transform_out_path = output_dir / "box_transform_used.txt"
    summary_path = output_dir / "box_object_extraction_summary.json"
    box_mesh_path = output_dir / "box_model_registered_light_blue.ply"

    np.savetxt(transform_out_path, box_transform, fmt="%.9f")
    box_mesh = make_transformed_box_mesh(box_transform) if args.show_box_model else None
    if box_mesh is not None:
        import open3d as o3d
        o3d.io.write_triangle_mesh(str(box_mesh_path), box_mesh, write_ascii=False)

    summary = {
        "capture_dir": str(capture.capture_dir),
        "target_ply": None if capture.target_ply is None else str(capture.target_ply),
        "camera_ply": None if capture.camera_ply is None else str(capture.camera_ply),
        "rgb_path": None if capture.rgb_path is None else str(capture.rgb_path),
        "target_frame": capture.frame,
        "box_transform_source": str(box_transform_path),
        "detected_box_now": bool(detected_box_now),
        "box_transform_used": str(transform_out_path),
        "concave_xy_min": bounds.xy_min.tolist(),
        "concave_xy_max": bounds.xy_max.tolist(),
        "concave_z_min": bounds.z_min,
        "concave_z_max": bounds.z_max,
        "raw_point_count": int(len(points)),
        "box_region_count": int(masks.box_region.sum()),
        "candidate_count": int(masks.candidate.sum()),
        "removed_count": int(masks.removed.sum()),
        "mask_projected_count": int(masks.mask_projected.sum()),
        "selected_count": int(masks.selected.sum()),
        "mapped_point_count": int(masks.mapped_points.sum()),
        "inner_xy_margin": args.inner_xy_margin,
        "top_overhang_xy_margin": args.top_overhang_xy_margin,
        "above_top_margin": args.above_top_margin,
        "remove_blue_box_points": args.remove_blue_box_points,
        "show_removed_context": args.show_removed_context,
        "removed_context_xy_margin": args.removed_context_xy_margin,
        "removed_context_z_margin": args.removed_context_z_margin,
        "show_box_model": args.show_box_model,
        "mask_path": None if mask_path is None else str(mask_path),
        "point_hint_summary": point_hint_summary,
        "candidate_path": None,
        "removed_gray_path": None,
        "selected_red_path": None,
        "combined_colored_path": None,
        "concave_region_wireframe_path": None,
        "concave_region_corners_world_path": None,
        "box_model_registered_path": str(box_mesh_path) if box_mesh is not None else None,
        "pointcloud_outputs_saved": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary

def show_pointclouds(summary: dict) -> None:
    import open3d as o3d

    combined_path = summary.get("combined_colored_path")
    remaining_path = summary.get("remaining_pointcloud_path")
    if not combined_path and not remaining_path:
        log("[box_object] No saved point-cloud outputs to show.")
        return
    geometries = []
    if combined_path:
        pcd = o3d.io.read_point_cloud(combined_path)
        geometries.append(pcd)
    if remaining_path:
        remaining_pcd = o3d.io.read_point_cloud(remaining_path)
        if not remaining_pcd.is_empty():
            geometries.append(remaining_pcd)
    box_mesh_path = summary.get("box_model_registered_path")
    if box_mesh_path:
        box_mesh = o3d.io.read_triangle_mesh(box_mesh_path)
        if not box_mesh.is_empty():
            box_mesh.compute_vertex_normals()
            geometries.append(box_mesh)
    concave_path = summary.get("concave_region_wireframe_path")
    if concave_path:
        line_set = o3d.io.read_line_set(concave_path)
        geometries.append(line_set)
    bottle_summary = summary.get("bottle_icp") or {}
    registered_bottle_points_path = bottle_summary.get("global_registered_path")
    if registered_bottle_points_path:
        bottle_points = o3d.io.read_point_cloud(registered_bottle_points_path)
        if not bottle_points.is_empty():
            bottle_points.paint_uniform_color(GLOBAL_REGISTERED_POINTS_RGB)
            geometries.append(bottle_points)
    registered_bottle_model_path = bottle_summary.get("registered_model_path")
    if registered_bottle_model_path:
        bottle_mesh = o3d.io.read_triangle_mesh(registered_bottle_model_path)
        if not bottle_mesh.is_empty():
            bottle_mesh.compute_vertex_normals()
            geometries.append(bottle_mesh)
    log("[box_object] Viewer colors: red=selected object, green=kept candidate points, gray=removed/context points, cyan wireframe=concave region, purple=global registered template points, yellow mesh=registered bottle.STL, original-color=remaining point cloud (original minus bottle mask).")
    o3d.visualization.draw_geometries(geometries, window_name="box object point cloud + bottle ICP", width=1280, height=720)


def show_pointcloud(ply_path: Path) -> None:
    # 单独展示点云图
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        log(f"[box_object] point cloud is empty: {ply_path}")
        return
    log(f"[box_object] Showing point cloud: {ply_path} ({len(pcd.points)} points)")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="point cloud",
        width=1280,
        height=720,
    )


def prepare_pipeline_context(args: argparse.Namespace) -> PipelineContext:
    with timed_step("resolve/load saved capture"):
        capture = resolve_capture(args)
    return prepare_pipeline_context_from_capture(args, capture)


def prepare_pipeline_context_from_capture(args: argparse.Namespace, capture: CaptureData) -> PipelineContext:
    output_dir = resolve_path(args.output_dir) if args.output_dir is not None else default_output_dir(capture.capture_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[box_object] output_dir: {output_dir}")
    if capture.target_ply is None:
        log(f"[box_object] target_ply: in-memory ({capture.frame})")

    with timed_step("load or estimate box transform"):
        box_transform, box_transform_path, detected_box_now = load_box_transform(args, capture, output_dir)
    log(f"[box_object] box_transform_source: {box_transform_path}")

    with timed_step("compute configured concave local region"):
        bounds = compute_box_local_bounds()

    with timed_step("crop object candidates in box local frame and remove walls"):
        _points_local, box_region_mask, candidate_mask, removed_mask = compute_geometric_candidate_masks(
            capture.points_world,
            capture.colors,
            box_transform,
            bounds,
            args,
        )

    with timed_step("load RGB and apply captured point pixel indices"):
        if capture.rgb_image_bgr is not None:
            import cv2

            image_rgb = cv2.cvtColor(np.asarray(capture.rgb_image_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        else:
            if capture.rgb_path is None:
                raise RuntimeError("CaptureData needs either rgb_image_bgr or rgb_path for SAM segmentation.")
            image_rgb = load_rgb_image_rgb(capture.rgb_path)
        pixel_indices, mapped_points, _mapped_ratio = resolve_pixel_indices_for_capture(capture, image_rgb, args)
        auto_box = bbox_from_pixel_indices(pixel_indices, candidate_mask & mapped_points, image_rgb.shape[1], image_rgb.shape[0])
        if auto_box is not None:
            if args.auto_segment_box:
                log(f"[box_object] auto 2D box prompt enabled: {auto_box}")
            else:
                log(f"[box_object] auto 2D candidate bbox computed but not passed to SAM: {auto_box}")

    return PipelineContext(
        args=args,
        capture=capture,
        output_dir=output_dir,
        box_transform=box_transform,
        box_transform_path=box_transform_path,
        detected_box_now=detected_box_now,
        bounds=bounds,
        box_region_mask=box_region_mask,
        candidate_mask=candidate_mask,
        removed_mask=removed_mask,
        image_rgb=image_rgb,
        pixel_indices=pixel_indices,
        mapped_points=mapped_points,
        auto_box=auto_box,
    )

def run_segmentation_and_bottle_icp_attempt(ctx: PipelineContext) -> tuple[dict, ExtractionMasks, np.ndarray]:
    # 分割→点云补全→ICP匹配
    attempt_args = copy.copy(ctx.args)
    total_start = perf_counter()
    segmentation_start = perf_counter()
    bottle_config = bottle_icp_config_from_runtime_options(attempt_args)
    # 分割
    with timed_step("run/load point_hint_segment mask"):    # 包含人工点点的时间
        # sam分割
        mask_image, mask_path, point_hint_summary = run_or_load_point_hint_mask(
            attempt_args,
            ctx.capture.rgb_path,
            ctx.output_dir,
            ctx.auto_box,
            image_bgr=ctx.capture.rgb_image_bgr,
        )

    with timed_step("project 2D mask to 3D and select one object"):
        # 2D mask 投影到 3D，选出瓶子点云
        selected_mask, mask_projected = apply_2d_mask_to_points(mask_image, ctx.pixel_indices, ctx.candidate_mask)
        if not np.any(selected_mask):
            raise RuntimeError("No selected object points after applying the 2D mask and box candidate constraints.")
        masks = ExtractionMasks(
            box_region=ctx.box_region_mask, # 箱子区域 mask
            candidate=ctx.candidate_mask,   # 候选 mask（箱子内的点）
            removed=ctx.removed_mask,   # 被移除的点（蓝色箱体等）
            selected=selected_mask, # 最终选中的瓶子点
            mask_projected=mask_projected,  # 2D mask 投影到 3D 的结果
            mapped_points=ctx.mapped_points,    # 深度图到点云的像素映射
        )
        segmentation_elapsed_s = perf_counter() - segmentation_start

    with timed_step("write extraction summary"):
        summary = write_extraction_outputs(
            output_dir=ctx.output_dir,
            capture=ctx.capture,
            box_transform=ctx.box_transform,
            box_transform_path=ctx.box_transform_path,
            detected_box_now=ctx.detected_box_now,
            bounds=ctx.bounds,
            masks=masks,
            mask_path=mask_path,
            point_hint_summary=point_hint_summary,
            args=attempt_args,
        )
    # 输出去除物体掩码对应的点云后的剩余点云
    with timed_step("save remaining point cloud (original minus bottle mask)"): # 大约0.05s
        import open3d as o3d
        import cv2
        # 对2D物体掩码进行闭运算（膨胀再腐蚀）填洞
        kernel_size = 15    # 核大小，改大→填更大的洞，改小→只填小洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))   # 创建结构元素。形状是椭圆形，大小是 kernel_size x kernel_size
        # mask_image_closed = cv2.morphologyEx(mask_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)  # 对掩码图像执行形态学闭运算,先膨胀后腐蚀
        mask_image_closed = cv2.morphologyEx(mask_image.astype(np.uint8), cv2.MORPH_DILATE, kernel) # 只膨胀
        mask_image_closed = mask_image_closed.astype(bool)
        log(f"[box_object] Mask closing (dilate+erode, kernel={kernel_size}): "
            f"before={int(mask_image.sum())}, after={int(mask_image_closed.sum())}")
        # 将填洞后的2D掩码重新投影到3D点云
        flat_mask_closed = mask_image_closed.reshape(-1)
        closed_projected = np.zeros(len(ctx.pixel_indices), dtype=bool)
        mapped = (ctx.pixel_indices >= 0) & (ctx.pixel_indices < len(flat_mask_closed))
        closed_projected[mapped] = flat_mask_closed[ctx.pixel_indices[mapped]]
        closed_selected = ctx.candidate_mask & closed_projected
        log(f"[box_object] Closed mask projected to 3D: selected={int(closed_selected.sum())} "
            f"(original selected={int(selected_mask.sum())})")
        # 用填洞后的掩码计算剩余点云
        remaining_mask = ~closed_selected
        # 对掩码取最小外接旋转矩形，扩大后投影回3D过滤远点
        mask_uint8 = mask_image_closed.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_contour_pts = np.vstack(contours)
            rect = cv2.minAreaRect(all_contour_pts)  # (center, (w, h), angle)
            rect_w, rect_h = rect[1]
            # 估算像素→米的比例：用物体3D外接盒XY尺寸 / 2D矩形尺寸
            object_points = ctx.capture.points_world[closed_selected]
            obj_3d_extent = np.ptp(object_points[:, :2], axis=0)  # XY平面上的尺寸 [dx, dy]
            ref_3d = max(obj_3d_extent[0], obj_3d_extent[1])
            ref_2d = max(rect_w, rect_h)
            meters_per_pixel = ref_3d / ref_2d if ref_2d > 0 else 0.001
            expand_m = 0.025  # 每边扩大2.5cm（长宽各加5cm）
            expand_px = max(1, int(round(expand_m / meters_per_pixel)))
            # 扩大矩形
            new_rect = (rect[0], (rect_w + 2 * expand_px, rect_h + 2 * expand_px), rect[2])
            expanded_box = np.intp(cv2.boxPoints(new_rect))
            # 创建扩大矩形的2D掩码
            rect_mask_img = np.zeros(mask_image_closed.shape, dtype=np.uint8)
            cv2.fillPoly(rect_mask_img, [expanded_box], 1)
            # 投影回3D
            flat_rect_mask = rect_mask_img.reshape(-1)
            rect_projected = np.zeros(len(ctx.pixel_indices), dtype=bool)
            mapped_r = (ctx.pixel_indices >= 0) & (ctx.pixel_indices < len(flat_rect_mask))
            rect_projected[mapped_r] = flat_rect_mask[ctx.pixel_indices[mapped_r]].astype(bool)
            # 只保留在扩大矩形内且不在物体掩码内的点
            final_mask = remaining_mask & rect_projected
            log(f"[box_object] MinAreaRect expand (rect=({rect_w:.0f}x{rect_h:.0f}px, "
                f"expand={expand_m}m={expand_px}px, m/px={meters_per_pixel:.5f}): "
                f"remaining={int(remaining_mask.sum())}, in_rect={int(final_mask.sum())}")
        else:
            final_mask = remaining_mask
            log("[box_object] No contour found, skip rect filter")
        # 生成剩余点云
        remaining_points = ctx.capture.points_world[final_mask]
        remaining_pcd = o3d.geometry.PointCloud()
        remaining_pcd.points = o3d.utility.Vector3dVector(remaining_points)
        if ctx.capture.colors is not None:
            remaining_colors = ctx.capture.colors[final_mask]
            if remaining_colors.shape[1] == 4:
                remaining_colors = remaining_colors[:, :3]
            remaining_pcd.colors = o3d.utility.Vector3dVector(np.clip(remaining_colors, 0.0, 1.0))
        # 体素下采样，减少点数使点云稀疏
        voxel_size = 0.01  # 体素边长(米)，改大→更稀疏，改小→更密集
        down_pcd = remaining_pcd.voxel_down_sample(voxel_size)
        log(f"[box_object] Voxel downsample (voxel_size={voxel_size}m): "
            f"before={len(remaining_pcd.points)}, after={len(down_pcd.points)}")
        remaining_ply_path = ctx.output_dir / "remaining_pointcloud.ply"
        o3d.io.write_point_cloud(str(remaining_ply_path), down_pcd, write_ascii=False) # type:ignore
        summary["remaining_pointcloud_path"] = str(remaining_ply_path)
        Path(summary["summary_path"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"[box_object] Saved remaining point cloud (rect-expanded, downsampled): {remaining_ply_path} ({len(down_pcd.points)} points)")

    if bottle_config.enabled and bottle_config.template == "prompt":
        with timed_step("choose bottle template after object segmentation"):
            bottle_config.template = choose_bottle_template_with_prompt(ctx.output_dir, bottle_config)

    bottle_matching_elapsed_s = None
    completion_matching_elapsed_s = None
    if bottle_config.enabled:
        with timed_step("register bottle model to selected object"):
            matching_start = perf_counter()
            bottle_summary = bottle_pose.run_bottle_registration_on_selected(ctx.capture.points_world[selected_mask], ctx.output_dir, bottle_config)
            summary["bottle_icp"] = bottle_summary
            bottle_matching_elapsed_s = perf_counter() - matching_start
            Path(summary["summary_path"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if getattr(attempt_args, "completion_matching", False):
        # 补全完整点云
        with timed_step("complete selected point cloud with AdaPoinTr"):    # 10.03s
            matching_start = perf_counter()
            completion_config = completion_matching_config_from_runtime_options(attempt_args, ctx.output_dir)
            selected_points_world = ctx.capture.points_world[selected_mask]
            #  AdaPoinTr 点云补全 + 第一阶段 ICP（相机坐标系）
            with timed_step("complete selected point cloud +RANSAC 粗配准 + ICP 精配准"): # 12.28s
                completion_result = completion_matching.run_completion_matching_on_sam_world_points(
                    selected_points_world,
                    completion_config,
                    output_prefix=attempt_args.completion_output_prefix,
                    run_matching=True,
                )
                completion_summary = completion_result.summary
                completion_summary["surface_template_icp"] = "enabled"
            # 第二阶段 ICP（RANSAC 粗配准 + ICP 精配准）
            with timed_step("completion_bottle_icp"):   # 1.25s左右
                if getattr(attempt_args, "completion_bottle_icp", False):   # 如果有完整物体点云
                    completion_bottle_config = completion_bottle_icp_config_from_runtime_options(attempt_args)
                    completion_bottle_summary = run_completion_bottle_icp_on_selected_and_completed(
                        selected_points_world,
                        completion_result.completed_world_points,
                        ctx.output_dir / "completion_bottle_icp",
                        completion_bottle_config,
                        getattr(attempt_args, "completion_bottle_target_voxel_size", None),
                    )
                    completion_summary["completion_bottle_icp"] = completion_bottle_summary
                else:
                    completion_summary["completion_bottle_icp"] = "disabled"
                summary["completion_matching"] = completion_summary
                completion_matching_elapsed_s = perf_counter() - matching_start
                Path(summary["summary_path"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    matching_parts = [
        value for value in (bottle_matching_elapsed_s, completion_matching_elapsed_s)
        if value is not None
    ]
    matching_elapsed_s = sum(matching_parts) if matching_parts else None
    total_elapsed_s = perf_counter() - total_start
    summary["timing"] = {
        "segmentation_elapsed_s": segmentation_elapsed_s,
        "bottle_matching_elapsed_s": bottle_matching_elapsed_s,
        "completion_matching_elapsed_s": completion_matching_elapsed_s,
        "matching_elapsed_s": matching_elapsed_s,
        "total_attempt_elapsed_s": total_elapsed_s,
    }
    if "summary_path" in summary:
        Path(summary["summary_path"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    matching_text = "skipped" if matching_elapsed_s is None else f"{matching_elapsed_s:.2f}s"
    log(
        f"[box_object] timing: segmentation={segmentation_elapsed_s:.2f}s, "
        f"matching={matching_text}, total={total_elapsed_s:.2f}s"
    )
    print_summary(summary)
    return summary, masks, selected_mask


def print_summary(summary: dict) -> None:
    print(f"Kept candidate points (green): {summary['candidate_count']}")
    print(f"Removed/context points (gray): {summary['removed_count']}")
    print(f"Selected object points (red): {summary['selected_count']}")
    if summary.get("pointcloud_outputs_saved", True):
        print(f"Saved selected red PLY: {summary['selected_red_path']}")
        print(f"Saved removed gray PLY: {summary['removed_gray_path']}")
        print(f"Saved combined colored PLY: {summary['combined_colored_path']}")
        print(f"Saved concave region wireframe: {summary['concave_region_wireframe_path']}")
    else:
        print("Skipped extracted point-cloud PLY outputs.")
    if summary.get("box_model_registered_path"):
        print(f"Saved registered box model: {summary['box_model_registered_path']}")
    if summary.get("bottle_icp"):
        bottle_summary = summary["bottle_icp"]
        print(f"Bottle template: {bottle_summary['template']} ({bottle_summary['template_path']})")
        if bottle_summary.get("source_original_point_count") is not None:
            print(f"Bottle template points: {bottle_summary['source_original_point_count']} -> {bottle_summary['source_point_count']} (template_voxel={bottle_summary['template_voxel_size']}, ransac_n={bottle_summary['global_ransac_n']})")
        if bottle_summary.get("global_centroid_distance") is not None:
            print(f"Global registration centroid_dist={bottle_summary['global_centroid_distance']:.4f} m")
        print(f"Bottle ICP fitness/rmse: {bottle_summary['icp_fitness']:.6f} / {bottle_summary['icp_inlier_rmse']:.6f}")
        if bottle_summary.get("global_registered_path"):
            print(f"Saved global registered template points: {bottle_summary['global_registered_path']}")
        if bottle_summary.get("registered_model_path"):
            print(f"Saved registered bottle model: {bottle_summary['registered_model_path']}")
        print(f"Saved bottle ICP transform: {bottle_summary['icp_transform_path']}")
    if summary.get("completion_matching"):
        completion_summary = summary["completion_matching"]
        adapointr_summary = completion_summary["adapointr"]
        paths_summary = completion_summary["paths"]
        center = adapointr_summary.get("selected_center_camera") or [0.0, 0.0, 0.0]
        print(f"AdaPoinTr selected center(camera): ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
        print(f"AdaPoinTr network input: {adapointr_summary['network_input_camera_point_count']} centered + XY-flipped camera-frame points")
        outlier_summary = adapointr_summary.get("selected_outlier_filter") or {}
        if outlier_summary:
            print(f"Selected downsample/filter: {outlier_summary.get('input_count')} -> {outlier_summary.get('kept_count')} kept, removed={outlier_summary.get('removed_count')}, applied={outlier_summary.get('applied')}")
        print(f"Completion points: {completion_summary['sam_partial_world_point_count']} -> {completion_summary['completed_world_point_count']}")
        matching_summary = completion_summary.get("matching") or {}
        if matching_summary:
            print(f"Surface template ICP fitness/rmse: {matching_summary['icp_fitness']:.6f} / {matching_summary['icp_inlier_rmse']:.6f}")
            print(f"Saved surface template ICP world PLY: {matching_summary['icp_registered_world_path']}")
            print(f"Saved surface template ICP world transform: {matching_summary['icp_transform_world_path']}")
        else:
            print("Surface template ICP: not available")
        completion_bottle_summary = completion_summary.get("completion_bottle_icp") or {}
        if isinstance(completion_bottle_summary, dict) and completion_bottle_summary:
            print(
                "Completion bottle registration targets: "
                f"selected {completion_bottle_summary['selected_original_point_count']} -> {completion_bottle_summary['selected_downsampled_point_count']}, "
                f"completed {completion_bottle_summary['completed_original_point_count']} -> {completion_bottle_summary['completed_downsampled_point_count']}, "
                f"global combined={completion_bottle_summary['global_target_point_count']}, "
                f"ICP selected={completion_bottle_summary['icp_target_point_count']}"
            )
            print(
                "Completion bottle template points: "
                f"{completion_bottle_summary['source_original_point_count']} -> {completion_bottle_summary['source_point_count']} "
                f"(template_voxel={completion_bottle_summary['template_voxel_size']})"
            )
            print(f"Completion bottle global fitness/rmse: {completion_bottle_summary['global_fitness']:.6f} / {completion_bottle_summary['global_inlier_rmse']:.6f}")
            print(f"Completion bottle selected-only ICP fitness/rmse: {completion_bottle_summary['icp_fitness']:.6f} / {completion_bottle_summary['icp_inlier_rmse']:.6f}")
            print(f"Saved completion bottle global target PLY: {completion_bottle_summary['global_target_path']}")
            print(f"Saved completion bottle ICP target PLY: {completion_bottle_summary['icp_target_path']}")
            print(f"Saved completion bottle ICP transform: {completion_bottle_summary['icp_transform_path']}")
        elif completion_bottle_summary == "disabled":
            print("Completion bottle ICP: disabled")
        print(f"Saved pre-completion origin XY-flipped PLY: {adapointr_summary['partial_camera_origin_path']}")
        print(f"Saved network input origin PLY: {adapointr_summary['network_input_camera_path']}")
        print(f"Saved completed origin PLY: {adapointr_summary['completed_camera_path']}")
        print(f"Saved completed world PLY: {paths_summary['completed_world_path']}")
    print(f"Saved summary: {summary['summary_path']}")

def compute_camera_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        lookat = np.zeros(3, dtype=np.float64)
        return np.array([0.5, -0.5, 0.35], dtype=np.float64), lookat, 1.0
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    extent = max(float(np.max(max_corner - min_corner)), 1e-6)
    cam_dist = max(0.45, extent * 2.8)
    cam_pos = center + np.array([cam_dist, -cam_dist, max(cam_dist * 0.65, extent * 0.9)])
    return cam_pos, center, extent


def build_bottle_mesh_model(
    bottle_stl: Path,
    transform: np.ndarray,
    name: str = "bottle_icp_registered",
    alpha: float = 0.55,
    rgb: Optional[np.ndarray] = None,
):
    import wrs.modeling.geometric_model as mgm

    bottle_model = mgm.GeometricModel(
        initor=str(resolve_path(Path(bottle_stl))),
        name=name,
        toggle_twosided=True,
        rgb=np.array([1.0, 0.76, 0.18]) if rgb is None else np.asarray(rgb, dtype=np.float64),
        alpha=float(alpha),
    )
    transform = np.asarray(transform, dtype=np.float64)
    try:
        bottle_model.homomat = transform
    except Exception:
        bottle_model.pos = np.asarray(transform[:3, 3], dtype=np.float64)
        bottle_model.rotmat = np.asarray(transform[:3, :3], dtype=np.float64)
    return bottle_model


class InteractiveBottleIcpApp:
    def __init__(self, ctx: PipelineContext):
        import wrs.modeling.geometric_model as mgm
        import wrs.visualization.panda.world as wd
        from wrs.visualization.panda import panda3d_utils as pdu
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        self.ctx = ctx
        self.running = False
        self.attempt_count = 0
        self.result_models = []
        self.extra_window_models = {}
        self.extra_windows = {}
        self.mgm = mgm
        self.pdu = pdu

        static_mask = ctx.candidate_mask | ctx.removed_mask
        scene_points = ctx.capture.points_world[static_mask]
        if len(scene_points) == 0:
            scene_points = ctx.capture.points_world
        cam_pos, lookat_pos, extent = compute_camera_from_points(scene_points)
        self.base = wd.World(cam_pos=cam_pos, lookat_pos=lookat_pos, w=1280, h=720)

        frame_length = max(extent * 0.25, 0.03)
        frame_radius = max(frame_length * 0.015, 0.0005)
        mgm.gen_frame(ax_length=frame_length, ax_radius=frame_radius).attach_to(self.base)

        self.attach_static_pointclouds()
        action_text = "segment + complete + surface ICP + bottle ICP" if getattr(ctx.args, "completion_matching", False) else "segment + choose template + run bottle ICP"
        self.status_text = OnscreenText(
            text=f"Press D to {action_text}. Press D again to retry.",
            pos=(-1.28, 0.92),
            align=TextNode.ALeft,
            scale=0.045,
            fg=(0.02, 0.02, 0.02, 1.0),
            mayChange=True,
        )
        self.base.accept("d", self.run_attempt)
        print("Viewer colors: main window blue=remaining point cloud (original minus bottle mask), green=kept candidate points, gray=removed/context points, red=selected. ExtraWindow1 red=selected target, green=AdaPoinTr completion; ExtraWindow2 red=selected target, green=completion, purple=downsampled global template; ExtraWindow3 red=selected target, orange=downsampled ICP template, translucent yellow=bottle STL pose.")
        print(f"WRS viewer is ready. Press D in the viewer to run {action_text}.")

    def attach_static_pointclouds(self) -> None:
        args = self.ctx.args
        candidate_points = self.ctx.capture.points_world[self.ctx.candidate_mask]
        removed_points = self.ctx.capture.points_world[self.ctx.removed_mask]
        candidate_points, _ = voxel_downsample_arrays(candidate_points, None, args.candidate_voxel)
        removed_points, _ = voxel_downsample_arrays(removed_points, None, args.removed_voxel)
        if len(removed_points) > 0:
            self.mgm.gen_pointcloud(
                removed_points,
                rgba=np.array([0.55, 0.55, 0.55, 0.7]),
                point_size=args.point_size,
            ).attach_to(self.base)
        if len(candidate_points) > 0:
            self.mgm.gen_pointcloud(
                candidate_points,
                rgba=np.array([0.0, 0.85, 0.15, 0.78]),
                point_size=args.point_size,
            ).attach_to(self.base)

    def clear_result_models(self) -> None:
        for model in self.result_models:
            try:
                model.detach()
            except Exception:
                pass
        self.result_models = []
        self.clear_extra_result_models()

    def clear_extra_result_models(self, window_key: Optional[str] = None) -> None:
        keys = list(self.extra_window_models.keys()) if window_key is None else [window_key]
        for key in keys:
            for model in self.extra_window_models.get(key, []):
                try:
                    model.detach()
                except Exception:
                    pass
            self.extra_window_models[key] = []
            window = self.extra_windows.get(key)
            if window is None:
                continue
            try:
                cam_node = window.cam.node()
                for child in list(window.render.getChildren()):
                    if child.node() == cam_node:
                        continue
                    child.detachNode()
            except Exception as exc:
                print(f"[box_object] Warning: failed to clear ExtraWindow '{key}' render children: {exc}")

    def attach_remaining_pointcloud_window(self, summary: dict) -> None:
        remaining_ply_path = summary.get("remaining_pointcloud_path")
        if not remaining_ply_path:
            return
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(remaining_ply_path))
            if pcd.is_empty():
                print(f"[box_object] Remaining point cloud is empty: {remaining_ply_path}")
                return
            remaining_points = np.asarray(pcd.points, dtype=np.float64)
            remaining_points, _ = voxel_downsample_arrays(remaining_points, None, self.ctx.args.removed_voxel)
            if len(remaining_points) == 0:
                return
            self.clear_extra_result_models("remaining")
            self.ensure_extra_window(
                "remaining",
                remaining_points,
                "Remaining point cloud (original minus bottle mask)",
                origin=(40, 780),
                show_frame=True,
            )
            colors = None
            if pcd.has_colors():
                colors = np.asarray(pcd.colors, dtype=np.float64)
                if colors.shape[0] == len(np.asarray(pcd.points)):
                    colors = colors[np.asarray(pcd.points)[:, 0].argsort()]
            self.attach_points_to_extra_window(
                "remaining",
                remaining_points,
                rgba=np.array([0.2, 0.5, 0.9, 0.5]),
                label="remaining point cloud",
                point_size=self.ctx.args.point_size,
            )
            print(f"[box_object] ExtraWindow 'remaining' updated: {len(remaining_points)} points (original minus bottle mask).")
        except Exception as exc:
            print(f"[box_object] Warning: failed to show remaining point cloud in ExtraWindow: {exc}")
    def attach_completion_registered_world_result(self, completion_summary: dict) -> None:
        matching_summary = completion_summary.get("matching") or {}
        if not matching_summary:
            return
        registered_points_path = matching_summary.get("icp_registered_world_path") or matching_summary.get("global_registered_world_path")
        registered_points = self.read_pointcloud_points(
            registered_points_path,
            "surface template ICP registered world point cloud",
        )
        registered_points, _ = voxel_downsample_arrays(registered_points, None, self.ctx.args.selected_voxel)
        if len(registered_points) > 0:
            template_model = self.mgm.gen_pointcloud(
                registered_points,
                rgba=np.array([*COMPLETION_REGISTERED_POINTS_RGB, 0.95]),
                point_size=max(self.ctx.args.point_size, 0.0035),
            )
            template_model.attach_to(self.base)
            self.result_models.append(template_model)

        transform = matching_summary.get("icp_transform_world") or matching_summary.get("global_transform_world")
        if transform is not None:
            bottle_model = build_bottle_mesh_model(
                self.ctx.args.bottle_stl,
                np.asarray(transform, dtype=np.float64),
                name="completion_surface_icp_bottle_world",
                alpha=0.35,
                rgb=np.array([1.0, 0.76, 0.18]),
            )
            bottle_model.attach_to(self.base)
            self.result_models.append(bottle_model)

    def attach_result(self, summary: dict, selected_mask: np.ndarray) -> None:
        args = self.ctx.args
        selected_points = self.ctx.capture.points_world[selected_mask]
        selected_points, _ = voxel_downsample_arrays(selected_points, None, args.selected_voxel)
        if len(selected_points) > 0:
            selected_model = self.mgm.gen_pointcloud(
                selected_points,
                rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                point_size=max(args.point_size, 0.0025),
            )
            selected_model.attach_to(self.base)
            self.result_models.append(selected_model)
        bottle_summary = summary.get("bottle_icp") or {}
        registered_points_path = bottle_summary.get("global_registered_path")
        if registered_points_path:
            try:
                import open3d as o3d

                registered_pcd = o3d.io.read_point_cloud(str(registered_points_path))
                registered_points = np.asarray(registered_pcd.points, dtype=np.float64)
                registered_points, _ = voxel_downsample_arrays(registered_points, None, args.selected_voxel)
                if len(registered_points) > 0:
                    registered_points_model = self.mgm.gen_pointcloud(
                        registered_points,
                        rgba=np.array([*GLOBAL_REGISTERED_POINTS_RGB, 0.95]),
                        point_size=max(args.point_size, 0.0035),
                    )
                    registered_points_model.attach_to(self.base)
                    self.result_models.append(registered_points_model)
            except Exception as exc:
                print(f"[box_object] Warning: failed to draw global registered template points: {exc}")
        transform = bottle_summary.get("icp_transform")
        if transform is not None:
            bottle_model = build_bottle_mesh_model(self.ctx.args.bottle_stl, np.asarray(transform, dtype=np.float64))
            bottle_model.attach_to(self.base)
            self.result_models.append(bottle_model)
        completion_summary = summary.get("completion_matching")
        if completion_summary:
            self.attach_completion_result_extra_window(completion_summary)

    def read_pointcloud_points(self, path: Optional[str], label: str) -> np.ndarray:
        if not path:
            return np.empty((0, 3), dtype=np.float64)
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                return np.empty((0, 3), dtype=np.float64)
            return points
        except Exception as exc:
            print(f"[box_object] Warning: failed to read {label}: {exc}")
            return np.empty((0, 3), dtype=np.float64)

    def ensure_extra_window(
        self,
        window_key: str,
        points: np.ndarray,
        title: str,
        origin: tuple[int, int],
        show_frame: bool = False,
    ):
        cam_pos, lookat_pos, extent = compute_camera_from_points(points)
        window = self.extra_windows.get(window_key)
        if window is None:
            window = self.pdu.ExtraWindow(
                self.base,
                window_title=title,
                cam_pos=cam_pos,
                lookat_pos=lookat_pos,
                w=960,
                h=720,
            )
            self.extra_windows[window_key] = window
            self.extra_window_models.setdefault(window_key, [])
            try:
                window.set_origin(np.array(origin))
            except Exception:
                pass
        else:
            from panda3d.core import Point3, Vec3

            window.cam.setPos(Point3(cam_pos[0], cam_pos[1], cam_pos[2]))
            window.cam.lookAt(
                Point3(lookat_pos[0], lookat_pos[1], lookat_pos[2]),
                Vec3(0, 0, 1),
            )
            try:
                window.set_win_props(title=title, size=tuple(window.size))
            except Exception:
                pass
        if show_frame:
            frame = self.mgm.gen_frame(ax_length=max(extent * 0.18, 0.025), ax_radius=max(extent * 0.0015, 0.0005))
            frame.pdndp.reparentTo(window.render)
            self.extra_window_models.setdefault(window_key, []).append(frame)
        return window

    def attach_points_to_extra_window(self, window_key: str, points: np.ndarray, rgba: np.ndarray, label: str, point_size: float) -> None:
        window = self.extra_windows.get(window_key)
        if window is None or len(points) == 0:
            return
        try:
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            if len(points) == 0:
                return
            model = self.mgm.gen_pointcloud(points, rgba=rgba, point_size=point_size)
            model.pdndp.reparentTo(window.render)
            self.extra_window_models.setdefault(window_key, []).append(model)
        except Exception as exc:
            print(f"[box_object] Warning: failed to draw {label} in ExtraWindow '{window_key}': {exc}")

    def attach_model_to_extra_window(self, window_key: str, model, label: str) -> None:
        window = self.extra_windows.get(window_key)
        if window is None or model is None:
            return
        try:
            model.pdndp.reparentTo(window.render)
            self.extra_window_models.setdefault(window_key, []).append(model)
        except Exception as exc:
            print(f"[box_object] Warning: failed to draw {label} in ExtraWindow '{window_key}': {exc}")

    def completion_template_display_voxel(self, completion_bottle_summary: dict) -> Optional[float]:
        template_voxel = completion_bottle_summary.get("template_voxel_size")
        if template_voxel is None or template_voxel <= 0:
            template_voxel = getattr(self.ctx.args, "completion_bottle_template_voxel_size", None)
        if template_voxel is None or template_voxel <= 0:
            return 0.006
        return max(float(template_voxel) * 2.0, 0.006)

    def attach_completion_result_extra_window(self, completion_summary: dict) -> None:
        paths_summary = completion_summary.get("paths") or {}
        completion_bottle_summary = completion_summary.get("completion_bottle_icp") or {}
        has_completion_bottle = isinstance(completion_bottle_summary, dict) and bool(completion_bottle_summary)

        selected_points = self.read_pointcloud_points(
            completion_bottle_summary.get("target_selected_downsampled_path") if has_completion_bottle else paths_summary.get("partial_world_path"),
            "selected world point cloud",
        )
        completed_points = self.read_pointcloud_points(
            completion_bottle_summary.get("target_completed_downsampled_path") if has_completion_bottle else paths_summary.get("completed_world_path"),
            "AdaPoinTr completed world point cloud",
        )
        display_voxel = getattr(self.ctx.args, "selected_voxel", None)
        if not has_completion_bottle:
            selected_points, _ = voxel_downsample_arrays(selected_points, None, display_voxel)
            completed_points, _ = voxel_downsample_arrays(completed_points, None, display_voxel)

        completion_sets = [pts for pts in (selected_points, completed_points) if len(pts) > 0]
        if completion_sets:
            self.clear_extra_result_models("completion")
            self.ensure_extra_window(
                "completion",
                np.vstack(completion_sets),
                "1 AdaPoinTr completion world frame",
                origin=(1320, 40),
                show_frame=False,
            )
            self.attach_points_to_extra_window(
                "completion",
                selected_points,
                rgba=np.array([*NETWORK_INPUT_POINTS_RGB, 1.0]),
                label="selected world target point cloud",
                point_size=max(self.ctx.args.point_size, 0.0035),
            )
            self.attach_points_to_extra_window(
                "completion",
                completed_points,
                rgba=np.array([*COMPLETED_POINTS_RGB, 0.72]),
                label="AdaPoinTr completed world point cloud",
                point_size=max(self.ctx.args.point_size, 0.003),
            )
        else:
            print("[box_object] Warning: no selected/completed point clouds to draw in completion ExtraWindow.")

        if has_completion_bottle:
            template_display_voxel = self.completion_template_display_voxel(completion_bottle_summary)
            global_registered_points = self.read_pointcloud_points(
                completion_bottle_summary.get("global_registered_path"),
                "completion bottle global registered template point cloud",
            )
            global_registered_points, _ = voxel_downsample_arrays(
                global_registered_points,
                None,
                template_display_voxel,
            )
            global_sets = [pts for pts in (selected_points, completed_points, global_registered_points) if len(pts) > 0]
            if global_sets:
                self.clear_extra_result_models("global")
                self.ensure_extra_window(
                    "global",
                    np.vstack(global_sets),
                    "2 Global ICP selected + completion + template",
                    origin=(1320, 800),
                    show_frame=False,
                )
                self.attach_points_to_extra_window(
                    "global",
                    selected_points,
                    rgba=np.array([*NETWORK_INPUT_POINTS_RGB, 1.0]),
                    label="selected world global target point cloud",
                    point_size=max(self.ctx.args.point_size, 0.0035),
                )
                self.attach_points_to_extra_window(
                    "global",
                    completed_points,
                    rgba=np.array([*COMPLETED_POINTS_RGB, 0.72]),
                    label="AdaPoinTr completed world point cloud",
                    point_size=max(self.ctx.args.point_size, 0.003),
                )
                self.attach_points_to_extra_window(
                    "global",
                    global_registered_points,
                    rgba=np.array([*COMPLETION_GLOBAL_REGISTERED_POINTS_RGB, 0.95]),
                    label="downsampled global registered template point cloud",
                    point_size=max(self.ctx.args.point_size, 0.0038),
                )
            else:
                print("[box_object] Warning: no points to draw in global ExtraWindow.")

            icp_registered_points = self.read_pointcloud_points(
                completion_bottle_summary.get("icp_registered_path"),
                "completion bottle ICP registered template point cloud",
            )
            icp_registered_points, _ = voxel_downsample_arrays(
                icp_registered_points,
                None,
                template_display_voxel,
            )
            icp_sets = [pts for pts in (selected_points, icp_registered_points) if len(pts) > 0]
            transform = completion_bottle_summary.get("icp_transform")
            if icp_sets or transform is not None:
                camera_points = np.vstack(icp_sets) if icp_sets else np.zeros((1, 3), dtype=np.float64)
                self.clear_extra_result_models("icp")
                self.ensure_extra_window(
                    "icp",
                    camera_points,
                    "3 Selected ICP target + template + bottle pose",
                    origin=(2300, 40),
                    show_frame=False,
                )
                self.attach_points_to_extra_window(
                    "icp",
                    selected_points,
                    rgba=np.array([*NETWORK_INPUT_POINTS_RGB, 1.0]),
                    label="selected world ICP target point cloud",
                    point_size=max(self.ctx.args.point_size, 0.0035),
                )
                self.attach_points_to_extra_window(
                    "icp",
                    icp_registered_points,
                    rgba=np.array([*COMPLETION_REGISTERED_POINTS_RGB, 0.95]),
                    label="downsampled ICP registered template point cloud",
                    point_size=max(self.ctx.args.point_size, 0.0038),
                )
                if transform is not None:
                    bottle_model = build_bottle_mesh_model(
                        self.ctx.args.bottle_stl,
                        np.asarray(transform, dtype=np.float64),
                        name="completion_bottle_icp_registered_model_world",
                        alpha=0.35,
                        rgb=np.array([1.0, 0.76, 0.18]),
                    )
                    self.attach_model_to_extra_window("icp", bottle_model, "completion bottle ICP STL model")
            print("[box_object] ExtraWindows updated: 1 completion, 2 selected+completion+downsampled global template, 3 selected+downsampled ICP template+transparent bottle model.")
        else:
            self.clear_extra_result_models("global")
            self.clear_extra_result_models("icp")
            print("[box_object] ExtraWindow updated: completion window shows red=selected target and green=completion.")

    def run_attempt(self) -> None:
        if self.running:
            print("[box_object] Detection is already running; ignoring D key.")
            return
        self.running = True
        self.attempt_count += 1
        self.status_text.setText(f"Attempt {self.attempt_count}: running segmentation/template/ICP...") # type:ignore
        self.clear_result_models()
        try:
            summary, _masks, selected_mask = run_segmentation_and_bottle_icp_attempt(self.ctx)
            self.attach_result(summary, selected_mask)
            self.attach_remaining_pointcloud_window(summary)
            if summary.get("completion_matching"):
                completion_summary = summary["completion_matching"]
                adapointr_summary = completion_summary["adapointr"]
                completion_bottle_summary = completion_summary.get("completion_bottle_icp") or {}
                if isinstance(completion_bottle_summary, dict) and completion_bottle_summary:
                    self.status_text.setText(
                        f"Attempt {self.attempt_count}: done. AdaPoinTr output={completion_summary['completed_world_point_count']}, "
                        f"bottle ICP fitness={completion_bottle_summary['icp_fitness']:.4f}. Press D to retry."
                    )
                else:
                    self.status_text.setText(
                        f"Attempt {self.attempt_count}: done. AdaPoinTr input={adapointr_summary['network_input_camera_point_count']} "
                        f"output={completion_summary['completed_world_point_count']}. Press D to retry."
                    )
            elif summary.get("bottle_icp"):
                bottle_summary = summary["bottle_icp"]
                self.status_text.setText(
                    f"Attempt {self.attempt_count}: done. Template={bottle_summary['template']} "
                    f"fitness={bottle_summary['icp_fitness']:.4f}. Press D to retry."
                )
            else:
                self.status_text.setText(f"Attempt {self.attempt_count}: done. Press D to retry.")
        except Exception as exc:
            self.status_text.setText(f"Attempt {self.attempt_count}: failed: {exc}. Press D to retry.")
            print(f"[box_object] Attempt {self.attempt_count} failed: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False

    def run(self) -> None:
        self.base.run()


def run_batch_once(ctx: PipelineContext) -> dict:
    summary, _masks, _selected_mask = run_segmentation_and_bottle_icp_attempt(ctx)
    return summary

def main() -> None:
    args = parse_args()
    if args.capture_dir is not None:
        args.capture_dir = resolve_path(args.capture_dir)
    if args.ply is not None:
        args.ply = resolve_path(args.ply)
    if args.bottle_stl is not None:
        args.bottle_stl = resolve_path(args.bottle_stl)
    if args.bottle_template_ply is not None:
        args.bottle_template_ply = resolve_path(args.bottle_template_ply)
    if args.completion_bottle_template_ply is not None:
        args.completion_bottle_template_ply = resolve_path(args.completion_bottle_template_ply)

    ctx = prepare_pipeline_context(args)

    if args.show_viewer:
        app = InteractiveBottleIcpApp(ctx)
        app.run()
    else:
        run_batch_once(ctx)



if __name__ == "__main__":
    main()
    # pointcloud_path = "E:/py_project/wrsrobot/wrs_v2/yanjiuyuan/captures/20260704-170658/box_object_extraction/remaining_pointcloud.ply"
    # show_pointcloud(Path(pointcloud_path))