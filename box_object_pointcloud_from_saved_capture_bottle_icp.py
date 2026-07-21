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
    parser.add_argument("--point-size", type=float, default=BOX_OBJECT_POINT_SIZE)
    return parser.parse_args()

def resolve_path(path: Path) -> Path:
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
    return resolve_path(Path(BOX_OBJECT_OUTPUT_DIR)) if BOX_OBJECT_OUTPUT_DIR is not None else capture_dir / "box_object_extraction"


def candidate_transform_paths(capture_dir: Path) -> list[Path]:
    return [
        capture_dir / "detected_box_transform.txt",
        capture_dir / "box_icp" / "box_obb_icp_transform.txt",
        capture_dir / "box_icp" / "box_icp_transform.txt",
    ]


def load_box_transform(args: argparse.Namespace, capture: CaptureData, output_dir: Path) -> tuple[np.ndarray, Path, bool]:
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
    target_pcd = box_icp.crop_pointcloud_by_range(world_pcd)
    target_pcd, _ = box_icp.segment_blue_box_points(target_pcd)
    target_pcd = box_icp.voxel_downsample_target(target_pcd)
    target_pcd, _ = box_icp.keep_largest_cluster(target_pcd)
    target_pcd = box_icp.remove_target_outliers(target_pcd)
    target_pcd = box_icp.filter_pointcloud_by_z_range(
        target_pcd, box_icp.MATCH_TARGET_Z_MIN, box_icp.MATCH_TARGET_Z_MAX, "Box target point cloud"
    )
    source_pcd, _ = box_icp.load_box_template_pointcloud()
    _, _, icp_result, *_ = box_icp.run_obb_initialized_icp(source_pcd, target_pcd, voxel_size=box_icp.VOXEL_SIZE)
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
    flat_mask = mask_image.reshape(-1)
    mapped_points = pixel_indices >= 0
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
    if not combined_path:
        log("[box_object] No saved point-cloud outputs to show.")
        return
    pcd = o3d.io.read_point_cloud(combined_path)
    geometries = [pcd]
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
    log("[box_object] Viewer colors: red=selected object, green=kept candidate points, gray=removed/context points, cyan wireframe=concave region, purple=global registered template points, yellow mesh=registered bottle.STL.")
    o3d.visualization.draw_geometries(geometries, window_name="box object point cloud + bottle ICP", width=1280, height=720)

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
        if capture.pixel_indices is None:
            raise RuntimeError("CaptureData.pixel_indices is required for fast SAM mask projection.")
        pixel_indices = np.asarray(capture.pixel_indices, dtype=np.int64).reshape(-1)
        if len(pixel_indices) != len(capture.points_world):
            raise RuntimeError(
                "CaptureData.pixel_indices length does not match point cloud length: "
                f"{len(pixel_indices)} != {len(capture.points_world)}."
            )
        pixel_count = int(image_rgb.shape[0] * image_rgb.shape[1])
        mapped_points = (pixel_indices >= 0) & (pixel_indices < pixel_count)
        mapped_ratio = float(mapped_points.sum()) / max(1, len(pixel_indices))
        log(f"[box_object] direct pixel indices: {int(mapped_points.sum())}/{len(pixel_indices)} ({mapped_ratio * 100.0:.1f}%)")
        if mapped_ratio < args.pixel_mapping_min_ratio:
            raise RuntimeError(
                "Too few captured point pixel indices are valid. "
                f"ratio={mapped_ratio:.3f}, required={args.pixel_mapping_min_ratio:.3f}."
            )
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
    attempt_args = copy.copy(ctx.args)
    total_start = perf_counter()
    segmentation_start = perf_counter()
    bottle_config = bottle_icp_config_from_runtime_options(attempt_args)
    with timed_step("run/load point_hint_segment mask"):
        mask_image, mask_path, point_hint_summary = run_or_load_point_hint_mask(
            attempt_args,
            ctx.capture.rgb_path,
            ctx.output_dir,
            ctx.auto_box,
            image_bgr=ctx.capture.rgb_image_bgr,
        )

    with timed_step("project 2D mask to 3D and select one object"):
        selected_mask, mask_projected = apply_2d_mask_to_points(mask_image, ctx.pixel_indices, ctx.candidate_mask)
        if not np.any(selected_mask):
            raise RuntimeError("No selected object points after applying the 2D mask and box candidate constraints.")
        masks = ExtractionMasks(
            box_region=ctx.box_region_mask,
            candidate=ctx.candidate_mask,
            removed=ctx.removed_mask,
            selected=selected_mask,
            mask_projected=mask_projected,
            mapped_points=ctx.mapped_points,
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

    if bottle_config.enabled and bottle_config.template == "prompt":
        with timed_step("choose bottle template after object segmentation"):
            bottle_config.template = choose_bottle_template_with_prompt(ctx.output_dir, bottle_config)

    matching_elapsed_s = None
    if bottle_config.enabled:
        with timed_step("register bottle model to selected object"):
            matching_start = perf_counter()
            bottle_summary = bottle_pose.run_bottle_registration_on_selected(ctx.capture.points_world[selected_mask], ctx.output_dir, bottle_config)
            summary["bottle_icp"] = bottle_summary
            matching_elapsed_s = perf_counter() - matching_start
            Path(summary["summary_path"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    total_elapsed_s = perf_counter() - total_start
    summary["timing"] = {
        "segmentation_elapsed_s": segmentation_elapsed_s,
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


def build_bottle_mesh_model(bottle_stl: Path, transform: np.ndarray):
    import wrs.modeling.geometric_model as mgm

    bottle_model = mgm.GeometricModel(
        initor=str(resolve_path(Path(bottle_stl))),
        name="bottle_icp_registered",
        toggle_twosided=True,
        rgb=np.array([1.0, 0.76, 0.18]),
        alpha=0.55,
    )
    bottle_model.pos = np.asarray(transform[:3, 3], dtype=np.float64)
    bottle_model.rotmat = np.asarray(transform[:3, :3], dtype=np.float64)
    return bottle_model


class InteractiveBottleIcpApp:
    def __init__(self, ctx: PipelineContext):
        import wrs.modeling.geometric_model as mgm
        import wrs.visualization.panda.world as wd
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        self.ctx = ctx
        self.running = False
        self.attempt_count = 0
        self.result_models = []
        self.mgm = mgm

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
        self.status_text = OnscreenText(
            text="Press D to segment + choose template + run bottle ICP. Press D again to retry.",
            pos=(-1.28, 0.92),
            align=TextNode.ALeft,
            scale=0.045,
            fg=(0.02, 0.02, 0.02, 1.0),
            mayChange=True,
        )
        self.base.accept("d", self.run_attempt)
        print("Viewer colors: green=kept candidate points, gray=removed/context points, red=selected object, purple=global registered template points, yellow=registered bottle mesh.")
        print("WRS viewer is ready. Press D in the viewer to run segmentation + template selection + bottle ICP.")

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

    def run_attempt(self) -> None:
        if self.running:
            print("[box_object] Detection is already running; ignoring D key.")
            return
        self.running = True
        self.attempt_count += 1
        self.status_text.setText(f"Attempt {self.attempt_count}: running segmentation/template/ICP...")
        self.clear_result_models()
        try:
            summary, _masks, selected_mask = run_segmentation_and_bottle_icp_attempt(self.ctx)
            self.attach_result(summary, selected_mask)
            if summary.get("bottle_icp"):
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

    ctx = prepare_pipeline_context(args)

    if args.show_viewer:
        app = InteractiveBottleIcpApp(ctx)
        app.run()
    else:
        run_batch_once(ctx)


if __name__ == "__main__":
    main()
