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
from yanjiuyuan.mech_eye_ur7e_pointcloud_env import (  # noqa: E402
    CAM_TO_WORLD,
    load_ply_pointcloud,
    save_numpy_pointcloud,
    transform_points,
)


@dataclass
class CaptureData:
    pcd_world: object
    points_world: np.ndarray
    colors: Optional[np.ndarray]
    target_ply: Path
    camera_ply: Optional[Path]
    rgb_path: Path
    capture_dir: Path
    frame: str


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
        transform = detect_box_transform_from_pointcloud(capture.pcd_world)
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


def run_or_load_point_hint_mask(
    args: argparse.Namespace,
    image_path: Path,
    output_dir: Path,
    auto_box: Optional[tuple[int, int, int, int]],
) -> tuple[np.ndarray, Optional[Path], Optional[dict]]:
    image_rgb = load_rgb_image_rgb(image_path)
    image_shape = image_rgb.shape[:2]
    if args.mask is not None:
        mask_path = resolve_path(args.mask)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask image not found: {mask_path}")
        return load_mask_image(mask_path, image_shape), mask_path, None

    import cv2
    from yanjiuyuan import point_hint_segment as phs

    segment_box = args.segment_box or (format_segment_box(auto_box) if args.auto_segment_box else None)
    segment_args = argparse.Namespace(
        image=image_path.resolve(),
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
    )
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image with cv2: {image_path}")

    h, w = image_bgr.shape[:2]
    points = phs.clamp_points([phs.parse_point(point) for point in segment_args.point], w, h)
    box = phs.clamp_box(phs.parse_box(segment_args.box), w, h) if segment_args.box else None
    log(f"[box_object] point_hint backend={segment_args.backend}, model={segment_args.model}")
    if box is not None:
        log(f"[box_object] point_hint box prompt: {box}")

    model = phs.load_model(segment_args.backend, segment_args.model)
    if segment_args.no_gui or points:
        if not points:
            raise ValueError("--no-gui requires at least one --point, or use --mask.")
        result = phs.run_model(
            model,
            segment_args.backend,
            segment_args.image,
            points,
            box,
            segment_args.imgsz,
            segment_args.conf,
            segment_args.iou,
            segment_args.device,
        )
        summary = phs.save_outputs(
            result,
            image_bgr,
            segment_args.image,
            segment_args.out_dir,
            points,
            box,
            segment_args.backend,
            segment_args.model,
            segment_args.keep,
        )
    else:
        session = phs.InteractiveSession(segment_args, image_bgr, model)
        summary = session.run()

    if not summary or not summary.get("detections"):
        raise RuntimeError("point_hint_segment.py did not save any mask. Add foreground points, press 's', then press 'q'.")
    mask_path = Path(summary["detections"][0]["mask_path"])
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
    colors = capture.colors
    kept_green = np.array([[0.0, 0.85, 0.15]], dtype=np.float64)
    removed_gray = np.array([[0.55, 0.55, 0.55]], dtype=np.float64)
    selected_red = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    candidate_points = points[masks.candidate]
    candidate_colors = np.tile(kept_green, (len(candidate_points), 1))
    display_candidate_mask = masks.candidate & ~masks.selected
    display_candidate_points = points[display_candidate_mask]
    display_candidate_colors = np.tile(kept_green, (len(display_candidate_points), 1))
    removed_points = points[masks.removed]
    removed_colors = np.tile(removed_gray, (len(removed_points), 1))
    selected_points = points[masks.selected]
    selected_colors = np.tile(selected_red, (len(selected_points), 1))

    candidate_points_out, candidate_colors_out = voxel_downsample_arrays(candidate_points, candidate_colors, args.candidate_voxel)
    display_candidate_points_out, display_candidate_colors_out = voxel_downsample_arrays(display_candidate_points, display_candidate_colors, args.candidate_voxel)
    removed_points_out, removed_colors_out = voxel_downsample_arrays(removed_points, removed_colors, args.removed_voxel)
    selected_points_out, selected_colors_out = voxel_downsample_arrays(selected_points, selected_colors, args.selected_voxel)
    candidate_colors_for_combined = display_candidate_colors_out

    candidate_path = output_dir / "box_object_candidate_points.ply"
    removed_path = output_dir / "box_object_removed_gray_points.ply"
    selected_path = output_dir / "box_selected_object_red_points.ply"
    combined_path = output_dir / "box_object_selection_colored_points.ply"
    transform_out_path = output_dir / "box_transform_used.txt"
    summary_path = output_dir / "box_object_extraction_summary.json"
    concave_region_path = output_dir / "box_concave_region_wireframe.ply"
    concave_corners_path = output_dir / "box_concave_region_corners_world.txt"
    box_mesh_path = output_dir / "box_model_registered_light_blue.ply"

    save_numpy_pointcloud(candidate_points_out, candidate_colors_out, candidate_path)
    save_numpy_pointcloud(removed_points_out, removed_colors_out, removed_path)
    save_numpy_pointcloud(selected_points_out, selected_colors_out, selected_path)
    combined_points = np.vstack([removed_points_out, display_candidate_points_out, selected_points_out])
    combined_colors = np.vstack([removed_colors_out, candidate_colors_for_combined, selected_colors_out])
    save_numpy_pointcloud(combined_points, combined_colors, combined_path)
    np.savetxt(transform_out_path, box_transform, fmt="%.9f")
    concave_line_set, concave_corners_world = make_concave_region_lineset(bounds, box_transform)
    import open3d as o3d
    o3d.io.write_line_set(str(concave_region_path), concave_line_set, write_ascii=False)
    np.savetxt(concave_corners_path, concave_corners_world, fmt="%.9f")
    box_mesh = make_transformed_box_mesh(box_transform) if args.show_box_model else None
    if box_mesh is not None:
        o3d.io.write_triangle_mesh(str(box_mesh_path), box_mesh, write_ascii=False)

    summary = {
        "capture_dir": str(capture.capture_dir),
        "target_ply": str(capture.target_ply),
        "camera_ply": None if capture.camera_ply is None else str(capture.camera_ply),
        "rgb_path": str(capture.rgb_path),
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
        "candidate_path": str(candidate_path),
        "removed_gray_path": str(removed_path),
        "selected_red_path": str(selected_path),
        "combined_colored_path": str(combined_path),
        "concave_region_wireframe_path": str(concave_region_path),
        "concave_region_corners_world_path": str(concave_corners_path),
        "box_model_registered_path": str(box_mesh_path) if box_mesh is not None else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def show_pointclouds(summary: dict) -> None:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(summary["combined_colored_path"])
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
    log("[box_object] Viewer colors: red=selected object, green=kept candidate points, gray=removed box/wall points, cyan wireframe=concave region.")
    o3d.visualization.draw_geometries(geometries, window_name="box object point cloud", width=1280, height=720)


def main() -> None:
    args = parse_args()
    if args.capture_dir is not None:
        args.capture_dir = resolve_path(args.capture_dir)
    if args.ply is not None:
        args.ply = resolve_path(args.ply)

    with timed_step("resolve/load saved capture"):
        capture = resolve_capture(args)

    output_dir = resolve_path(args.output_dir) if args.output_dir is not None else default_output_dir(capture.capture_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[box_object] output_dir: {output_dir}")

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

    with timed_step("load RGB and map point colors back to image pixels"):
        image_rgb = load_rgb_image_rgb(capture.rgb_path)
        colors_u8 = colors_to_u8(capture.colors)
        if colors_u8 is None:
            raise RuntimeError("Point cloud has no colors; cannot use point_hint_segment.py mask.")
        pixel_indices = build_pixel_to_point_indices_by_color_sequence(image_rgb, colors_u8, args.pixel_color_tolerance)
        mapped_points = pixel_indices >= 0
        mapped_ratio = float(mapped_points.sum()) / max(1, len(pixel_indices))
        log(f"[box_object] pixel mapping: {int(mapped_points.sum())}/{len(pixel_indices)} ({mapped_ratio * 100.0:.1f}%)")
        if mapped_ratio < args.pixel_mapping_min_ratio:
            raise RuntimeError(
                "Too few points could be mapped to RGB pixels. "
                f"ratio={mapped_ratio:.3f}, required={args.pixel_mapping_min_ratio:.3f}."
            )
        auto_box = bbox_from_pixel_indices(pixel_indices, candidate_mask & mapped_points, image_rgb.shape[1], image_rgb.shape[0])
        if auto_box is not None:
            if args.auto_segment_box:
                log(f"[box_object] auto 2D box prompt enabled: {auto_box}")
            else:
                log(f"[box_object] auto 2D candidate bbox computed but not passed to SAM: {auto_box}")

    with timed_step("run/load point_hint_segment mask"):
        mask_image, mask_path, point_hint_summary = run_or_load_point_hint_mask(args, capture.rgb_path, output_dir, auto_box)

    with timed_step("project 2D mask to 3D and select one object"):
        selected_mask, mask_projected = apply_2d_mask_to_points(mask_image, pixel_indices, candidate_mask)
        if not np.any(selected_mask):
            raise RuntimeError("No selected object points after applying the 2D mask and box candidate constraints.")
        masks = ExtractionMasks(
            box_region=box_region_mask,
            candidate=candidate_mask,
            removed=removed_mask,
            selected=selected_mask,
            mask_projected=mask_projected,
            mapped_points=mapped_points,
        )

    with timed_step("write extracted point clouds and summary"):
        summary = write_extraction_outputs(
            output_dir=output_dir,
            capture=capture,
            box_transform=box_transform,
            box_transform_path=box_transform_path,
            detected_box_now=detected_box_now,
            bounds=bounds,
            masks=masks,
            mask_path=mask_path,
            point_hint_summary=point_hint_summary,
            args=args,
        )

    print(f"Kept candidate points (green): {summary['candidate_count']}")
    print(f"Removed/context points (gray): {summary['removed_count']}")
    print(f"Selected object points (red): {summary['selected_count']}")
    print(f"Saved selected red PLY: {summary['selected_red_path']}")
    print(f"Saved removed gray PLY: {summary['removed_gray_path']}")
    print(f"Saved combined colored PLY: {summary['combined_colored_path']}")
    print(f"Saved concave region wireframe: {summary['concave_region_wireframe_path']}")
    if summary.get("box_model_registered_path"):
        print(f"Saved registered box model: {summary['box_model_registered_path']}")
    print(f"Saved summary: {summary['summary_path']}")

    if args.show_viewer:
        with timed_step("open Open3D viewer"):
            show_pointclouds(summary)


if __name__ == "__main__":
    main()
