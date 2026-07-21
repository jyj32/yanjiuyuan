"""
Detect the blue box/bin from a saved Mech-Eye colored point cloud and register
the pre-sampled box top-surface template with constrained-OBB initialization + point-to-plane ICP.

Run:
    python yanjiuyuan/box_icp_from_saved_capture.py

Adjust shared parameters in yanjiuyuan/constants.py instead of passing command-line parameters.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import sys
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yanjiuyuan.constants import (  # noqa: E402
    BOX_BLUE_CLUSTER_EPS,
    BOX_BLUE_CLUSTER_MIN_POINTS,
    BOX_BLUE_DOMINANCE_MARGIN,
    BOX_BLUE_HUE_RANGE_DEG,
    BOX_BLUE_MIN_SATURATION,
    BOX_BLUE_MIN_VALUE,
    BOX_CAPTURE_ROOT,
    BOX_FINAL_MODEL_VOXEL_DOWNSAMPLE,
    BOX_FINAL_TARGET_VOXEL_DOWNSAMPLE,
    BOX_WRITE_DEBUG_POINTCLOUDS,
    BOX_ICP_MAX_ITERATION,
    BOX_KEEP_LARGEST_BLUE_CLUSTER,
    BOX_LOG_STEPS,
    BOX_MATCH_TARGET_Z_MIN,
    BOX_MATCH_TARGET_Z_MAX,
    BOX_MODEL_POINT_SIZE,
    BOX_OPEN3D_PRINT_PROGRESS,
    BOX_OBB_HORIZONTAL_AXIS_MIN_NORM,
    BOX_OBB_LONG_SHORT_MIN_RATIO,
    BOX_OBB_REQUIRE_LONG_SHORT_RATIO,
    BOX_OUTLIER_NB_NEIGHBORS,
    BOX_OUTLIER_STD_RATIO,
    BOX_OUTPUT_DIR,
    BOX_REGISTRATION_VOXEL_SIZE,
    BOX_REMOVE_STATISTICAL_OUTLIERS,
    BOX_SHOW_RESULT_VIEWER,
    BOX_TARGET_POINT_SIZE,
    BOX_TARGET_VOXEL_DOWNSAMPLE,
    BOX_TEMPLATE_NORMAL_Z_ABS_MIN,
    BOX_TEMPLATE_PLY as DEFAULT_BOX_TEMPLATE_PLY,
    BOX_TEMPLATE_TOP_Z_MIN,
    BOX_USE_BLUE_RGB_SEGMENTATION,
    BOX_X_RANGE,
    BOX_Y_RANGE,
    BOX_Z_RANGE,
)
from yanjiuyuan.mech_eye_ur7e_pointcloud_env import (  # noqa: E402
    CAM_TO_WORLD,
    load_ply_pointcloud,
)


BOX_TEMPLATE_PLY = DEFAULT_BOX_TEMPLATE_PLY
CAPTURE_ROOT = BOX_CAPTURE_ROOT

# If CAPTURE_DIR is None, the newest folder under CAPTURE_ROOT is used.
# You may also set CAPTURE_PLY directly to a specific .ply file.
CAPTURE_DIR = None
CAPTURE_PLY = None
PREFER_WORLD_FRAME_PLY = True
TRANSFORM_CAMERA_PLY_TO_WORLD = True

# RGB segmentation for the blue box. Open3D stores colors in [0, 1].
USE_BLUE_RGB_SEGMENTATION = BOX_USE_BLUE_RGB_SEGMENTATION
BLUE_HUE_RANGE_DEG = BOX_BLUE_HUE_RANGE_DEG
BLUE_MIN_SATURATION = BOX_BLUE_MIN_SATURATION
BLUE_MIN_VALUE = BOX_BLUE_MIN_VALUE
BLUE_DOMINANCE_MARGIN = BOX_BLUE_DOMINANCE_MARGIN

# Extra target crop before registration. Z_RANGE=(0.03, None) removes the table.
X_RANGE = BOX_X_RANGE
Y_RANGE = BOX_Y_RANGE
Z_RANGE = BOX_Z_RANGE

# Clean up the segmented target.
TARGET_VOXEL_DOWNSAMPLE = BOX_TARGET_VOXEL_DOWNSAMPLE
MATCH_TARGET_Z_MIN = BOX_MATCH_TARGET_Z_MIN
MATCH_TARGET_Z_MAX = BOX_MATCH_TARGET_Z_MAX
KEEP_LARGEST_BLUE_CLUSTER = BOX_KEEP_LARGEST_BLUE_CLUSTER
BLUE_CLUSTER_EPS = BOX_BLUE_CLUSTER_EPS
BLUE_CLUSTER_MIN_POINTS = BOX_BLUE_CLUSTER_MIN_POINTS
REMOVE_STATISTICAL_OUTLIERS = BOX_REMOVE_STATISTICAL_OUTLIERS
OUTLIER_NB_NEIGHBORS = BOX_OUTLIER_NB_NEIGHBORS
OUTLIER_STD_RATIO = BOX_OUTLIER_STD_RATIO

# Registration settings.
VOXEL_SIZE = BOX_REGISTRATION_VOXEL_SIZE
MODEL_TOP_Z_MIN = BOX_TEMPLATE_TOP_Z_MIN
MODEL_TOP_NORMAL_Z_ABS_MIN = BOX_TEMPLATE_NORMAL_Z_ABS_MIN
OBB_LONG_SHORT_MIN_RATIO = BOX_OBB_LONG_SHORT_MIN_RATIO
OBB_REQUIRE_LONG_SHORT_RATIO = BOX_OBB_REQUIRE_LONG_SHORT_RATIO
OBB_HORIZONTAL_AXIS_MIN_NORM = BOX_OBB_HORIZONTAL_AXIS_MIN_NORM
ICP_MAX_ITERATION = BOX_ICP_MAX_ITERATION

# Output and optional visualization.
OUTPUT_DIR = BOX_OUTPUT_DIR
SHOW_RESULT_VIEWER = BOX_SHOW_RESULT_VIEWER
TARGET_POINT_SIZE = BOX_TARGET_POINT_SIZE
MODEL_POINT_SIZE = BOX_MODEL_POINT_SIZE
FINAL_TARGET_VOXEL_DOWNSAMPLE = BOX_FINAL_TARGET_VOXEL_DOWNSAMPLE
FINAL_MODEL_VOXEL_DOWNSAMPLE = BOX_FINAL_MODEL_VOXEL_DOWNSAMPLE
WRITE_DEBUG_POINTCLOUDS = BOX_WRITE_DEBUG_POINTCLOUDS

# Progress output.
LOG_STEPS = BOX_LOG_STEPS
OPEN3D_PRINT_PROGRESS = BOX_OPEN3D_PRINT_PROGRESS


Range3D = Optional[Tuple[Optional[float], Optional[float]]]


def log(message: str) -> None:
    if LOG_STEPS:
        print(message, flush=True)


@contextmanager
def timed_step(name: str):
    start_time = perf_counter()
    log(f"[box_icp] START {name}")
    try:
        yield
    except Exception:
        elapsed = perf_counter() - start_time
        log(f"[box_icp] FAILED {name} after {elapsed:.2f}s")
        raise
    elapsed = perf_counter() - start_time
    log(f"[box_icp] DONE {name} in {elapsed:.2f}s")


def describe_pointcloud(label: str, pcd) -> None:
    points = np.asarray(pcd.points)
    if len(points) == 0:
        log(f"[box_icp] {label}: 0 points")
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    log(
        "[box_icp] "
        f"{label}: {len(points)} points, "
        f"x=[{mins[0]:.4f}, {maxs[0]:.4f}], "
        f"y=[{mins[1]:.4f}, {maxs[1]:.4f}], "
        f"z=[{mins[2]:.4f}, {maxs[2]:.4f}]"
    )


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def transform_open3d_pointcloud(pcd, homomat: np.ndarray):
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(homomat)
    return pcd_copy


def find_latest_capture_dir(capture_root: Path = CAPTURE_ROOT) -> Path:
    capture_root = resolve_path(capture_root)
    if not capture_root.exists():
        raise FileNotFoundError(f"Capture root does not exist: {capture_root}")

    candidates = []
    for path in capture_root.iterdir():
        if not path.is_dir():
            continue
        if (path / "world_colored_pointcloud.ply").exists() or (path / "colored_pointcloud.ply").exists():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No saved captures found under: {capture_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_capture_ply(
    capture_dir: Optional[Path] = CAPTURE_DIR,
    capture_ply: Optional[Path] = CAPTURE_PLY,
    prefer_world_frame: bool = PREFER_WORLD_FRAME_PLY,
) -> Tuple[Path, Path, str]:
    if capture_ply is not None:
        ply_path = resolve_path(capture_ply)
        if not ply_path.exists():
            raise FileNotFoundError(f"Capture PLY does not exist: {ply_path}")
        frame = "world" if ply_path.name == "world_colored_pointcloud.ply" else "camera"
        return ply_path, ply_path.parent, frame

    if capture_dir is None:
        capture_dir = find_latest_capture_dir(CAPTURE_ROOT)
    else:
        capture_dir = resolve_path(capture_dir)

    world_ply = capture_dir / "world_colored_pointcloud.ply"
    camera_ply = capture_dir / "colored_pointcloud.ply"
    if prefer_world_frame and world_ply.exists():
        return world_ply, capture_dir, "world"
    if camera_ply.exists():
        return camera_ply, capture_dir, "camera"
    if world_ply.exists():
        return world_ply, capture_dir, "world"
    raise FileNotFoundError(f"No colored point cloud PLY found in: {capture_dir}")


def load_saved_capture_pointcloud():
    ply_path, capture_dir, frame = resolve_capture_ply()
    pcd = load_ply_pointcloud(ply_path)
    if frame == "camera" and TRANSFORM_CAMERA_PLY_TO_WORLD:
        pcd = transform_open3d_pointcloud(pcd, CAM_TO_WORLD)
        frame = "world_from_camera"
    return pcd, ply_path, capture_dir, frame


def rgb_to_hsv_degrees(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.clip(rgb, 0.0, 1.0)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    delta = maxc - minc

    hue = np.zeros(len(rgb), dtype=np.float64)
    nonzero = delta > 1e-9
    rmax = nonzero & (maxc == r)
    gmax = nonzero & (maxc == g)
    bmax = nonzero & (maxc == b)
    hue[rmax] = (60.0 * ((g[rmax] - b[rmax]) / delta[rmax]) + 360.0) % 360.0
    hue[gmax] = 60.0 * ((b[gmax] - r[gmax]) / delta[gmax] + 2.0)
    hue[bmax] = 60.0 * ((r[bmax] - g[bmax]) / delta[bmax] + 4.0)

    saturation = np.zeros(len(rgb), dtype=np.float64)
    valid_value = maxc > 1e-9
    saturation[valid_value] = delta[valid_value] / maxc[valid_value]
    value = maxc
    return hue, saturation, value


def hue_range_mask(hue: np.ndarray, hue_range: Tuple[float, float]) -> np.ndarray:
    blue_low, blue_high = hue_range
    if blue_low <= blue_high:
        return (hue >= blue_low) & (hue <= blue_high)
    return (hue >= blue_low) | (hue <= blue_high)


def blue_color_mask_components(colors: np.ndarray):
    hue, saturation, value = rgb_to_hsv_degrees(colors)
    hue_mask = hue_range_mask(hue, BLUE_HUE_RANGE_DEG)
    saturation_mask = saturation >= BLUE_MIN_SATURATION
    value_mask = value >= BLUE_MIN_VALUE

    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    dominance_mask = (b >= r + BLUE_DOMINANCE_MARGIN) & (b >= g + BLUE_DOMINANCE_MARGIN)
    final_mask = hue_mask & saturation_mask & value_mask & dominance_mask
    components = {
        "hue": hue_mask,
        "saturation": saturation_mask,
        "value": value_mask,
        "dominance": dominance_mask,
        "hue_saturation_value": hue_mask & saturation_mask & value_mask,
        "final": final_mask,
    }
    return final_mask, components, (hue, saturation, value)


def format_count(count: int, total: int) -> str:
    if total <= 0:
        return "0 (0.0%)"
    return f"{count} ({count / total * 100.0:.1f}%)"


def format_percentiles(values: np.ndarray) -> str:
    if len(values) == 0:
        return "[]"
    return np.array2string(np.percentile(values, [5, 50, 95]), precision=3)


def log_blue_mask_diagnostics(colors: np.ndarray, mask: np.ndarray, components: dict, hsv_values) -> None:
    total = len(colors)
    hue, saturation, value = hsv_values
    log(
        "[box_icp] blue thresholds: "
        f"hue={BLUE_HUE_RANGE_DEG}, sat>={BLUE_MIN_SATURATION}, "
        f"value>={BLUE_MIN_VALUE}, dominance_margin={BLUE_DOMINANCE_MARGIN}"
    )
    log(
        "[box_icp] blue mask counts: "
        f"hue={format_count(int(components['hue'].sum()), total)}, "
        f"sat={format_count(int(components['saturation'].sum()), total)}, "
        f"value={format_count(int(components['value'].sum()), total)}, "
        f"dominance={format_count(int(components['dominance'].sum()), total)}, "
        f"hue_sat_value={format_count(int(components['hue_saturation_value'].sum()), total)}, "
        f"final={format_count(int(mask.sum()), total)}"
    )
    if np.any(mask):
        log(
            "[box_icp] blue kept HSV percentiles p05/p50/p95: "
            f"hue={format_percentiles(hue[mask])}, "
            f"sat={format_percentiles(saturation[mask])}, "
            f"value={format_percentiles(value[mask])}"
        )


def blue_color_mask(colors: np.ndarray) -> np.ndarray:
    mask, _, _ = blue_color_mask_components(colors)
    return mask


def crop_pointcloud_by_range(
    pcd,
    x_range: Range3D = X_RANGE,
    y_range: Range3D = Y_RANGE,
    z_range: Range3D = Z_RANGE,
):
    # 通过xyz范围裁剪点云
    if x_range is None and y_range is None and z_range is None:
        return pcd

    points = np.asarray(pcd.points)
    mask = np.ones(len(points), dtype=bool)
    for axis, axis_range in enumerate((x_range, y_range, z_range)):
        if axis_range is None:
            continue
        low, high = axis_range
        if low is not None:
            mask &= points[:, axis] >= low
        if high is not None:
            mask &= points[:, axis] <= high

    indices = np.flatnonzero(mask).tolist()
    if not indices:
        raise RuntimeError("Target point cloud is empty after XYZ range filtering.")
    return pcd.select_by_index(indices)


def segment_blue_box_points(pcd):
    # 分割出蓝箱的点云
    if not USE_BLUE_RGB_SEGMENTATION:   # 不分割蓝箱点云
        return pcd, len(pcd.points)
    if not pcd.has_colors():    # 点云图无颜色
        raise RuntimeError("The saved point cloud has no RGB colors; blue box segmentation needs colored PLY data.")

    colors = np.asarray(pcd.colors)
    mask, components, hsv_values = blue_color_mask_components(colors)
    log_blue_mask_diagnostics(colors, mask, components, hsv_values)
    indices = np.flatnonzero(mask).tolist()
    if not indices:
        raise RuntimeError("No blue box points found. Loosen BOX_BLUE_* thresholds in yanjiuyuan/constants.py.")
    return pcd.select_by_index(indices), len(indices)


def keep_largest_cluster(pcd):
    # DBSCAN 聚类，提取出点数最多的簇（即最大的连通密集区域），并剔除其他较小的簇或离群噪点
    if not KEEP_LARGEST_BLUE_CLUSTER:
        return pcd, -1
    if len(pcd.points) < BLUE_CLUSTER_MIN_POINTS:
        return pcd, -1

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=BLUE_CLUSTER_EPS,
            min_points=BLUE_CLUSTER_MIN_POINTS,
            print_progress=OPEN3D_PRINT_PROGRESS,
        )
    )
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return pcd, -1

    cluster_ids, counts = np.unique(valid_labels, return_counts=True)
    largest_cluster_id = int(cluster_ids[np.argmax(counts)])
    indices = np.flatnonzero(labels == largest_cluster_id).tolist()
    return pcd.select_by_index(indices), largest_cluster_id

def voxel_downsample_pointcloud(pcd, voxel_size: Optional[float]):
    if voxel_size is not None and voxel_size > 0:
        return pcd.voxel_down_sample(voxel_size)
    return pcd


def voxel_downsample_target(pcd):

    return voxel_downsample_pointcloud(pcd, TARGET_VOXEL_DOWNSAMPLE)


def filter_pointcloud_by_z_min(pcd, z_min: Optional[float], label: str):
    if z_min is None:
        return pcd

    points = np.asarray(pcd.points)
    indices = np.flatnonzero(points[:, 2] > z_min).tolist()
    if not indices:
        raise RuntimeError(f"{label} is empty after z > {z_min} filtering.")
    return pcd.select_by_index(indices)


def filter_pointcloud_by_z_range(
    pcd,
    z_min: Optional[float],
    z_max: Optional[float],
    label: str,
):
    # 通过z范围裁剪点云
    if z_min is None and z_max is None:
        return pcd

    points = np.asarray(pcd.points)
    mask = np.ones(len(points), dtype=bool)
    range_parts = []
    if z_min is not None:
        mask &= points[:, 2] > z_min
        range_parts.append(f"z > {z_min}")
    if z_max is not None:
        mask &= points[:, 2] < z_max
        range_parts.append(f"z < {z_max}")

    indices = np.flatnonzero(mask).tolist()
    if not indices:
        raise RuntimeError(f"{label} is empty after {' and '.join(range_parts)} filtering.")
    return pcd.select_by_index(indices)


def remove_target_outliers(pcd):
    # 对目标点云进行统计离群点（噪声点）滤波
    if REMOVE_STATISTICAL_OUTLIERS and len(pcd.points) > OUTLIER_NB_NEIGHBORS:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS,
            std_ratio=OUTLIER_STD_RATIO,
        )
    if len(pcd.points) == 0:
        raise RuntimeError("Target point cloud is empty after cleanup.")
    return pcd


def filter_pointcloud_by_normal_z_abs_min(pcd, normal_z_abs_min: Optional[float], label: str):
    if normal_z_abs_min is None:
        return pcd
    if not pcd.has_normals():
        raise RuntimeError(
            f"{label} has no normals. Run yanjiuyuan/sample_box_surface.py with INCLUDE_NORMALS=True."
        )

    normals = np.asarray(pcd.normals)
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_normals = normal_lengths > 1e-9
    normal_z_abs = np.zeros(len(normals), dtype=np.float64)
    normal_z_abs[valid_normals] = np.abs(normals[valid_normals, 2] / normal_lengths[valid_normals])
    indices = np.flatnonzero(normal_z_abs >= normal_z_abs_min).tolist()
    if not indices:
        raise RuntimeError(f"{label} is empty after abs(normal_z) >= {normal_z_abs_min} filtering.")
    return pcd.select_by_index(indices)


def load_box_template_pointcloud(template_ply: Path = BOX_TEMPLATE_PLY):
    # 加载盒体模板点云，并对其进行滤波，仅保留顶部平面上的点
    template_ply = resolve_path(template_ply)
    if not template_ply.exists():
        raise FileNotFoundError(
            f"Box template PLY does not exist: {template_ply}\n"
            "Run `python yanjiuyuan/sample_box_surface.py` once to generate it."
        )

    pcd = load_ply_pointcloud(template_ply)
    loaded_count = len(pcd.points)
    pcd = filter_pointcloud_by_z_min(pcd, MODEL_TOP_Z_MIN, "Box template point cloud")
    pcd = filter_pointcloud_by_normal_z_abs_min(
        pcd,
        MODEL_TOP_NORMAL_Z_ABS_MIN,
        "Box template point cloud",
    )
    log(
        "[box_icp] loaded box template PLY: "
        f"kept {len(pcd.points)} / {loaded_count} points "
        f"with z > {MODEL_TOP_Z_MIN} and abs(normal_z) >= {MODEL_TOP_NORMAL_Z_ABS_MIN}"
    )
    return pcd, template_ply


def preprocess_point_cloud_for_icp(pcd, voxel_size: float = VOXEL_SIZE):
    import open3d as o3d

    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) == 0:
        raise RuntimeError(f"Point cloud is empty after ICP voxel downsample voxel={voxel_size}.")

    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    return pcd_down


def normalize_vector(vector: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm <= 1e-9:
        raise RuntimeError(f"Cannot normalize near-zero vector for {label}.")
    return vector / norm


def canonicalize_xy_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).copy()
    axis[2] = 0.0
    axis = normalize_vector(axis, "OBB horizontal axis")
    if axis[0] < -1e-9 or (abs(axis[0]) <= 1e-9 and axis[1] < 0.0):
        axis = -axis
    return axis


def compute_frame_center_and_extent(points: np.ndarray, axes: np.ndarray):
    coords = points @ axes
    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)
    center = axes @ ((coord_min + coord_max) / 2.0)
    extent = coord_max - coord_min
    return center, extent


def build_constrained_obb_frame(points: np.ndarray, long_axis: np.ndarray, label: str, method: str):
    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    long_axis = canonicalize_xy_axis(long_axis)
    short_axis = normalize_vector(np.cross(world_z, long_axis), f"{label} short axis")
    axes = np.column_stack((long_axis, short_axis, world_z))
    center, extent = compute_frame_center_and_extent(points, axes)

    if extent[1] > extent[0]:
        long_axis, short_axis = short_axis, -long_axis
        long_axis = canonicalize_xy_axis(long_axis)
        short_axis = normalize_vector(np.cross(world_z, long_axis), f"{label} swapped short axis")
        axes = np.column_stack((long_axis, short_axis, world_z))
        center, extent = compute_frame_center_and_extent(points, axes)

    long_short_ratio = extent[0] / max(extent[1], 1e-9)
    if long_short_ratio < OBB_LONG_SHORT_MIN_RATIO:
        message = (
            f"[box_icp] {label} OBB long/short ratio {long_short_ratio:.3f} "
            f"is below {OBB_LONG_SHORT_MIN_RATIO:.3f}; extent={extent}"
        )
        if OBB_REQUIRE_LONG_SHORT_RATIO:
            raise RuntimeError(message)
        log("Warning: " + message)

    log(
        f"[box_icp] {label} constrained OBB ({method}): "
        f"center={np.array2string(center, precision=5)}, "
        f"extent={np.array2string(extent, precision=5)}, "
        f"long_short_ratio={long_short_ratio:.3f}"
    )
    return {
        "center": center,
        "axes": axes,
        "extent": extent,
        "long_short_ratio": long_short_ratio,
        "method": method,
    }


def xy_pca_long_axis(points: np.ndarray) -> np.ndarray:
    xy = points[:, :2]
    centered_xy = xy - xy.mean(axis=0)
    if len(centered_xy) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    covariance = np.cov(centered_xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_xy = eigenvectors[:, int(np.argmax(eigenvalues))]
    return np.array([axis_xy[0], axis_xy[1], 0.0], dtype=np.float64)


def compute_constrained_obb_frame(pcd, label: str):
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) < 3:
        raise RuntimeError(f"{label} needs at least 3 points for OBB initialization.")

    # The source template is a single top plane, so a 3D convex hull based OBB
    # triggers Qhull lower-dimensional warnings. The box pose we need is
    # constrained to world Z anyway, so estimate the long side in XY directly.
    long_axis = xy_pca_long_axis(points)
    return build_constrained_obb_frame(points, long_axis, label, "xy_pca_z_up")


def estimate_obb_initial_transform(source_pcd, target_pcd):
    source_obb = compute_constrained_obb_frame(source_pcd, "source template")
    target_obb = compute_constrained_obb_frame(target_pcd, "target box")

    rotation = target_obb["axes"] @ source_obb["axes"].T
    translation = target_obb["center"] - rotation @ source_obb["center"]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    log("[box_icp] OBB initial transform:\n" + np.array2string(transform, precision=6))
    return transform, source_obb, target_obb


def run_point_to_plane_icp(source_down, target_down, init_transform: np.ndarray, voxel_size: float = VOXEL_SIZE):
    import open3d as o3d

    reg = o3d.pipelines.registration
    return reg.registration_icp(
        source_down,
        target_down,
        voxel_size * 0.8,
        init_transform,
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=ICP_MAX_ITERATION,
        ),
    )


def run_obb_initialized_icp(source_pcd, target_pcd, voxel_size: float = VOXEL_SIZE):
    #
    import open3d as o3d
    # 模板下采样
    reg = o3d.pipelines.registration
    with timed_step(f"preprocess source for ICP voxel={voxel_size}"):
        source_down = preprocess_point_cloud_for_icp(source_pcd, voxel_size)
    describe_pointcloud("source downsampled for ICP", source_down)
    # 目标（实际物体）下采样
    with timed_step(f"preprocess target for ICP voxel={voxel_size}"):
        target_down = preprocess_point_cloud_for_icp(target_pcd, voxel_size)
    describe_pointcloud("target downsampled for ICP", target_down)
    # OBB初始变换估计
    with timed_step("estimate constrained OBB initial transform"):
        obb_initial_transform, source_obb, target_obb = estimate_obb_initial_transform(source_down, target_down)
    # 评估初始变换
    distance_threshold = voxel_size * 0.8
    with timed_step(f"evaluate OBB initial transform threshold={distance_threshold}"):
        obb_initial_result = reg.evaluate_registration(
            source_down,
            target_down,
            distance_threshold,
            obb_initial_transform,
        )
    log(
        "[box_icp] OBB initial fitness/rmse: "
        f"{obb_initial_result.fitness:.6f} / {obb_initial_result.inlier_rmse:.6f}"
    )
    # 点对面 ICP 精配准
    with timed_step(f"point-to-plane ICP max_iteration={ICP_MAX_ITERATION}"):
        icp_result = run_point_to_plane_icp(
            source_down,
            target_down,
            obb_initial_transform,
            voxel_size=voxel_size,
        )
    log(f"[box_icp] ICP result fitness/rmse: {icp_result.fitness:.6f} / {icp_result.inlier_rmse:.6f}")
    return obb_initial_transform, obb_initial_result, icp_result, source_down, target_down, source_obb, target_obb


def write_registration_outputs(
    output_dir: Path,
    source_pcd,
    target_pcd,
    initial_source_pcd,
    refined_source_pcd,
    obb_initial_result,
    obb_initial_transform: np.ndarray,
    icp_result,
    target_ply: Path,
    target_frame: str,
    box_template_ply: Path,
    raw_target_count: int,
    segmented_target_count: int,
    downsampled_target_count: int,
    largest_cluster_id: int,
    largest_cluster_count: int,
    clean_target_count: int,
    source_count: int,
    source_obb: dict,
    target_obb: dict,
    debug_pointclouds: Optional[dict] = None,
):
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "box_template_surface_points.ply"
    target_path = output_dir / "box_blue_target_points.ply"
    initial_path = output_dir / "box_obb_initial_registered_points.ply"
    refined_path = output_dir / "box_icp_registered_points.ply"
    obb_initial_transform_path = output_dir / "box_obb_initial_transform.txt"
    icp_transform_path = output_dir / "box_obb_icp_transform.txt"
    summary_path = output_dir / "box_icp_summary.txt"
    debug_paths = {}

    o3d.io.write_point_cloud(str(source_path), source_pcd, write_ascii=False)
    o3d.io.write_point_cloud(str(target_path), target_pcd, write_ascii=False)
    o3d.io.write_point_cloud(str(initial_path), initial_source_pcd, write_ascii=False)
    o3d.io.write_point_cloud(str(refined_path), refined_source_pcd, write_ascii=False)
    if debug_pointclouds:
        for debug_name, debug_pcd in debug_pointclouds.items():
            debug_path = output_dir / f"{debug_name}.ply"
            o3d.io.write_point_cloud(str(debug_path), debug_pcd, write_ascii=False)
            debug_paths[debug_name] = debug_path
    np.savetxt(obb_initial_transform_path, obb_initial_transform, fmt="%.9f")
    np.savetxt(icp_transform_path, icp_result.transformation, fmt="%.9f")

    summary = [
        f"box_template_ply: {box_template_ply}",
        f"target_ply: {target_ply}",
        f"target_frame: {target_frame}",
        f"raw_target_points: {raw_target_count}",
        f"blue_segmented_points: {segmented_target_count}",
        f"downsampled_blue_target_points: {downsampled_target_count}",
        f"largest_cluster_id: {largest_cluster_id}",
        f"largest_cluster_points: {largest_cluster_count}",
        f"clean_target_points_before_final_downsample: {clean_target_count}",
        f"source_points_before_final_downsample: {source_count}",
        f"written_target_points: {len(target_pcd.points)}",
        f"written_source_points: {len(source_pcd.points)}",
        f"written_obb_initial_points: {len(initial_source_pcd.points)}",
        f"written_icp_points: {len(refined_source_pcd.points)}",
        f"blue_hue_range_deg: {BLUE_HUE_RANGE_DEG}",
        f"blue_min_saturation: {BLUE_MIN_SATURATION}",
        f"blue_min_value: {BLUE_MIN_VALUE}",
        f"blue_dominance_margin: {BLUE_DOMINANCE_MARGIN}",
        f"x_range: {X_RANGE}",
        f"y_range: {Y_RANGE}",
        f"z_range: {Z_RANGE}",
        f"match_target_z_min: {MATCH_TARGET_Z_MIN}",
        f"match_target_z_max: {MATCH_TARGET_Z_MAX}",
        f"keep_largest_blue_cluster: {KEEP_LARGEST_BLUE_CLUSTER}",
        f"blue_cluster_eps: {BLUE_CLUSTER_EPS}",
        f"blue_cluster_min_points: {BLUE_CLUSTER_MIN_POINTS}",
        f"target_voxel_downsample: {TARGET_VOXEL_DOWNSAMPLE}",
        f"final_target_voxel_downsample: {FINAL_TARGET_VOXEL_DOWNSAMPLE}",
        f"final_model_voxel_downsample: {FINAL_MODEL_VOXEL_DOWNSAMPLE}",
        f"write_debug_pointclouds: {WRITE_DEBUG_POINTCLOUDS}",
        f"voxel_size: {VOXEL_SIZE}",
        f"model_top_z_min: {MODEL_TOP_Z_MIN}",
        f"model_top_normal_z_abs_min: {MODEL_TOP_NORMAL_Z_ABS_MIN}",
        f"obb_long_short_min_ratio: {OBB_LONG_SHORT_MIN_RATIO}",
        f"obb_require_long_short_ratio: {OBB_REQUIRE_LONG_SHORT_RATIO}",
        f"obb_horizontal_axis_min_norm: {OBB_HORIZONTAL_AXIS_MIN_NORM}",
        f"source_obb_method: {source_obb['method']}",
        f"source_obb_center: {np.array2string(source_obb['center'], precision=9)}",
        f"source_obb_extent: {np.array2string(source_obb['extent'], precision=9)}",
        f"source_obb_long_short_ratio: {source_obb['long_short_ratio']:.9f}",
        f"target_obb_method: {target_obb['method']}",
        f"target_obb_center: {np.array2string(target_obb['center'], precision=9)}",
        f"target_obb_extent: {np.array2string(target_obb['extent'], precision=9)}",
        f"target_obb_long_short_ratio: {target_obb['long_short_ratio']:.9f}",
        f"obb_initial_fitness: {obb_initial_result.fitness:.9f}",
        f"obb_initial_inlier_rmse: {obb_initial_result.inlier_rmse:.9f}",
        f"icp_fitness: {icp_result.fitness:.9f}",
        f"icp_inlier_rmse: {icp_result.inlier_rmse:.9f}",
        f"obb_initial_transform: {obb_initial_transform_path}",
        f"icp_transform: {icp_transform_path}",
        f"icp_registered_points: {refined_path}",
    ]
    for debug_name, debug_path in debug_paths.items():
        summary.append(f"{debug_name}: {debug_path}")
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    return {
        "source_path": source_path,
        "target_path": target_path,
        "obb_initial_registered_path": initial_path,
        "icp_registered_path": refined_path,
        "obb_initial_transform_path": obb_initial_transform_path,
        "icp_transform_path": icp_transform_path,
        "summary_path": summary_path,
        "debug_paths": debug_paths,
    }



def show_registration_result(target_pcd, initial_source_pcd, refined_source_pcd):
    from wrs import mgm, wd

    target_points = np.asarray(target_pcd.points)
    if len(target_points) == 0:
        return
    center = (target_points.min(axis=0) + target_points.max(axis=0)) / 2.0
    extent = max(float(np.max(target_points.max(axis=0) - target_points.min(axis=0))), 1e-6)
    cam_dist = max(0.45, extent * 2.6)
    cam_pos = center + np.array([cam_dist, -cam_dist, max(cam_dist * 0.65, extent * 0.9)])

    base = wd.World(cam_pos=cam_pos, lookat_pos=center, w=1280, h=720)
    mgm.gen_frame(ax_length=max(extent * 0.35, 0.03), ax_radius=0.001).attach_to(base)
    mgm.gen_pointcloud(target_points, rgba=np.array([0.05, 0.45, 1.0, 0.6]), point_size=TARGET_POINT_SIZE).attach_to(base)
    mgm.gen_pointcloud(np.asarray(initial_source_pcd.points), rgba=np.array([1.0, 0.75, 0.0, 0.85]), point_size=MODEL_POINT_SIZE).attach_to(base)
    mgm.gen_pointcloud(np.asarray(refined_source_pcd.points), rgba=np.array([1.0, 0.05, 0.02, 1.0]), point_size=MODEL_POINT_SIZE).attach_to(base)
    print("Viewer colors: blue target=segmented box, yellow=OBB initial, red=ICP refined", flush=True)
    print("Close the viewer window to finish this script.", flush=True)
    base.run()


def main() -> None:
    debug_pointclouds = {}
    with timed_step("load saved capture point cloud"):
        target_pcd, target_ply, capture_dir, target_frame = load_saved_capture_pointcloud()
    raw_target_count = len(target_pcd.points)
    log(f"[box_icp] Loaded target: {target_ply} ({target_frame})")
    describe_pointcloud("raw target", target_pcd)

    with timed_step(f"crop target by XYZ ranges x={X_RANGE} y={Y_RANGE} z={Z_RANGE}"):
        target_pcd = crop_pointcloud_by_range(target_pcd)
    describe_pointcloud("cropped target", target_pcd)

    with timed_step("segment blue box points by RGB/HSV thresholds"):
        target_pcd, segmented_target_count = segment_blue_box_points(target_pcd)
    describe_pointcloud("blue segmented target", target_pcd)

    with timed_step(f"voxel downsample blue target voxel={TARGET_VOXEL_DOWNSAMPLE}"):
        target_pcd = voxel_downsample_target(target_pcd)
    if WRITE_DEBUG_POINTCLOUDS:
        debug_pointclouds["box_blue_downsampled_before_cluster"] = copy.deepcopy(target_pcd)
    downsampled_target_count = len(target_pcd.points)
    describe_pointcloud("downsampled blue target", target_pcd)

    with timed_step(
        f"keep largest blue cluster eps={BLUE_CLUSTER_EPS} min_points={BLUE_CLUSTER_MIN_POINTS}"
    ):
        target_pcd, largest_cluster_id = keep_largest_cluster(target_pcd)
    if WRITE_DEBUG_POINTCLOUDS:
        debug_pointclouds["box_blue_largest_cluster_before_outlier_z"] = copy.deepcopy(target_pcd)
    largest_cluster_count = len(target_pcd.points)
    describe_pointcloud("largest blue cluster", target_pcd)

    with timed_step(
        f"remove statistical outliers nb_neighbors={OUTLIER_NB_NEIGHBORS} std_ratio={OUTLIER_STD_RATIO}"
    ):
        target_pcd = remove_target_outliers(target_pcd)
    if WRITE_DEBUG_POINTCLOUDS:
        debug_pointclouds["box_blue_clean_before_z"] = copy.deepcopy(target_pcd)
    describe_pointcloud("clean target", target_pcd)

    with timed_step(f"filter match target z range min={MATCH_TARGET_Z_MIN} max={MATCH_TARGET_Z_MAX}"):
        target_pcd = filter_pointcloud_by_z_range(
            target_pcd,
            MATCH_TARGET_Z_MIN,
            MATCH_TARGET_Z_MAX,
            "Target point cloud",
        )
    if WRITE_DEBUG_POINTCLOUDS:
        debug_pointclouds["box_blue_match_z_filtered"] = copy.deepcopy(target_pcd)
    describe_pointcloud("match target z-filtered", target_pcd)

    with timed_step(f"load precomputed box template PLY {BOX_TEMPLATE_PLY}"):
        source_pcd, box_template_ply = load_box_template_pointcloud()
    log(f"[box_icp] Loaded box template: {box_template_ply}")
    describe_pointcloud("source template surface", source_pcd)

    (
        obb_initial_transform,
        obb_initial_result,
        icp_result,
        source_down,
        target_down,
        source_obb,
        target_obb,
    ) = run_obb_initialized_icp(source_pcd, target_pcd, voxel_size=VOXEL_SIZE)

    with timed_step("transform source point clouds"):
        initial_source_pcd = transform_open3d_pointcloud(source_pcd, obb_initial_transform)
        refined_source_pcd = transform_open3d_pointcloud(source_pcd, icp_result.transformation)

    clean_target_count = len(target_pcd.points)
    source_count = len(source_pcd.points)
    with timed_step(
        "final downsample point clouds for output/viewer "
        f"target_voxel={FINAL_TARGET_VOXEL_DOWNSAMPLE} model_voxel={FINAL_MODEL_VOXEL_DOWNSAMPLE}"
    ):
        output_target_pcd = voxel_downsample_pointcloud(target_pcd, FINAL_TARGET_VOXEL_DOWNSAMPLE)
        output_source_pcd = voxel_downsample_pointcloud(source_pcd, FINAL_MODEL_VOXEL_DOWNSAMPLE)
        output_initial_source_pcd = voxel_downsample_pointcloud(initial_source_pcd, FINAL_MODEL_VOXEL_DOWNSAMPLE)
        output_refined_source_pcd = voxel_downsample_pointcloud(refined_source_pcd, FINAL_MODEL_VOXEL_DOWNSAMPLE)
    describe_pointcloud("output target", output_target_pcd)
    describe_pointcloud("output source", output_source_pcd)
    describe_pointcloud("output OBB initial source", output_initial_source_pcd)
    describe_pointcloud("output ICP source", output_refined_source_pcd)

    output_dir = resolve_path(OUTPUT_DIR) if OUTPUT_DIR is not None else capture_dir / "box_icp"
    with timed_step(f"write outputs to {output_dir}"):
        paths = write_registration_outputs(
            output_dir=output_dir,
            source_pcd=output_source_pcd,
            target_pcd=output_target_pcd,
            initial_source_pcd=output_initial_source_pcd,
            refined_source_pcd=output_refined_source_pcd,
            obb_initial_result=obb_initial_result,
            obb_initial_transform=obb_initial_transform,
            icp_result=icp_result,
            target_ply=target_ply,
            target_frame=target_frame,
            box_template_ply=box_template_ply,
            raw_target_count=raw_target_count,
            segmented_target_count=segmented_target_count,
            downsampled_target_count=downsampled_target_count,
            largest_cluster_id=largest_cluster_id,
            largest_cluster_count=largest_cluster_count,
            clean_target_count=clean_target_count,
            source_count=source_count,
            source_obb=source_obb,
            target_obb=target_obb,
            debug_pointclouds=debug_pointclouds if WRITE_DEBUG_POINTCLOUDS else None,
        )

    print(f"Loaded target: {target_ply} ({target_frame})")
    print(f"Loaded box template: {box_template_ply}")
    print(f"Raw target points: {raw_target_count}")
    print(f"Blue segmented points: {segmented_target_count}")
    print(f"Downsampled blue target points: {downsampled_target_count}")
    print(f"Largest blue cluster id: {largest_cluster_id}")
    print(f"Largest blue cluster points: {largest_cluster_count}")
    print(f"Match target z range: min={MATCH_TARGET_Z_MIN}, max={MATCH_TARGET_Z_MAX}")
    print(f"Model top z min / normal abs min: {MODEL_TOP_Z_MIN} / {MODEL_TOP_NORMAL_Z_ABS_MIN}")
    print(f"Clean target points before final downsample: {clean_target_count}")
    print(f"Source points before final downsample: {source_count}")
    print(f"Output target points: {len(output_target_pcd.points)}")
    print(f"Output source/OBB/ICP points: {len(output_source_pcd.points)} / {len(output_initial_source_pcd.points)} / {len(output_refined_source_pcd.points)}")
    print(f"Source OBB extent / ratio: {np.array2string(source_obb['extent'], precision=4)} / {source_obb['long_short_ratio']:.3f}")
    print(f"Target OBB extent / ratio: {np.array2string(target_obb['extent'], precision=4)} / {target_obb['long_short_ratio']:.3f}")
    print(f"OBB initial fitness/rmse: {obb_initial_result.fitness:.6f} / {obb_initial_result.inlier_rmse:.6f}")
    print(f"ICP fitness/rmse: {icp_result.fitness:.6f} / {icp_result.inlier_rmse:.6f}")
    print(f"Saved OBB initial transform: {paths['obb_initial_transform_path']}")
    print(f"Saved ICP transform: {paths['icp_transform_path']}")
    print(f"Saved summary: {paths['summary_path']}")
    if paths["debug_paths"]:
        print("Saved debug point clouds:")
        for debug_name, debug_path in paths["debug_paths"].items():
            print(f"  {debug_name}: {debug_path}")

    if SHOW_RESULT_VIEWER:
        with timed_step("open result viewer"):
            show_registration_result(output_target_pcd, output_initial_source_pcd, output_refined_source_pcd)


if __name__ == "__main__":
    main()
