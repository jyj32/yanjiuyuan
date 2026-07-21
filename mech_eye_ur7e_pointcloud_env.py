"""
Show Mech-Eye point cloud data in a UR7e-without-table WRS environment.

Default usage:
    python yanjiuyuan/mech_eye_ur7e_pointcloud_env.py

Use an existing PLY instead of capturing from the camera:
    python yanjiuyuan/mech_eye_ur7e_pointcloud_env.py --ply path/to/pointcloud.ply
"""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "captures"
UR7E_MESH_DIR = REPO_ROOT / "wrs" / "robot_sim" / "robots" / "ur7e" / "meshes"
CAMERA_Z_AXIS_LENGTH = 1.0

# Calibrated camera-to-world extrinsics.
CAM_TO_WORLD = np.array([
        [-0.998885, 0.022034, -0.041751, 0.647500],
        [-0.021796, -0.999744, -0.006148, 0.018000],
        [-0.041876, -0.005231, 0.999109, 1.267000],
        [0.000000, 0.000000, 0.000000, 1.000000],
])

OBSTACLE_SPECS = [
    {
        "name": "table",
        "mesh": "wholetable.STL",
        "pos": None,
        "ex_radius": 0.01,
        "rgba": np.array([0.55, 0.55, 0.55, 0.65]),
    },
    {
        "name": "box1",
        "mesh": "box1.stl",
        "pos": np.array([0.32, 0.22, 0.0]),
        "ex_radius": 0.003,
        "rgba": np.array([0.1, 0.65, 0.2, 0.75]),
    },
    {
        "name": "box2",
        "mesh": "box2.stl",
        "pos": np.array([0.32, 0.22, 0.0]),
        "ex_radius": 0.003,
        "rgba": np.array([0.1, 0.65, 0.2, 0.75]),
    },
    {
        "name": "box3",
        "mesh": "box3.stl",
        "pos": np.array([0.32, 0.22, 0.0]),
        "ex_radius": 0.003,
        "rgba": np.array([0.1, 0.65, 0.2, 0.75]),
    },
    {
        "name": "box4",
        "mesh": "box4.stl",
        "pos": np.array([0.32, 0.22, 0.0]),
        "ex_radius": 0.003,
        "rgba": np.array([0.1, 0.65, 0.2, 0.75]),
    },
    {
        "name": "Mecheye",
        "mesh": "Mecheye.STL",
        "pos": np.array([0.53, -0.07, 1.02]),
        "ex_radius": 0.05,
        "rgba": np.array([0.2, 0.45, 0.85, 0.75]),
    },
]


Range3D = Optional[Tuple[float, float]]


def parse_range(value: Optional[str]) -> Range3D:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range must be formatted as min,max")
    low, high = float(values[0]), float(values[1])
    if low > high:
        raise argparse.ArgumentTypeError("range min cannot be greater than max")
    return low, high


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture or load a Mech-Eye point cloud and draw it in a UR7e WRS scene."
    )
    parser.add_argument("--ply", type=Path, default=None, help="Load an existing PLY instead of capturing.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for captures.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Exact output directory for this capture.")
    parser.add_argument("--ply-out", type=Path, default=None, help="Override colored PLY output path when capturing.")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Depth scale passed to Mech_camera.")
    parser.add_argument("--depth-trunc", type=float, default=3.0, help="Depth truncation distance in meters.")
    parser.add_argument("--max-points", type=int, default=150000, help="Maximum points drawn in Panda3D.")
    parser.add_argument("--point-size", type=float, default=0.002, help="Rendered point size in meters.")
    parser.add_argument("--x-range", type=parse_range, default=None, help="Optional world X filter: min,max.")
    parser.add_argument("--y-range", type=parse_range, default=None, help="Optional world Y filter: min,max.")
    parser.add_argument("--z-range", type=parse_range, default=None, help="Optional world Z filter: min,max.")
    parser.add_argument(
        "--show-obstacle-collision",
        action="store_true",
        help="Also show collision primitives for loaded obstacles.",
    )
    parser.add_argument(
        "--detect-box",
        dest="detect_box",
        action="store_true",
        default=True,
        help="Detect the blue box and draw yanjiuyuan/models/box.STL in the estimated pose.",
    )
    parser.add_argument(
        "--no-detect-box",
        dest="detect_box",
        action="store_false",
        help="Skip blue box detection and only display the point cloud scene.",
    )
    parser.add_argument(
        "--box-transform-out",
        type=Path,
        default=None,
        help="Optional path to save the detected box 4x4 transform.",
    )
    parser.add_argument("--icp-source", type=Path, default=None, help="Object/model point cloud to align to the scene.")
    parser.add_argument("--icp-target", type=Path, default=None, help="Target scene/object point cloud. Defaults to current scene PLY.")
    parser.add_argument("--icp-voxel-size", type=float, default=0.01, help="Voxel size for Open3D registration.")
    parser.add_argument("--icp-transform-out", type=Path, default=None, help="Where to save the estimated 4x4 transform.")
    return parser.parse_args()


def make_output_dir(output_root: Path, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        run_dir = output_dir
    else:
        run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def capture_mech_eye_pointcloud(output_dir: Path, ply_out: Optional[Path], depth_scale: float, depth_trunc: float):
    import cv2
    import open3d as o3d
    from wrs.drivers.devices.Mech_eye.Mech_camera import CaptureImage

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / "rgb.png"
    colored_ply_path = ply_out if ply_out is not None else output_dir / "colored_pointcloud.ply"
    colored_ply_path.parent.mkdir(parents=True, exist_ok=True)

    camera = CaptureImage(save_directory=str(output_dir))
    try:
        rgb, _, pcd = camera.capture_and_generate_pointcloud(
            save=False,
            show=False,
            pcb_out_path=str(colored_ply_path),
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            keep_invalid=False,
        )
        if rgb is None or pcd is None:
            raise RuntimeError("Mech-Eye capture did not return RGB data and point cloud data.")
        cv2.imwrite(str(rgb_path), rgb)
        if not pcd.has_colors():
            print("Warning: captured point cloud has no color data.")
        o3d.io.write_point_cloud(str(colored_ply_path), pcd, write_ascii=False)
        return pcd, rgb_path, colored_ply_path
    finally:
        try:
            camera.camera.disconnect()
        except Exception:
            pass


def load_ply_pointcloud(ply_path: Path):
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if not pcd.has_points():
        raise RuntimeError(f"No valid points found in {ply_path}")
    return pcd


def open3d_to_numpy(pcd):
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float64)
        if len(colors) != len(points):
            colors = None

    valid_mask = np.all(np.isfinite(points), axis=1)
    valid_mask &= ~np.all(np.isclose(points, 0.0), axis=1)

    points = points[valid_mask]
    if colors is not None:
        colors = colors[valid_mask]
    if len(points) == 0:
        raise RuntimeError("Point cloud is empty after filtering invalid points.")
    return points, colors


def transform_points(points: np.ndarray, homomat: np.ndarray) -> np.ndarray:
    if np.allclose(homomat, np.eye(4)):
        return points
    points_h = np.column_stack((points, np.ones(len(points))))
    return (homomat @ points_h.T).T[:, :3]


def apply_range_filter(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    x_range: Range3D,
    y_range: Range3D,
    z_range: Range3D,
):
    mask = np.ones(len(points), dtype=bool)
    for axis, axis_range in enumerate((x_range, y_range, z_range)):
        if axis_range is None:
            continue
        low, high = axis_range
        mask &= (points[:, axis] >= low) & (points[:, axis] <= high)

    points = points[mask]
    if colors is not None:
        colors = colors[mask]
    if len(points) == 0:
        raise RuntimeError("Point cloud is empty after applying XYZ range filters.")
    return points, colors


def downsample_points(points: np.ndarray, colors: Optional[np.ndarray], max_points: int):
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    points = points[indices]
    if colors is not None:
        colors = colors[indices]
    return points, colors


def make_rgba(colors: Optional[np.ndarray], point_count: int) -> np.ndarray:
    if colors is None:
        return np.array([0.05, 0.45, 1.0, 0.75])
    alpha = np.ones((point_count, 1), dtype=np.float64)
    return np.column_stack((np.clip(colors, 0.0, 1.0), alpha))


def numpy_to_open3d_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def save_numpy_pointcloud(points: np.ndarray, colors: Optional[np.ndarray], ply_path: Path) -> None:
    import open3d as o3d

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = numpy_to_open3d_pointcloud(points, colors)
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)



def attach_camera_z_axis(base, mgm) -> None:
    camera_center = CAM_TO_WORLD[:3, 3]
    camera_z_dir = -CAM_TO_WORLD[:3, 2]
    z_axis_end = camera_center + camera_z_dir * CAMERA_Z_AXIS_LENGTH

    mgm.gen_sphere(
        pos=camera_center,
        radius=0.015,
        rgb=np.array([0.0, 0.35, 1.0]),
        alpha=1.0,
    ).attach_to(base)
    mgm.gen_arrow(
        spos=camera_center,
        epos=z_axis_end,
        rgb=np.array([0.0, 0.35, 1.0]),
        alpha=1.0,
        stick_radius=0.006,
    ).attach_to(base)

def preprocess_point_cloud_for_registration(pcd, voxel_size: float):
    # RANSAC 配准前的预处理步骤
    import open3d as o3d

    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0
    # 体素降采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # 估计法向量
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    # 计算 FPFH 特征描述子
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def estimate_object_pose_with_registration(source_path: Path, target_path: Path, voxel_size: float):
    import open3d as o3d

    reg = o3d.pipelines.registration
    source = load_ply_pointcloud(source_path)
    target = load_ply_pointcloud(target_path)
    source_down, source_fpfh = preprocess_point_cloud_for_registration(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud_for_registration(target, voxel_size)

    distance_threshold = voxel_size * 1.5
    ransac_args = [
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        reg.TransformationEstimationPointToPoint(False),
        3,
        [
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        reg.RANSACConvergenceCriteria(100000, 0.999),
    ]
    try:
        result_global = reg.registration_ransac_based_on_feature_matching(*ransac_args)
    except TypeError:
        result_global = reg.registration_ransac_based_on_feature_matching(
            *ransac_args[:4], *ransac_args[5:]
        )

    if source_down.has_colors() and target_down.has_colors() and hasattr(reg, "registration_colored_icp"):
        result_refined = reg.registration_colored_icp(
            source_down,
            target_down,
            voxel_size,
            result_global.transformation,
            reg.TransformationEstimationForColoredICP(),
            reg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=80),
        )
        method = "colored_icp"
    else:
        result_refined = reg.registration_icp(
            source_down,
            target_down,
            distance_threshold,
            result_global.transformation,
            reg.TransformationEstimationPointToPlane(),
            reg.ICPConvergenceCriteria(max_iteration=80),
        )
        method = "point_to_plane_icp"

    return result_global, result_refined, method

def mask_points_by_range(
    points: np.ndarray,
    x_range: Range3D,
    y_range: Range3D,
    z_range: Range3D,
) -> np.ndarray:
    mask = np.ones(len(points), dtype=bool)
    for axis, axis_range in enumerate((x_range, y_range, z_range)):
        if axis_range is None:
            continue
        low, high = axis_range
        if low is not None:
            mask &= points[:, axis] >= low
        if high is not None:
            mask &= points[:, axis] <= high
    return mask


def filter_arrays_by_z_range(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    z_min: Optional[float],
    z_max: Optional[float],
    label: str,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    mask = np.ones(len(points), dtype=bool)
    parts = []
    if z_min is not None:
        mask &= points[:, 2] > z_min
        parts.append(f"z > {z_min}")
    if z_max is not None:
        mask &= points[:, 2] < z_max
        parts.append(f"z < {z_max}")
    if not np.any(mask):
        raise RuntimeError(f"{label} is empty after {' and '.join(parts)} filtering.")
    filtered_colors = None if colors is None else colors[mask]
    return points[mask], filtered_colors


def estimate_box_pose_with_box_icp_from_arrays(points_world: np.ndarray, colors: Optional[np.ndarray]) -> dict:
    from yanjiuyuan import box_icp_from_saved_capture as box_icp

    points_world = np.asarray(points_world, dtype=np.float64)
    if colors is None:
        raise RuntimeError("The captured point cloud has no RGB colors; blue box segmentation needs color data.")
    colors = np.asarray(colors, dtype=np.float64)
    if len(points_world) != len(colors):
        raise RuntimeError("Point/color array length mismatch; cannot detect blue box.")

    with box_icp.timed_step("box detection: numpy crop target by configured XYZ ranges"):
        # XYZ空间范围裁剪
        crop_mask = mask_points_by_range(points_world, box_icp.X_RANGE, box_icp.Y_RANGE, box_icp.Z_RANGE)
        if not np.any(crop_mask):
            raise RuntimeError("Target point cloud is empty after XYZ range filtering.")
        cropped_points = points_world[crop_mask]
        cropped_colors = colors[crop_mask]
    box_icp.log(f"[box_icp] env box cropped target: {len(cropped_points)} / {len(points_world)} points")

    with box_icp.timed_step("box detection: numpy segment blue box points"):
        # 蓝色颜色分割
        blue_mask, components, hsv_values = box_icp.blue_color_mask_components(cropped_colors)
        box_icp.log_blue_mask_diagnostics(cropped_colors, blue_mask, components, hsv_values)
        if not np.any(blue_mask):
            raise RuntimeError("No blue box points found. Loosen BOX_BLUE_* thresholds in yanjiuyuan/constants.py.")
        blue_points = cropped_points[blue_mask]
        blue_colors = cropped_colors[blue_mask]
    segmented_target_count = len(blue_points)
    box_icp.log(f"[box_icp] env box blue segmented target: {segmented_target_count} points")

    with box_icp.timed_step(
        f"box detection: numpy filter target z range min={box_icp.MATCH_TARGET_Z_MIN} max={box_icp.MATCH_TARGET_Z_MAX}"
    ):
        #  Z高度二次过滤
        target_points, target_colors = filter_arrays_by_z_range(
            blue_points,
            blue_colors,
            box_icp.MATCH_TARGET_Z_MIN,
            box_icp.MATCH_TARGET_Z_MAX,
            "Box target point cloud",
        )
    clean_target_count = len(target_points)
    box_icp.log(f"[box_icp] env box match target before Open3D: {clean_target_count} points")

    with box_icp.timed_step("box detection: build Open3D target after numpy filtering"):
        target_pcd = numpy_to_open3d_pointcloud(target_points, target_colors)

    with box_icp.timed_step(f"box detection: voxel downsample target voxel={box_icp.TARGET_VOXEL_DOWNSAMPLE}"):
        # 体素降采样
        target_pcd = box_icp.voxel_downsample_target(target_pcd)
    downsampled_target_count = len(target_pcd.points)
    box_icp.describe_pointcloud("env box downsampled target", target_pcd)

    with box_icp.timed_step(
        f"box detection: keep largest cluster eps={box_icp.BLUE_CLUSTER_EPS} "
        f"min_points={box_icp.BLUE_CLUSTER_MIN_POINTS}"
    ):
        # 保留最大连通簇
        target_pcd, largest_cluster_id = box_icp.keep_largest_cluster(target_pcd)
    largest_cluster_count = len(target_pcd.points)
    box_icp.describe_pointcloud("env box clustered target", target_pcd)

    with box_icp.timed_step(
        "box detection: remove statistical outliers "
        f"nb_neighbors={box_icp.OUTLIER_NB_NEIGHBORS} std_ratio={box_icp.OUTLIER_STD_RATIO}"
    ):
        # 统计离群点去除
        target_pcd = box_icp.remove_target_outliers(target_pcd)
    box_icp.describe_pointcloud("env box clean target", target_pcd)

    with box_icp.timed_step(f"box detection: load box template {box_icp.BOX_TEMPLATE_PLY}"):
        # 加载箱子模板点云
        source_pcd, box_template_ply = box_icp.load_box_template_pointcloud()
    source_count = len(source_pcd.points)
    box_icp.describe_pointcloud("env box source template", source_pcd)
    # OBB初始化的Point-to-Plane ICP
    (   obb_initial_transform,
        obb_initial_result,
        icp_result,
        _source_down,
        _target_down,
        source_obb,
        target_obb,
    ) = box_icp.run_obb_initialized_icp(source_pcd, target_pcd, voxel_size=box_icp.VOXEL_SIZE)

    return {
        "transform": np.asarray(icp_result.transformation, dtype=np.float64),
        "obb_initial_transform": obb_initial_transform,
        "obb_initial_fitness": obb_initial_result.fitness,
        "obb_initial_rmse": obb_initial_result.inlier_rmse,
        "icp_fitness": icp_result.fitness,
        "icp_rmse": icp_result.inlier_rmse,
        "box_template_ply": box_template_ply,
        "segmented_target_count": segmented_target_count,
        "downsampled_target_count": downsampled_target_count,
        "largest_cluster_id": largest_cluster_id,
        "largest_cluster_count": largest_cluster_count,
        "clean_target_count": clean_target_count,
        "source_count": source_count,
        "source_obb": source_obb,
        "target_obb": target_obb,
    }

def estimate_box_pose_with_box_icp(world_pcd) -> dict:
    from yanjiuyuan import box_icp_from_saved_capture as box_icp

    with box_icp.timed_step("box detection: crop target by configured XYZ ranges"):
        target_pcd = box_icp.crop_pointcloud_by_range(world_pcd)
    box_icp.describe_pointcloud("env box cropped target", target_pcd)

    with box_icp.timed_step("box detection: segment blue box points"):
        target_pcd, segmented_target_count = box_icp.segment_blue_box_points(target_pcd)
    box_icp.describe_pointcloud("env box blue segmented target", target_pcd)

    with box_icp.timed_step(f"box detection: voxel downsample target voxel={box_icp.TARGET_VOXEL_DOWNSAMPLE}"):
        target_pcd = box_icp.voxel_downsample_target(target_pcd)
    downsampled_target_count = len(target_pcd.points)
    box_icp.describe_pointcloud("env box downsampled target", target_pcd)

    with box_icp.timed_step(
        f"box detection: keep largest cluster eps={box_icp.BLUE_CLUSTER_EPS} "
        f"min_points={box_icp.BLUE_CLUSTER_MIN_POINTS}"
    ):
        target_pcd, largest_cluster_id = box_icp.keep_largest_cluster(target_pcd)
    largest_cluster_count = len(target_pcd.points)
    box_icp.describe_pointcloud("env box clustered target", target_pcd)

    with box_icp.timed_step(
        "box detection: remove statistical outliers "
        f"nb_neighbors={box_icp.OUTLIER_NB_NEIGHBORS} std_ratio={box_icp.OUTLIER_STD_RATIO}"
    ):
        target_pcd = box_icp.remove_target_outliers(target_pcd)
    box_icp.describe_pointcloud("env box clean target", target_pcd)

    with box_icp.timed_step(
        f"box detection: filter target z range min={box_icp.MATCH_TARGET_Z_MIN} "
        f"max={box_icp.MATCH_TARGET_Z_MAX}"
    ):
        target_pcd = box_icp.filter_pointcloud_by_z_range(
            target_pcd,
            box_icp.MATCH_TARGET_Z_MIN,
            box_icp.MATCH_TARGET_Z_MAX,
            "Box target point cloud",
        )
    clean_target_count = len(target_pcd.points)
    box_icp.describe_pointcloud("env box match target", target_pcd)

    with box_icp.timed_step(f"box detection: load box template {box_icp.BOX_TEMPLATE_PLY}"):
        source_pcd, box_template_ply = box_icp.load_box_template_pointcloud()
    source_count = len(source_pcd.points)
    box_icp.describe_pointcloud("env box source template", source_pcd)

    (
        obb_initial_transform,
        obb_initial_result,
        icp_result,
        _source_down,
        _target_down,
        source_obb,
        target_obb,
    ) = box_icp.run_obb_initialized_icp(source_pcd, target_pcd, voxel_size=box_icp.VOXEL_SIZE)

    return {
        "transform": np.asarray(icp_result.transformation, dtype=np.float64),
        "obb_initial_transform": obb_initial_transform,
        "obb_initial_fitness": obb_initial_result.fitness,
        "obb_initial_rmse": obb_initial_result.inlier_rmse,
        "icp_fitness": icp_result.fitness,
        "icp_rmse": icp_result.inlier_rmse,
        "box_template_ply": box_template_ply,
        "segmented_target_count": segmented_target_count,
        "downsampled_target_count": downsampled_target_count,
        "largest_cluster_id": largest_cluster_id,
        "largest_cluster_count": largest_cluster_count,
        "clean_target_count": clean_target_count,
        "source_count": source_count,
        "source_obb": source_obb,
        "target_obb": target_obb,
    }

def resolve_mesh_path(mesh_name: str) -> Optional[Path]:
    mesh_path = UR7E_MESH_DIR / mesh_name
    if mesh_path.exists():
        return mesh_path

    mesh_name_lower = mesh_name.lower()
    for candidate in UR7E_MESH_DIR.iterdir():
        if candidate.name.lower() == mesh_name_lower:
            return candidate
    return None


def attach_obstacles(base, show_collision: bool = False):
    from wrs import mcm, mgm

    obstacle_models = []
    obstacle_names = []
    for spec in OBSTACLE_SPECS:
        mesh_path = resolve_mesh_path(spec["mesh"])
        rgba = spec["rgba"]
        if mesh_path is None:
            if spec["pos"] is None:
                print(f"Skipping obstacle {spec['name']}: missing mesh {spec['mesh']}")
                continue
            marker = mgm.gen_box(
                xyz_lengths=np.array([0.05, 0.05, 0.05]),
                pos=spec["pos"],
                rgb=rgba[:3],
                alpha=rgba[3],
            )
            marker.attach_to(base)
            obstacle_models.append(marker)
            obstacle_names.append(f"{spec['name']}(marker)")
            print(f"Displayed marker for obstacle {spec['name']}: missing mesh {spec['mesh']}")
            continue

        obstacle = mcm.CollisionModel(
            initor=str(mesh_path),
            name=spec["name"],
            cdprim_type=mcm.const.CDPrimType.AABB,
            ex_radius=spec["ex_radius"],
            rgb=rgba[:3],
            alpha=rgba[3],
        )
        if spec["pos"] is not None:
            obstacle.pos = spec["pos"]
        obstacle.attach_to(base)
        if show_collision:
            obstacle.show_cdprim()
        obstacle_models.append(obstacle)
        obstacle_names.append(spec["name"])

    print(f"Displayed {len(obstacle_models)} obstacle models or markers.")
    return obstacle_models, obstacle_names


def attach_detected_box(base, box_homomat: np.ndarray, show_collision: bool = False):
    from wrs import mcm
    from yanjiuyuan.constants import BOX_MODEL_PATH

    if not BOX_MODEL_PATH.exists():
        print(f"Skipping detected box model: missing mesh {BOX_MODEL_PATH}")
        return None

    box_model = mcm.CollisionModel(
        initor=str(BOX_MODEL_PATH),
        name="detected_box",
        cdprim_type=mcm.const.CDPrimType.AABB,
        ex_radius=0.003,
        rgb=np.array([0.55, 0.82, 1.0]),
        alpha=0.45,
    )
    box_model.homomat = box_homomat
    box_model.attach_to(base)
    if show_collision:
        box_model.show_cdprim()
    return box_model

def build_scene(
    points: np.ndarray,
    rgba: np.ndarray,
    point_size: float,
    show_obstacle_collision: bool,
    detected_box_homomat: Optional[np.ndarray] = None,
) -> None:
    from wrs import mgm, wd
    from wrs.robot_sim.robots.ur7e.ur7e_withouttable import UR7E

    base = wd.World(cam_pos=[2.0, -1.6, 1.2], lookat_pos=[0.4, -0.25, 0.3])
    mgm.gen_frame(ax_length=0.25, ax_radius=0.004).attach_to(base)

    robot = UR7E(enable_cc=True)
    robot.gen_meshmodel(toggle_flange_frame=True, toggle_jnt_frames=False, alpha=0.75).attach_to(base)

    obstacle_models, obstacle_names = attach_obstacles(base, show_collision=show_obstacle_collision)
    if detected_box_homomat is not None:
        detected_box_model = attach_detected_box(
            base,
            detected_box_homomat,
            show_collision=show_obstacle_collision,
        )
        if detected_box_model is not None:
            print("Displayed detected box.STL model in light blue.")
    attach_camera_z_axis(base, mgm)

    pointcloud_model = mgm.gen_pointcloud(points=points, rgba=rgba, point_size=point_size)
    pointcloud_model.attach_to(base)

    CAM_TO_WORLD
    mgm.gen_frame(CAM_TO_WORLD[:3,3], -CAM_TO_WORLD[:3,:3], ax_length=2).attach_to(base)

    print(f"Displayed {len(points)} points in the UR7e environment.")
    print(f"Loaded obstacle names: {obstacle_names}")
    base.run()


def main() -> None:
    args = parse_args()

    output_dir = None
    rgb_path = None
    colored_ply_path = None
    world_ply_path = None

    if args.ply is None:
        output_dir = make_output_dir(args.output_root, args.output_dir)
        pcd, rgb_path, colored_ply_path = capture_mech_eye_pointcloud(
            output_dir=output_dir,
            ply_out=args.ply_out,
            depth_scale=args.depth_scale,
            depth_trunc=args.depth_trunc,
        )
        print(f"Saved RGB image to: {rgb_path}")
        print(f"Saved colored point cloud to: {colored_ply_path}")
    else:
        colored_ply_path = args.ply
        pcd = load_ply_pointcloud(args.ply)
        print(f"Loaded point cloud from: {args.ply}")

    points, colors = open3d_to_numpy(pcd)
    raw_count = len(points)
    points = transform_points(points, CAM_TO_WORLD)

    if output_dir is not None:
        world_ply_path = output_dir / "world_colored_pointcloud.ply"
        save_numpy_pointcloud(points, colors, world_ply_path)
        print(f"Saved world-frame colored point cloud to: {world_ply_path}")

    box_detection = None
    if args.detect_box:
        try:
            print("Detecting blue box with OBB-initialized local ICP...")
            world_pcd_for_box = numpy_to_open3d_pointcloud(points, colors)
            box_detection = estimate_box_pose_with_box_icp(world_pcd_for_box)
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
            print("Detected box transform:\n" + np.array2string(box_detection["transform"], precision=6))

            box_transform_out = args.box_transform_out
            if box_transform_out is None and output_dir is not None:
                box_transform_out = output_dir / "detected_box_transform.txt"
            if box_transform_out is not None:
                box_transform_out.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(box_transform_out, box_detection["transform"], fmt="%.9f")
                print(f"Saved detected box transform to: {box_transform_out}")
        except Exception as exc:
            print(f"Warning: box detection failed; continuing without detected box model. {type(exc).__name__}: {exc}")
    if args.icp_source is not None:
        target_path = args.icp_target or world_ply_path or colored_ply_path
        if target_path is None:
            raise RuntimeError("No ICP target point cloud is available.")
        result_global, result_refined, icp_method = estimate_object_pose_with_registration(
            source_path=args.icp_source,
            target_path=target_path,
            voxel_size=args.icp_voxel_size,
        )
        transform_out = args.icp_transform_out
        if transform_out is None:
            transform_dir = output_dir if output_dir is not None else args.output_root
            transform_out = transform_dir / "icp_transform.txt"
        transform_out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(transform_out, result_refined.transformation, fmt="%.9f")
        print(f"ICP method: {icp_method}")
        print(f"Global registration fitness/rmse: {result_global.fitness:.6f} / {result_global.inlier_rmse:.6f}")
        print(f"Refined ICP fitness/rmse: {result_refined.fitness:.6f} / {result_refined.inlier_rmse:.6f}")
        print(f"Saved estimated transform to: {transform_out}")

    points, colors = apply_range_filter(points, colors, args.x_range, args.y_range, args.z_range)
    points, colors = downsample_points(points, colors, args.max_points)
    rgba = make_rgba(colors, len(points))

    print(f"Prepared {len(points)} / {raw_count} points for WRS gen_pointcloud.")
    build_scene(
        points,
        rgba,
        args.point_size,
        args.show_obstacle_collision,
        detected_box_homomat=box_detection["transform"] if box_detection is not None else None,
    )


if __name__ == "__main__":
    main()
