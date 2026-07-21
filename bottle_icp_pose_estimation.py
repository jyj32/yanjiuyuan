"""Shared bottle point-cloud registration and pose-estimation helpers."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from yanjiuyuan import bottle_icp_config as bottle_cfg
from yanjiuyuan import bottle_icp_from_saved_capture as bottle_icp


BOTTLE_TEMPLATE_PATHS = {
    "surface": Path(__file__).resolve().parent / "models" / "bottle_surface_points.ply",
    "top": Path(__file__).resolve().parent / "models" / "bottle_top_view_points.ply",
    "front": Path(__file__).resolve().parent / "models" / "bottle_front_view_points.ply",
    "left": Path(__file__).resolve().parent / "models" / "bottle_left_view_points.ply",
    "right": Path(__file__).resolve().parent / "models" / "bottle_right_view_points.ply",
}
BOTTLE_PROMPT_TEMPLATE_CHOICES = ("surface", "top", "front", "left", "right")
BOTTLE_TEMPLATE_CHOICES = ("prompt", "surface", "top", "front", "left", "right", "custom")
GLOBAL_REGISTERED_POINTS_RGB = (0.62, 0.14, 1.0)
BOTTLE_GLOBAL_RANSAC_ATTEMPTS = 3   # RANSAC粗配准循环次数


@dataclass
class GlobalRegistrationChoice:
    result: object
    source_down: object
    target_down: object
    centroid_distance: float
    attempt_count: int

@dataclass
class BottleIcpPoseConfig:
    enabled: bool = True
    template: str = "prompt"
    template_ply: Optional[Path] = None
    template_prompt_gui: bool = True
    template_preview_size: int = 260
    stl: Path = bottle_cfg.BOTTLE_STL
    voxel: float = bottle_cfg.VOXEL_SIZE
    template_voxel: float = bottle_cfg.TEMPLATE_VOXEL_SIZE
    global_ransac_n: int = bottle_cfg.GLOBAL_RANSAC_N
    global_ransac_attempts: int = BOTTLE_GLOBAL_RANSAC_ATTEMPTS
    model_sample_count: int = bottle_cfg.MODEL_SAMPLE_COUNT
    model_even_radius: Optional[float] = bottle_cfg.MODEL_EVEN_RADIUS
    icp_max_iteration: int = bottle_cfg.ICP_MAX_ITERATION



def log(message: str) -> None:
    print(message, flush=True)


def resolve_path(path: Path) -> Path:
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def transform_open3d_pointcloud(pcd, homomat: np.ndarray):
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(homomat)
    return pcd_copy


def default_bottle_icp_config() -> BottleIcpPoseConfig:
    return BottleIcpPoseConfig()


def make_transformed_bottle_mesh(bottle_stl: Path, transform: np.ndarray):
    import open3d as o3d

    bottle_stl = resolve_path(Path(bottle_stl))
    if not bottle_stl.exists():
        log(f"[box_object] Warning: missing bottle model {bottle_stl}")
        return None
    mesh = o3d.io.read_triangle_mesh(str(bottle_stl))
    if mesh.is_empty():
        log(f"[box_object] Warning: failed to load bottle model {bottle_stl}")
        return None
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1.0, 0.76, 0.18])
    mesh.transform(transform)
    return mesh


def load_bottle_template_pointcloud(config: BottleIcpPoseConfig):
    import open3d as o3d

    template_name = config.template
    if template_name == "prompt":
        raise RuntimeError("Bottle template prompt was not resolved before ICP.")
    if template_name == "custom":
        if config.template_ply is None:
            raise ValueError("Custom bottle template requires config.template_ply.")
        template_path = resolve_path(config.template_ply)
    else:
        template_path = resolve_path(BOTTLE_TEMPLATE_PATHS[template_name])

    if template_path.exists():
        source_pcd = o3d.io.read_point_cloud(str(template_path))
        if source_pcd.is_empty():
            raise RuntimeError(f"Bottle template PLY is empty: {template_path}")
        log(f"[box_object] loaded bottle template '{template_name}': {template_path} ({len(source_pcd.points)} points)")
        return source_pcd, template_path, template_name

    if template_name != "surface":
        raise FileNotFoundError(
            f"Bottle template '{template_name}' is missing: {template_path}. "
            "Run python yanjiuyuan/sample_bottle_surface.py first, or choose the surface template."
        )

    log(
        f"[box_object] bottle surface template PLY missing: {template_path}; "
        "sampling bottle.stl directly as fallback."
    )
    source_pcd, bottle_stl = bottle_icp.sample_bottle_surface_pointcloud(
        bottle_stl=resolve_path(config.stl),
        sample_count=config.model_sample_count,
        even_radius=config.model_even_radius,
    )
    return source_pcd, bottle_stl, "surface_sampled_from_stl"


def downsample_bottle_template_pointcloud(source_pcd, voxel_size: Optional[float]):
    original_count = int(len(source_pcd.points))
    if voxel_size is None or voxel_size <= 0 or original_count == 0:
        return source_pcd, original_count

    source_down = source_pcd.voxel_down_sample(float(voxel_size))
    down_count = int(len(source_down.points))
    if down_count == 0:
        log(f"[box_object] bottle template downsample voxel={float(voxel_size):.5f} produced no points; keeping original template.")
        return source_pcd, original_count

    log(f"[box_object] bottle template downsample: voxel={float(voxel_size):.5f}, points={original_count}->{down_count}")
    return source_down, original_count


def pointcloud_centroid(pcd) -> np.ndarray:
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(3, dtype=np.float64)
    return points.mean(axis=0)


def transformed_centroid_distance(source_pcd, target_pcd, transform: np.ndarray) -> float:
    source_centroid = pointcloud_centroid(source_pcd)
    target_centroid = pointcloud_centroid(target_pcd)
    source_h = np.append(source_centroid, 1.0)
    transformed_source = (np.asarray(transform, dtype=np.float64) @ source_h)[:3]
    return float(np.linalg.norm(transformed_source - target_centroid))


def preprocess_point_cloud_for_registration(pcd, voxel_size):
    import open3d as o3d
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    # 体素降采样
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5
    # 计算fpfh特征FPFH，（Fast Point Feature Histogram，快速点特征直方图） 是 3D 点云处理中的局部几何特征描述子。
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100) # type:ignore
    )
    return pcd_down, pcd_fpfh


def run_global_registration_with_retries(
    source_pcd, target_pcd, config: BottleIcpPoseConfig, ransac_n: int
) -> GlobalRegistrationChoice:  # 已修改
    # （第二次）粗匹配
    import open3d as o3d
    reg = o3d.pipelines.registration

    attempts = max(1, int(config.global_ransac_attempts))
    voxel_size = float(config.voxel)

    # ---------- 1. 预处理：体素下采样 + FPFH 特征提取（仅一次） ----------
    source_down, source_fpfh = preprocess_point_cloud_for_registration(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud_for_registration(target_pcd, voxel_size)

    distance_threshold = voxel_size * 1.5

    best_choice = None
    best_key = None

    # ---------- 2. 多次 RANSAC 尝试，复用预计算特征 ----------
    for attempt_index in range(1, attempts + 1):
        result = reg.registration_ransac_based_on_feature_matching(
            source_down, target_down,
            source_fpfh, target_fpfh,
            True,  # mutual_filter
            distance_threshold,
            reg.TransformationEstimationPointToPoint(False),
            ransac_n,
            [
                reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                reg.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            reg.RANSACConvergenceCriteria(
                50000, # RANSAC 单次最大迭代次数，原为100000
                0.999   # 置信度
            )
        )
        centroid_distance = transformed_centroid_distance(source_down, target_down, result.transformation)
        choice = GlobalRegistrationChoice(
            result=result,
            source_down=source_down,
            target_down=target_down,
            centroid_distance=centroid_distance,
            attempt_count=attempt_index,
        )
        key = (float(result.fitness), -float(result.inlier_rmse))
        log(
            "[box_object] global RANSAC attempt "
            f"{attempt_index}/{attempts}: fitness={result.fitness:.6f}, "
            f"rmse={result.inlier_rmse:.6f}, centroid_dist={centroid_distance:.4f}"
        )
        if best_key is None or key > best_key:
            best_key = key
            best_choice = choice

    if best_choice is None:
        raise RuntimeError("Global registration did not produce any result.")
    return best_choice


# def run_global_registration_with_retries(source_pcd, target_pcd, config: BottleIcpPoseConfig, ransac_n: int) -> GlobalRegistrationChoice:
#     attempts = max(1, int(config.global_ransac_attempts))
#     best_choice = None
#     best_key = None
#     # RANSAC粗配准（第二阶段ICP）
#     for attempt_index in range(1, attempts + 1):
#         result, source_down, target_down = bottle_icp.run_global_registration(
#             source_pcd,
#             target_pcd,
#             voxel_size=float(config.voxel),
#             ransac_n=ransac_n,
#         )
#         centroid_distance = transformed_centroid_distance(source_down, target_down, result.transformation)
#         choice = GlobalRegistrationChoice(
#             result=result,
#             source_down=source_down,
#             target_down=target_down,
#             centroid_distance=centroid_distance,
#             attempt_count=attempt_index,
#         )
#         key = (float(result.fitness), -float(result.inlier_rmse))
#         log(
#             "[box_object] global RANSAC attempt "
#             f"{attempt_index}/{attempts}: fitness={result.fitness:.6f}, "
#             f"rmse={result.inlier_rmse:.6f}, centroid_dist={centroid_distance:.4f}"
#         )
#         if best_key is None or key > best_key:
#             best_key = key
#             best_choice = choice
#
#     if best_choice is None:
#         raise RuntimeError("Global registration did not produce any result.")
#     return best_choice

def run_bottle_registration_on_selected(selected_points: np.ndarray, output_dir: Path, config: BottleIcpPoseConfig) -> dict:
    selected_points = np.asarray(selected_points, dtype=np.float64)
    if selected_points.ndim != 2 or selected_points.shape[1] != 3 or len(selected_points) == 0:
        raise RuntimeError("Selected object point cloud is empty; cannot run bottle ICP.")
    if len(selected_points) < 20:
        raise RuntimeError(f"Selected object point cloud is too small for bottle ICP: {len(selected_points)} points.")

    bottle_output_dir = output_dir / "bottle_icp"
    bottle_output_dir.mkdir(parents=True, exist_ok=True)
    bottle_stl = resolve_path(config.stl)

    log(
        "[box_object] bottle ICP settings: "
        f"stl={bottle_stl}, template={config.template}, target_points={len(selected_points)}, "
        f"voxel={config.voxel}, template_voxel={config.template_voxel}, "
        f"ransac_n={config.global_ransac_n}, samples={config.model_sample_count}, "
        f"even_radius={config.model_even_radius}, max_iter={config.icp_max_iteration}"
    )

    target_pcd = bottle_icp.numpy_to_open3d_pointcloud(selected_points)
    source_pcd, template_path, template_name = load_bottle_template_pointcloud(config)
    source_pcd, source_original_point_count = downsample_bottle_template_pointcloud(
        source_pcd,
        config.template_voxel,
    )
    ransac_n = max(3, int(config.global_ransac_n))

    old_icp_max_iteration = bottle_icp.ICP_MAX_ITERATION
    bottle_icp.ICP_MAX_ITERATION = int(config.icp_max_iteration)
    try:
        global_choice = run_global_registration_with_retries(source_pcd, target_pcd, config, ransac_n)
        global_result = global_choice.result
        source_down = global_choice.source_down
        target_down = global_choice.target_down
        icp_result = bottle_icp.run_point_to_plane_icp(
            source_down,
            target_down,
            global_result.transformation,
            voxel_size=float(config.voxel),
        )
    finally:
        bottle_icp.ICP_MAX_ITERATION = old_icp_max_iteration

    global_transform_path = bottle_output_dir / "bottle_global_transform.txt"
    icp_transform_path = bottle_output_dir / "bottle_icp_transform.txt"
    summary_path = bottle_output_dir / "bottle_icp_summary.json"

    np.savetxt(global_transform_path, global_result.transformation, fmt="%.9f")
    np.savetxt(icp_transform_path, icp_result.transformation, fmt="%.9f")

    summary = {
        "bottle_stl": str(bottle_stl),
        "template": template_name,
        "template_path": str(template_path),
        "target_point_count": int(len(target_pcd.points)),
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
        "target_path": None,
        "global_registered_path": None,
        "icp_registered_path": None,
        "registered_model_path": None,
        "pointcloud_outputs_saved": False,
        "global_transform_path": str(global_transform_path),
        "icp_transform_path": str(icp_transform_path),
        "global_transform": np.asarray(global_result.transformation, dtype=np.float64).tolist(),
        "icp_transform": np.asarray(icp_result.transformation, dtype=np.float64).tolist(),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(
        "[box_object] bottle ICP result: "
        f"global_selected_attempt={global_choice.attempt_count}, global_centroid_dist={global_choice.centroid_distance:.4f}, "
        f"global_fitness={global_result.fitness:.6f}, global_rmse={global_result.inlier_rmse:.6f}, "
        f"icp_fitness={icp_result.fitness:.6f}, icp_rmse={icp_result.inlier_rmse:.6f}"
    )
    return summary
