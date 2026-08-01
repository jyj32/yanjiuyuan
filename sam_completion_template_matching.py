from __future__ import annotations

import time

"""
Complete SAM-selected world-frame points with AdaPoinTr, then match a full template.
用AdaPoinTr补全完整点云，icp匹配
Coordinate contract:
  box_object_pointcloud_from_saved_capture_bottle_icp.py selects points in world frame;
  infer_AdaPoinTr.py expects real captured object points in camera frame and returns
  completed points in the same camera frame. This module does world -> camera ->
  completion -> world before full-template RANSAC+ICP.
"""

import argparse
import copy
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
YANJIUYUAN_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPOINTR_SCRIPT = REPO_ROOT / "poind_cloud_completion" / "v2" / "pcn_train" / "AdapoinTr" / "infer_AdaPoinTr.py"
DEFAULT_ADAPOINTR_CHECKPOINT = (
    REPO_ROOT / "poind_cloud_completion" / "v2" / "pcn_train" / "AdapoinTr" /
    "log" / "train_AdaPoinTr_corrosion" / "checkpoints" / "best_combo.pth"
)
DEFAULT_FULL_TEMPLATE_PLY = YANJIUYUAN_DIR / "models" / "bottle_surface_points.ply"


@dataclass
class CompletionMatchingConfig:
    """Configuration for SAM point-cloud completion and full-template matching."""

    adapointr_script: Path
    adapointr_checkpoint: Path
    full_template_ply: Path
    output_dir: Path
    cam_to_world: np.ndarray
    device: str = "cuda:0"
    global_scale: float = 0.4
    num_points: int = 1024
    num_query: int = 128
    voxel_size: float = 0.005   # 降采样参数
    template_voxel_size: Optional[float] = 0.005
    ransac_n: int = 3
    ransac_attempts: int = 3    # global registration的粗配准循环
    icp_max_iteration: int = 80
    network_input_points: int = 2048    # 点云补全采集的点数
    selected_outlier_nb_neighbors: int = 24
    selected_outlier_std_ratio: float = 1.8
    selected_outlier_min_keep_ratio: float = 0.65
    save_debug_outputs: bool = False


@dataclass
class CompletionMatchingResult:
    completed_world_points: np.ndarray
    transform: np.ndarray
    summary: dict
    summary_path: Path


def resolve_path(path: Path) -> Path:
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def normalize_homomat(homomat: np.ndarray, label: str = "homomat") -> np.ndarray:
    homomat = np.asarray(homomat, dtype=np.float64)
    if homomat.shape == (1, 4, 4):
        homomat = homomat[0]
    if homomat.shape != (4, 4):
        raise ValueError(f"Expected {label} to be 4x4, got {homomat.shape}.")
    return homomat


def default_cam_to_world() -> np.ndarray:
    from yanjiuyuan.mech_eye_ur7e_pointcloud_env import CAM_TO_WORLD

    return normalize_homomat(CAM_TO_WORLD, "CAM_TO_WORLD")


def load_homomat(path: Optional[Path]) -> np.ndarray:
    if path is None:
        return default_cam_to_world()
    return normalize_homomat(np.loadtxt(resolve_path(path), dtype=np.float64), str(path))


def transform_points(points: np.ndarray, homomat: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homomat = normalize_homomat(homomat)
    if len(points) == 0:
        return points.copy()
    if np.allclose(homomat, np.eye(4)):
        return points.copy()
    points_h = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (homomat @ points_h.T).T[:, :3]


XY_PLANE_REFLECTION = np.diag([1.0, 1.0, -1.0, 1.0])


def reflect_points_across_xy_plane(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    reflected = points.copy()
    reflected[:, 2] *= -1.0
    return reflected


def translation_homomat(offset: np.ndarray) -> np.ndarray:
    offset = np.asarray(offset, dtype=np.float64).reshape(3)
    homomat = np.eye(4, dtype=np.float64)
    homomat[:3, 3] = offset
    return homomat


def numpy_to_open3d_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def open3d_to_numpy_points(pcd) -> np.ndarray:
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise RuntimeError("Point cloud is empty or invalid.")
    return points


def read_points(path: Path) -> np.ndarray:
    import open3d as o3d

    path = resolve_path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
        return np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if suffix == ".ply":
        pcd = o3d.io.read_point_cloud(str(path))
        return open3d_to_numpy_points(pcd)
    raise ValueError(f"Unsupported point-cloud input: {path}")


def write_pointcloud(points: np.ndarray, path: Path, colors: Optional[np.ndarray] = None):
    import open3d as o3d

    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = numpy_to_open3d_pointcloud(points, colors=colors)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, np.asarray(points, dtype=np.float32))
    elif suffix == ".ply":
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    else:
        raise ValueError(f"Unsupported point-cloud output: {path}")
    return pcd

def farthest_point_sample_points(points: np.ndarray, n_sample: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n_sample = int(n_sample)
    if n_sample <= 0:
        raise ValueError(f"n_sample must be positive, got {n_sample}")
    if len(points) <= n_sample:
        return points.copy()

    distances = np.full(len(points), np.inf, dtype=np.float64)
    sampled_idx = np.zeros(n_sample, dtype=np.int64)
    center = points.mean(axis=0)
    farthest = int(np.argmax(np.sum((points - center) ** 2, axis=1)))
    for idx in range(n_sample):
        sampled_idx[idx] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))
    return points[sampled_idx]


def force_point_count(points: np.ndarray, target_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")
    if len(points) == 0:
        raise RuntimeError("Cannot resample an empty point cloud.")
    if len(points) > target_count:
        return farthest_point_sample_points(points, target_count)
    if len(points) == target_count:
        return points.copy()
    repeat_count = target_count // len(points)
    remainder = target_count % len(points)
    chunks = []
    if repeat_count > 0:
        chunks.append(np.tile(points, (repeat_count, 1)))
    if remainder > 0:
        chunks.append(points[:remainder])
    return np.vstack(chunks)


def remove_statistical_outlier_points(
    points: np.ndarray,
    nb_neighbors: int,
    std_ratio: float,
    min_keep_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    summary = {
        "method": "open3d.remove_statistical_outlier",
        "input_count": int(len(points)),
        "nb_neighbors": int(nb_neighbors),
        "std_ratio": float(std_ratio),
        "min_keep_ratio": float(min_keep_ratio),
        "applied": False,
        "kept_count": int(len(points)),
        "removed_count": 0,
        "reason": None,
    }
    if len(points) == 0:
        summary["reason"] = "empty_input"
        return points.copy(), np.zeros(0, dtype=bool), summary
    if nb_neighbors <= 1 or len(points) <= 3:
        summary["reason"] = "too_few_points_or_neighbors"
        return points.copy(), np.ones(len(points), dtype=bool), summary

    min_keep_count = max(20, int(np.ceil(len(points) * float(min_keep_ratio))))
    try:
        pcd = numpy_to_open3d_pointcloud(points)
        neighbor_count = min(max(2, int(nb_neighbors)), len(points) - 1)
        _filtered_pcd, indices = pcd.remove_statistical_outlier(
            nb_neighbors=neighbor_count,
            std_ratio=float(std_ratio),
        )
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) < min_keep_count:
            summary.update(
                {
                    "nb_neighbors": int(neighbor_count),
                    "kept_count": int(len(points)),
                    "removed_count": 0,
                    "reason": f"filter_would_keep_too_few_points:{len(indices)}<{min_keep_count}",
                }
            )
            return points.copy(), np.ones(len(points), dtype=bool), summary
        mask = np.zeros(len(points), dtype=bool)
        mask[indices] = True
        filtered = points[mask]
        summary.update(
            {
                "nb_neighbors": int(neighbor_count),
                "applied": True,
                "kept_count": int(len(filtered)),
                "removed_count": int(len(points) - len(filtered)),
            }
        )
        return filtered, mask, summary
    except Exception as exc:
        summary["reason"] = f"filter_failed:{exc}"
        return points.copy(), np.ones(len(points), dtype=bool), summary


def _voxel_downsample_for_fps(points: np.ndarray, voxel_size: float = 0.002, max_points: int = 8000) -> np.ndarray:
    """体素降采样减少点数，加速后续FPS。点数已少于max_points时直接返回。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) <= max_points:
        return points
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size)
    result = np.asarray(pcd.points, dtype=np.float64)
    if len(result) > max_points:
        idx = np.random.default_rng(42).choice(len(result), max_points, replace=False)
        result = result[idx]
    return result


def prepare_selected_network_input(camera_points: np.ndarray, config: CompletionMatchingConfig) -> dict:
    # 体素降采样 → FPS下采样 → 统计离群点滤波 → 强制点数对齐
    target_count = max(1, int(config.network_input_points))
    # 先体素降采样减少点数（十几万→约5千），大幅加速后续FPS（新加）
    camera_points = _voxel_downsample_for_fps(camera_points)
    downsampled_points = farthest_point_sample_points(camera_points, target_count)

    filtered_points, inlier_mask, outlier_summary = remove_statistical_outlier_points(
        downsampled_points,
        nb_neighbors=int(config.selected_outlier_nb_neighbors),
        std_ratio=float(config.selected_outlier_std_ratio),
        min_keep_ratio=float(config.selected_outlier_min_keep_ratio),
    )
    network_points = force_point_count(filtered_points, target_count)
    outlier_summary.update(
        {
            "downsampled_count": int(len(downsampled_points)),
            "network_input_count_after_resample": int(len(network_points)),
        }
    )
    return {
        "downsampled_points": downsampled_points,
        "filtered_points": filtered_points,
        "network_points": network_points,
        "inlier_mask": inlier_mask,
        "summary": outlier_summary,
    }

_ADAPOINTR_MODULE_CACHE: dict[str, object] = {}


def _load_adapointr_module(script_path: Path):
    script_path = resolve_path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"AdaPoinTr inference script does not exist: {script_path}")
    cache_key = str(script_path)
    cached = _ADAPOINTR_MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    old_path = list(sys.path)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module_name = f"_adapointr_infer_{abs(hash(str(script_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, str(script_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load AdaPoinTr inference script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "infer"):
            raise RuntimeError(f"AdaPoinTr script has no infer(params) function: {script_path}")
        _ADAPOINTR_MODULE_CACHE[cache_key] = module
        return module
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_dont_write_bytecode


def run_adapointr_completion(
    camera_points: np.ndarray,
    config: CompletionMatchingConfig,
    output_prefix: str = "sam_object",
) -> dict:
    # AdaPoinTr点云补全函数

    output_dir = resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_debug = bool(config.save_debug_outputs)

    partial_camera_path = output_dir / f"{output_prefix}_partial_camera.ply"
    selected_downsampled_camera_path = output_dir / f"{output_prefix}_selected_downsampled_camera_2048.ply"
    selected_filtered_camera_path = output_dir / f"{output_prefix}_selected_downsampled_filtered_camera.ply"
    adapointr_input_camera_path = output_dir / f"{output_prefix}_adapointr_input_camera_2048.ply"
    network_input_camera_path = output_dir / f"{output_prefix}_network_input_camera_2048.ply"
    normalized_input_camera_path = output_dir / f"{output_prefix}_network_input_normalized_2048.npy"
    preprocess_meta_path = output_dir / f"{output_prefix}_adapointr_preprocess.json"
    completed_camera_path = output_dir / f"{output_prefix}_completed_camera.ply"
    completed_camera_npy_path = output_dir / f"{output_prefix}_completed_camera.npy"
    camera_points = np.asarray(camera_points, dtype=np.float64).reshape(-1, 3)  # 0.0s把输入点云转成 (N, 3) 的 float64 数组
    if save_debug:
        write_pointcloud(camera_points, partial_camera_path)
    # FPS下采样 → 统计离群点滤波 → 强制点数对齐
    selected_input = prepare_selected_network_input(camera_points, config)  # 10s➡0.26s
    if save_debug:
        write_pointcloud(selected_input["downsampled_points"], selected_downsampled_camera_path)
        write_pointcloud(selected_input["filtered_points"], selected_filtered_camera_path)
        write_pointcloud(selected_input["network_points"], adapointr_input_camera_path)

    module = _load_adapointr_module(config.adapointr_script)    # 动态加载推理脚本,0.022s
    params = SimpleNamespace(
        input=str(adapointr_input_camera_path),
        input_points=np.asarray(selected_input["network_points"], dtype=np.float32),
        input_point_count=int(config.network_input_points),
        input_type=2,
        checkpoint=str(resolve_path(config.adapointr_checkpoint)),
        save_path=str(completed_camera_path) if save_debug else None,
        save_output=save_debug,
        global_scale=float(config.global_scale),
        num_points=int(config.num_points),
        num_query=int(config.num_query),
        device=str(config.device),
        visualize=False,
        save_network_input_path=str(network_input_camera_path) if save_debug else None,
        save_normalized_input_path=str(normalized_input_camera_path) if save_debug else None,
        save_preprocess_meta_path=str(preprocess_meta_path) if save_debug else None,
    )
    time2 = time.time()
    completed_in_memory = module.infer(params)    # 执行推理1.1s
    time3 = time.time()
    print(f"infer module time: {time3 - time2}")
    time4 = time.time()
    if completed_in_memory is not None:
        completed_camera_points = np.asarray(completed_in_memory, dtype=np.float64).reshape(-1, 3)
    elif completed_camera_path.exists():
        # Compatibility fallback for an older external AdaPoinTr script.
        completed_camera_points = read_points(completed_camera_path)
    else:
        raise RuntimeError("AdaPoinTr inference returned no completed point cloud.")
    if save_debug:
        np.save(completed_camera_npy_path, completed_camera_points.astype(np.float32))
    network_input_camera_points = np.asarray(selected_input["network_points"], dtype=np.float64)
    time5 = time.time()
    print(f"input time: {time5 - time4}")   # 0.01
    return {
        "partial_camera_path": partial_camera_path if save_debug else None,
        "selected_downsampled_camera_path": selected_downsampled_camera_path if save_debug else None,
        "selected_downsampled_camera_points": selected_input["downsampled_points"],
        "selected_filtered_camera_path": selected_filtered_camera_path if save_debug else None,
        "selected_filtered_camera_points": selected_input["filtered_points"],
        "selected_outlier_filter": selected_input["summary"],
        "adapointr_input_camera_path": adapointr_input_camera_path if save_debug else None,
        "network_input_camera_path": network_input_camera_path if save_debug else None,
        "network_input_camera_points": network_input_camera_points,
        "normalized_input_camera_path": normalized_input_camera_path if save_debug else None,
        "preprocess_meta_path": preprocess_meta_path if save_debug else None,
        "completed_camera_path": completed_camera_path if save_debug else None,
        "completed_camera_npy_path": completed_camera_npy_path if save_debug else None,
        "completed_camera_points": completed_camera_points,
    }

def preprocess_point_cloud_for_registration(pcd, voxel_size: float):
    import open3d as o3d

    if len(pcd.points) == 0:
        raise RuntimeError("Cannot register an empty point cloud.")
    pcd_down = pcd.voxel_down_sample(float(voxel_size))
    if len(pcd_down.points) == 0:
        raise RuntimeError(f"Voxel downsample produced an empty point cloud, voxel={voxel_size}.")

    radius_normal = float(voxel_size) * 2.0
    radius_feature = float(voxel_size) * 5.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def pointcloud_centroid(pcd) -> np.ndarray:
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(3, dtype=np.float64)
    return points.mean(axis=0)


def transformed_centroid_distance(source_pcd, target_pcd, transform: np.ndarray) -> float:
    source_centroid_h = np.append(pointcloud_centroid(source_pcd), 1.0)
    source_in_target = (np.asarray(transform, dtype=np.float64) @ source_centroid_h)[:3]
    return float(np.linalg.norm(source_in_target - pointcloud_centroid(target_pcd)))


def run_global_registration(source_pcd, target_pcd, voxel_size: float, ransac_n: int):
    # Global Registration（RANSAC 粗配准）
    import open3d as o3d

    reg = o3d.pipelines.registration
    source_down, source_fpfh = preprocess_point_cloud_for_registration(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud_for_registration(target_pcd, voxel_size)

    distance_threshold = float(voxel_size) * 1.5
    args = [
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        reg.TransformationEstimationPointToPoint(False),
        int(ransac_n),
        [
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        reg.RANSACConvergenceCriteria(100000, 0.999),
    ]
    try:
        result = reg.registration_ransac_based_on_feature_matching(*args)
    except TypeError:
        result = reg.registration_ransac_based_on_feature_matching(*args[:4], *args[5:])
    return result, source_down, target_down


# def run_global_registration_with_retries(source_pcd, target_pcd, config: CompletionMatchingConfig):
#     # 多次RANSAC重试取最优
#     best = None
#     best_key = None
#     attempts = max(1, int(config.ransac_attempts))
#     ransac_n = max(3, int(config.ransac_n))
#
#     for attempt_index in range(1, attempts + 1):
#         # 单次 RANSAC
#         result, source_down, target_down = run_global_registration(
#             source_pcd,
#             target_pcd,
#             voxel_size=float(config.voxel_size),    # 体素降采样参数
#             ransac_n=ransac_n,
#         )
#         centroid_distance = transformed_centroid_distance(source_down, target_down, result.transformation)
#         key = (float(result.fitness), -float(result.inlier_rmse), -centroid_distance)
#         if best_key is None or key > best_key:
#             best_key = key
#             best = {
#                 "attempt": attempt_index,
#                 "result": result,
#                 "source_down": source_down,
#                 "target_down": target_down,
#                 "centroid_distance": centroid_distance,
#             }
#     if best is None:
#         raise RuntimeError("Global registration did not produce a result.")
#     return best

def run_global_registration_with_retries(source_pcd, target_pcd, config: CompletionMatchingConfig):
    # global_registration粗配准（第一次）
    import open3d as o3d
    reg = o3d.pipelines.registration

    voxel_size = float(config.voxel_size)
    ransac_n = max(3, int(config.ransac_n))
    attempts = max(1, int(config.ransac_attempts))

    # ===== 提前计算一次 FPFH 特征 =====
    source_down, source_fpfh = preprocess_point_cloud_for_registration(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud_for_registration(target_pcd, voxel_size)

    distance_threshold = voxel_size * 1.5

    best = None
    best_key = None
    for attempt_index in range(1, attempts + 1):
        # 使用预计算的特征进行 RANSAC
        result = reg.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            True, distance_threshold,
            reg.TransformationEstimationPointToPoint(False),
            ransac_n,
            [
                reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                reg.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            reg.RANSACConvergenceCriteria(
                50000, # RANSAC 单次最大迭代次数，原为100000
                0.999 # 置信度
            )
        )
        centroid_distance = transformed_centroid_distance(source_down, target_down, result.transformation)
        key = (float(result.fitness), -float(result.inlier_rmse), -centroid_distance)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "attempt": attempt_index,
                "result": result,
                "source_down": source_down,
                "target_down": target_down,
                "centroid_distance": centroid_distance,
            }

    if best is None:
        raise RuntimeError("Global registration did not produce a result.")
    return best


def downsample_template_if_needed(template_pcd, voxel_size: Optional[float]):
    if voxel_size is None or float(voxel_size) <= 0:
        return template_pcd, int(len(template_pcd.points))
    original_count = int(len(template_pcd.points))
    down = template_pcd.voxel_down_sample(float(voxel_size))
    if len(down.points) == 0:
        return template_pcd, original_count
    return down, original_count


def run_template_matching(
    completed_world_points: np.ndarray,
    config: CompletionMatchingConfig,
    output_prefix: str = "sam_object",
) -> dict:
    # 把瓶子表面模板点云对齐到补全后的点云上，在相机坐标系下完成 RANSAC 粗配准 + ICP 精配准
    import open3d as o3d

    output_dir = resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_path = resolve_path(config.full_template_ply)
    if not template_path.exists():
        raise FileNotFoundError(f"Full template point cloud does not exist: {template_path}")
    # 读取瓶子表面模板 PLY
    source_template = o3d.io.read_point_cloud(str(template_path))
    if source_template.is_empty():
        raise RuntimeError(f"Full template point cloud is empty: {template_path}")
    # 体素下采样，减少点数加速配准
    source_template, source_original_count = downsample_template_if_needed(
        source_template,
        config.template_voxel_size,
    )

    target_pcd = numpy_to_open3d_pointcloud(completed_world_points)
    # RANSAC粗配准（global registration）（3次取最优）
    import time
    start_time = time.time()
    global_choice = run_global_registration_with_retries(source_template, target_pcd, config)
    RANSAC_time = time.time()
    print(f"global registration的粗配准时间：{RANSAC_time - start_time}")  # 4.576s➡优化后2.706s,1.76s,1.8s
    # ICP 精配准,0.0175s，0.0305s
    reg = o3d.pipelines.registration
    icp_result = reg.registration_icp(
        global_choice["source_down"],
        global_choice["target_down"],
        float(config.voxel_size) * 0.7,
        global_choice["result"].transformation,
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=int(config.icp_max_iteration),
        ),
    )
    print(f"ICP 精配准：{time.time() - RANSAC_time}")
    global_registered = copy.deepcopy(source_template)
    global_registered.transform(global_choice["result"].transformation)
    icp_registered = copy.deepcopy(source_template)
    icp_registered.transform(icp_result.transformation)

    global_transform_path = output_dir / f"{output_prefix}_template_global_transform.txt"
    icp_transform_path = output_dir / f"{output_prefix}_template_icp_transform.txt"
    global_registered_path = output_dir / f"{output_prefix}_template_global_registered.ply"
    icp_registered_path = output_dir / f"{output_prefix}_template_icp_registered.ply"

    np.savetxt(global_transform_path, global_choice["result"].transformation, fmt="%.9f")
    np.savetxt(icp_transform_path, icp_result.transformation, fmt="%.9f")
    o3d.io.write_point_cloud(str(global_registered_path), global_registered, write_ascii=False)
    o3d.io.write_point_cloud(str(icp_registered_path), icp_registered, write_ascii=False)

    return {
        "template_path": template_path,
        "template_original_point_count": source_original_count,
        "template_point_count": int(len(source_template.points)),
        "target_point_count": int(len(target_pcd.points)),
        "voxel_size": float(config.voxel_size),
        "template_voxel_size": None
        if config.template_voxel_size is None or config.template_voxel_size <= 0
        else float(config.template_voxel_size),
        "ransac_n": max(3, int(config.ransac_n)),
        "ransac_attempts": max(1, int(config.ransac_attempts)),
        "global_selected_attempt": int(global_choice["attempt"]),
        "global_centroid_distance": float(global_choice["centroid_distance"]),
        "global_fitness": float(global_choice["result"].fitness),
        "global_inlier_rmse": float(global_choice["result"].inlier_rmse),
        "icp_fitness": float(icp_result.fitness),
        "icp_inlier_rmse": float(icp_result.inlier_rmse),
        "global_transform": np.asarray(global_choice["result"].transformation, dtype=np.float64),
        "icp_transform": np.asarray(icp_result.transformation, dtype=np.float64),
        "global_transform_path": global_transform_path,
        "icp_transform_path": icp_transform_path,
        "global_registered_path": global_registered_path,
        "icp_registered_path": icp_registered_path,
    }


def project_template_matching_to_world(
    matching: dict,
    network_to_world: np.ndarray,
    output_dir: Path,
    output_prefix: str,
) -> dict:
    output_dir = resolve_path(output_dir)
    network_to_world = normalize_homomat(network_to_world, "network_to_world")
    projected: dict[str, object] = {}

    for stage in ("global", "icp"):
        local_registered_path = matching[f"{stage}_registered_path"]
        world_registered_path = output_dir / f"{output_prefix}_template_{stage}_registered_world.ply"
        local_transform = np.asarray(matching[f"{stage}_transform"], dtype=np.float64)
        world_transform = network_to_world @ local_transform
        world_transform_path = output_dir / f"{output_prefix}_template_{stage}_transform_world.txt"

        local_registered_points = read_points(local_registered_path)
        world_registered_points = transform_points(local_registered_points, network_to_world)
        write_pointcloud(world_registered_points, world_registered_path)
        np.savetxt(world_transform_path, world_transform, fmt="%.9f")

        projected[f"{stage}_registered_world_path"] = world_registered_path
        projected[f"{stage}_transform_world_path"] = world_transform_path
        projected[f"{stage}_transform_world"] = world_transform

    return projected


def run_completion_matching_on_sam_world_points(
    selected_points_world: np.ndarray,
    config: CompletionMatchingConfig,
    output_prefix: str = "sam_object",
    run_matching: bool = True,
) -> CompletionMatchingResult:
    """Complete SAM-selected world points after centering them in camera frame."""
    # 补全完整点云，在相机坐标系下完成 RANSAC 粗配准 + ICP 精配准
    selected_points_world = np.asarray(selected_points_world, dtype=np.float64).reshape(-1, 3)
    if len(selected_points_world) < 20:
        raise RuntimeError(
            f"SAM point cloud is too small for completion/matching: {len(selected_points_world)} points."
        )

    output_dir = resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    world_to_cam = np.linalg.inv(np.asarray(config.cam_to_world, dtype=np.float64))
    save_debug = bool(config.save_debug_outputs)

    partial_world_path = output_dir / f"{output_prefix}_partial_world.ply"
    partial_camera_original_path = output_dir / f"{output_prefix}_partial_camera_original.ply"
    partial_camera_centered_unflipped_path = output_dir / f"{output_prefix}_partial_camera_centered_unflipped.ply"
    network_input_camera_original_path = output_dir / f"{output_prefix}_network_input_camera_original_2048.ply"
    network_input_world_path = output_dir / f"{output_prefix}_network_input_world_2048.ply"
    completed_camera_original_path = output_dir / f"{output_prefix}_completed_camera_original.ply"
    completed_world_path = output_dir / f"{output_prefix}_completed_world.ply"
    matching_target_camera_path = output_dir / f"{output_prefix}_matching_target_selected_filtered_plus_completed_camera.ply"
    matching_target_world_path = output_dir / f"{output_prefix}_matching_target_selected_filtered_plus_completed_world.ply"
    summary_path = output_dir / f"{output_prefix}_completion_matching_summary.json"

    if save_debug:
        write_pointcloud(selected_points_world, partial_world_path)
    selected_points_camera = transform_points(selected_points_world, world_to_cam)
    if save_debug:
        write_pointcloud(selected_points_camera, partial_camera_original_path)

    selected_center_camera = selected_points_camera.mean(axis=0)
    selected_points_camera_centered = selected_points_camera - selected_center_camera
    if save_debug:
        write_pointcloud(selected_points_camera_centered, partial_camera_centered_unflipped_path)
    selected_points_camera_origin = reflect_points_across_xy_plane(selected_points_camera_centered)
    network_to_camera_original = translation_homomat(selected_center_camera) @ XY_PLANE_REFLECTION
    network_to_world = np.asarray(config.cam_to_world, dtype=np.float64) @ network_to_camera_original
    # AdaPoinTr点云补全
    start_time = time.time()
    completion = run_adapointr_completion(
        selected_points_camera_origin,
        config,
        output_prefix=output_prefix,
    )   # 1.03s左右
    print(f"AdaPoinTr点云补全时间: {time.time() - start_time}")
    network_input_camera_origin_points = np.asarray(completion["network_input_camera_points"], dtype=np.float64).reshape(-1, 3)
    network_input_camera_centered_points = reflect_points_across_xy_plane(network_input_camera_origin_points)
    network_input_camera_original_points = network_input_camera_centered_points + selected_center_camera
    if save_debug:
        write_pointcloud(network_input_camera_original_points, network_input_camera_original_path)
    network_input_world_points = transform_points(network_input_camera_original_points, config.cam_to_world)
    if save_debug:
        write_pointcloud(network_input_world_points, network_input_world_path)

    completed_camera_origin_points = np.asarray(completion["completed_camera_points"], dtype=np.float64).reshape(-1, 3)
    completed_camera_centered_points = reflect_points_across_xy_plane(completed_camera_origin_points)
    completed_camera_original_points = completed_camera_centered_points + selected_center_camera
    if save_debug:
        write_pointcloud(completed_camera_original_points, completed_camera_original_path)
    completed_world_points = transform_points(completed_camera_original_points, config.cam_to_world)
    if save_debug:
        write_pointcloud(completed_world_points, completed_world_path)

    selected_filtered_camera_origin_points = np.asarray(completion["selected_filtered_camera_points"], dtype=np.float64).reshape(-1, 3)
    matching_target_camera_origin_points = np.vstack((selected_filtered_camera_origin_points, completed_camera_origin_points))
    if save_debug:
        write_pointcloud(matching_target_camera_origin_points, matching_target_camera_path)
    matching_target_camera_centered_points = reflect_points_across_xy_plane(matching_target_camera_origin_points)
    matching_target_camera_original_points = matching_target_camera_centered_points + selected_center_camera
    matching_target_world_points = transform_points(matching_target_camera_original_points, config.cam_to_world)
    if save_debug:
        write_pointcloud(matching_target_world_points, matching_target_world_path)

    matching = None
    if run_matching:
        # 把瓶子表面模板点云对齐到补全后的点云上，在相机坐标系下完成 RANSAC 粗配准 + ICP 精配准
        matching = run_template_matching(
            matching_target_camera_origin_points,
            config,
            output_prefix=output_prefix,
        )   #
        matching.update(
            project_template_matching_to_world(
                matching,
                network_to_world,
                output_dir,
                output_prefix,
            )
        )

    def saved_debug_path(path) -> Optional[str]:
        if not save_debug or path is None:
            return None
        return str(path)

    summary = {
        "pipeline": "sam_world_points -> camera_frame_centered_at_selected_mean -> xy_plane_reflection -> adapointr_completion"
        + (" -> completion_frame_surface_template_icp_projected_to_world" if run_matching else ""),
        "sam_partial_world_point_count": int(len(selected_points_world)),
        "completed_world_point_count": int(len(completed_world_points)),
        "matching_target_camera_point_count": int(len(matching_target_camera_origin_points)),
        "cam_to_world": np.asarray(config.cam_to_world, dtype=np.float64).tolist(),
        "world_to_camera": world_to_cam.tolist(),
        "adapointr": {
            "script": str(resolve_path(config.adapointr_script)),
            "checkpoint": str(resolve_path(config.adapointr_checkpoint)),
            "device": str(config.device),
            "global_scale": float(config.global_scale),
            "num_points": int(config.num_points),
            "num_query": int(config.num_query),
            "input_coordinate_frame": "camera_frame_minus_selected_center_camera_reflected_across_xy_plane",
            "center_estimator": "mean_of_selected_camera_points_before_fps",
            "selected_center_camera": selected_center_camera.astype(float).tolist(),
            "xy_plane_reflection_applied": True,
            "xy_plane_reflection": "z := -z after centering; inverse reflection is applied before camera_original/world outputs",
            "xy_plane_reflection_matrix": XY_PLANE_REFLECTION.astype(float).tolist(),
            "network_to_camera_original_transform": network_to_camera_original.astype(float).tolist(),
            "network_to_world_transform": network_to_world.astype(float).tolist(),
            "network_target_point_count": int(config.network_input_points),
            "selected_downsampled_camera_path": saved_debug_path(completion["selected_downsampled_camera_path"]),
            "selected_downsampled_point_count": int(len(completion["selected_downsampled_camera_points"])),
            "selected_filtered_camera_path": saved_debug_path(completion["selected_filtered_camera_path"]),
            "selected_filtered_point_count": int(len(selected_filtered_camera_origin_points)),
            "selected_outlier_filter": completion["selected_outlier_filter"],
            "adapointr_input_camera_path": saved_debug_path(completion["adapointr_input_camera_path"]),
            "centered_partial_camera_mean": selected_points_camera_centered.mean(axis=0).astype(float).tolist(),
            "mirrored_partial_camera_mean": selected_points_camera_origin.mean(axis=0).astype(float).tolist(),
            "partial_camera_original_path": saved_debug_path(partial_camera_original_path),
            "partial_camera_centered_unflipped_path": saved_debug_path(partial_camera_centered_unflipped_path),
            "partial_camera_origin_path": saved_debug_path(completion["partial_camera_path"]),
            "network_input_camera_path": saved_debug_path(completion["network_input_camera_path"]),
            "network_input_camera_point_count": int(len(network_input_camera_origin_points)),
            "network_input_camera_original_path": saved_debug_path(network_input_camera_original_path),
            "network_input_world_path": saved_debug_path(network_input_world_path),
            "normalized_input_camera_path": saved_debug_path(completion["normalized_input_camera_path"]),
            "preprocess_meta_path": saved_debug_path(completion["preprocess_meta_path"]),
            "completed_camera_path": saved_debug_path(completion["completed_camera_path"]),
            "completed_camera_original_path": saved_debug_path(completed_camera_original_path),
            "completed_camera_npy_path": saved_debug_path(completion["completed_camera_npy_path"]),
        },
        "paths": {
            "partial_world_path": saved_debug_path(partial_world_path),
            "partial_camera_centered_unflipped_path": saved_debug_path(partial_camera_centered_unflipped_path),
            "partial_camera_origin_path": saved_debug_path(completion["partial_camera_path"]),
            "network_input_world_path": saved_debug_path(network_input_world_path),
            "completed_world_path": saved_debug_path(completed_world_path),
            "matching_target_camera_path": saved_debug_path(matching_target_camera_path),
            "matching_target_world_path": saved_debug_path(matching_target_world_path),
            "summary_path": str(summary_path),
        },
    }
    if matching is not None:
        summary["matching"] = {
            "template_path": str(matching["template_path"]),
            "template_original_point_count": int(matching["template_original_point_count"]),
            "template_point_count": int(matching["template_point_count"]),
            "target_point_count": int(matching["target_point_count"]),
            "target_source": "selected_downsampled_outlier_filtered_plus_completed",
            "selected_target_point_count": int(len(selected_filtered_camera_origin_points)),
            "completed_target_point_count": int(len(completed_camera_origin_points)),
            "target_camera_path": str(matching_target_camera_path),
            "target_world_path": str(matching_target_world_path),
            "voxel_size": float(matching["voxel_size"]),
            "template_voxel_size": matching["template_voxel_size"],
            "ransac_n": int(matching["ransac_n"]),
            "ransac_attempts": int(matching["ransac_attempts"]),
            "global_selected_attempt": int(matching["global_selected_attempt"]),
            "global_centroid_distance": float(matching["global_centroid_distance"]),
            "global_fitness": float(matching["global_fitness"]),
            "global_inlier_rmse": float(matching["global_inlier_rmse"]),
            "icp_fitness": float(matching["icp_fitness"]),
            "icp_inlier_rmse": float(matching["icp_inlier_rmse"]),
            "registration_frame": "camera_frame_minus_selected_center_camera_reflected_across_xy_plane",
            "world_projection": "world = cam_to_world @ translate(selected_center_camera) @ xy_plane_reflection @ registration_frame",
            "global_transform_path": str(matching["global_transform_path"]),
            "icp_transform_path": str(matching["icp_transform_path"]),
            "global_registered_path": str(matching["global_registered_path"]),
            "icp_registered_path": str(matching["icp_registered_path"]),
            "global_transform": matching["global_transform"].tolist(),
            "icp_transform": matching["icp_transform"].tolist(),
            "global_transform_world_path": str(matching["global_transform_world_path"]),
            "icp_transform_world_path": str(matching["icp_transform_world_path"]),
            "global_registered_world_path": str(matching["global_registered_world_path"]),
            "icp_registered_world_path": str(matching["icp_registered_world_path"]),
            "global_transform_world": matching["global_transform_world"].tolist(),
            "icp_transform_world": matching["icp_transform_world"].tolist(),
        }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return CompletionMatchingResult(
        completed_world_points=completed_world_points,
        transform=np.asarray(matching["icp_transform_world"], dtype=np.float64) if matching is not None else np.eye(4, dtype=np.float64),
        summary=summary,
        summary_path=summary_path,
    )

def config_from_args(args: argparse.Namespace) -> CompletionMatchingConfig:
    return CompletionMatchingConfig(
        adapointr_script=resolve_path(args.adapointr_script),
        adapointr_checkpoint=resolve_path(args.adapointr_checkpoint),
        full_template_ply=resolve_path(args.full_template_ply),
        output_dir=resolve_path(args.output_dir),
        cam_to_world=load_homomat(args.cam_to_world),
        device=args.device,
        global_scale=args.global_scale,
        num_points=args.num_points,
        num_query=args.num_query,
        voxel_size=args.voxel_size,
        template_voxel_size=args.template_voxel_size,
        ransac_n=args.ransac_n,
        ransac_attempts=args.ransac_attempts,
        icp_max_iteration=args.icp_max_iteration,
        network_input_points=args.network_input_points,
        selected_outlier_nb_neighbors=args.selected_outlier_nb_neighbors,
        selected_outlier_std_ratio=args.selected_outlier_std_ratio,
        selected_outlier_min_keep_ratio=args.selected_outlier_min_keep_ratio,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete a SAM-selected point cloud with AdaPoinTr and match a full template."
    )
    parser.add_argument("--sam-world-points", type=Path, required=True, help="SAM-selected object points in world frame (.ply or .npy).")
    parser.add_argument("--full-template-ply", type=Path, default=DEFAULT_FULL_TEMPLATE_PLY, help="Complete object template point cloud used as ICP source.")
    parser.add_argument("--adapointr-script", type=Path, default=DEFAULT_ADAPOINTR_SCRIPT, help="Path to infer_AdaPoinTr.py.")
    parser.add_argument("--adapointr-checkpoint", type=Path, default=DEFAULT_ADAPOINTR_CHECKPOINT, help="AdaPoinTr checkpoint.")
    parser.add_argument("--cam-to-world", type=Path, default=None, help="4x4 camera-to-world matrix. Defaults to yanjiuyuan.mech_eye_ur7e_pointcloud_env.CAM_TO_WORLD.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="sam_object")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--global-scale", type=float, default=0.4)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-query", type=int, default=128)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--template-voxel-size", type=float, default=0.005)
    parser.add_argument("--ransac-n", type=int, default=3)
    parser.add_argument("--ransac-attempts", type=int, default=5)
    parser.add_argument("--icp-max-iteration", type=int, default=80)
    parser.add_argument("--network-input-points", type=int, default=2048)
    parser.add_argument("--selected-outlier-nb-neighbors", type=int, default=24)
    parser.add_argument("--selected-outlier-std-ratio", type=float, default=1.8)
    parser.add_argument("--selected-outlier-min-keep-ratio", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    args = parse_args()
    selected_points_world = read_points(args.sam_world_points)
    result = run_completion_matching_on_sam_world_points(
        selected_points_world,
        config_from_args(args),
        output_prefix=args.output_prefix,
    )
    print(f"Completed world points: {len(result.completed_world_points)}")
    print(f"ICP fitness/rmse: {result.summary['matching']['icp_fitness']:.6f} / {result.summary['matching']['icp_inlier_rmse']:.6f}")
    print(f"Saved final world transform: {result.summary['matching']['icp_transform_world_path']}")
    print(f"Saved summary: {result.summary_path}")


if __name__ == "__main__":
    main()



