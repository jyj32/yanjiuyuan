"""
Load a saved Mech-Eye point cloud locally and register bottle.stl to it.

Run:
    python yanjiuyuan/bottle_icp_from_saved_capture.py

Adjust yanjiuyuan/bottle_icp_config.py instead of scattering constants in scripts.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yanjiuyuan.mech_eye_ur7e_pointcloud_env import (  # noqa: E402
    CAM_TO_WORLD,
    load_ply_pointcloud,
    preprocess_point_cloud_for_registration,
)
from yanjiuyuan import bottle_icp_config as bottle_cfg  # noqa: E402


BOTTLE_STL = bottle_cfg.BOTTLE_STL
CAPTURE_ROOT = bottle_cfg.CAPTURE_ROOT
CAPTURE_DIR = bottle_cfg.CAPTURE_DIR
CAPTURE_PLY = bottle_cfg.CAPTURE_PLY
PREFER_WORLD_FRAME_PLY = bottle_cfg.PREFER_WORLD_FRAME_PLY
TRANSFORM_CAMERA_PLY_TO_WORLD = bottle_cfg.TRANSFORM_CAMERA_PLY_TO_WORLD
X_RANGE = bottle_cfg.X_RANGE
Y_RANGE = bottle_cfg.Y_RANGE
Z_RANGE = bottle_cfg.Z_RANGE
VOXEL_SIZE = bottle_cfg.VOXEL_SIZE
MODEL_SAMPLE_COUNT = bottle_cfg.MODEL_SAMPLE_COUNT
MODEL_EVEN_RADIUS = bottle_cfg.MODEL_EVEN_RADIUS
GLOBAL_RANSAC_N = bottle_cfg.GLOBAL_RANSAC_N
ICP_MAX_ITERATION = bottle_cfg.ICP_MAX_ITERATION
OUTPUT_DIR = bottle_cfg.OUTPUT_DIR
SHOW_RESULT_VIEWER = bottle_cfg.SHOW_RESULT_VIEWER
TARGET_POINT_SIZE = bottle_cfg.TARGET_POINT_SIZE
MODEL_POINT_SIZE = bottle_cfg.MODEL_POINT_SIZE

Range3D = Optional[Tuple[Optional[float], Optional[float]]]


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


def crop_pointcloud_by_range(
    pcd,
    x_range: Range3D = X_RANGE,
    y_range: Range3D = Y_RANGE,
    z_range: Range3D = Z_RANGE,
):
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


def numpy_to_open3d_pointcloud(points: np.ndarray, normals: Optional[np.ndarray] = None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    return pcd


def sample_bottle_surface_pointcloud(
    bottle_stl: Path = BOTTLE_STL,
    sample_count: Optional[int] = MODEL_SAMPLE_COUNT,
    even_radius: Optional[float] = MODEL_EVEN_RADIUS,
):
    import wrs.modeling.geometric_model as mgm

    bottle_stl = resolve_path(bottle_stl)
    if not bottle_stl.exists():
        raise FileNotFoundError(f"Bottle STL does not exist: {bottle_stl}")

    model = mgm.GeometricModel(initor=str(bottle_stl), name=bottle_stl.stem)
    if model.trm_mesh is None:
        raise RuntimeError(f"Unable to load triangle mesh from: {bottle_stl}")

    n_samples = None if sample_count is None or sample_count <= 0 else sample_count
    if n_samples is None and even_radius is None:
        raise ValueError("Set MODEL_SAMPLE_COUNT > 0, or set MODEL_EVEN_RADIUS when MODEL_SAMPLE_COUNT <= 0.")

    points, normals = model.sample_surface(radius=even_radius, n_samples=n_samples, toggle_option="normals")
    return numpy_to_open3d_pointcloud(points, normals=normals), bottle_stl


def run_global_registration(
    source_pcd,
    target_pcd,
    voxel_size: float = VOXEL_SIZE,
    ransac_n: int = GLOBAL_RANSAC_N,
):
    import open3d as o3d

    reg = o3d.pipelines.registration
    source_down, source_fpfh = preprocess_point_cloud_for_registration(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud_for_registration(target_pcd, voxel_size)

    distance_threshold = voxel_size * 1.5
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


def run_point_to_plane_icp(source_down, target_down, init_transform: np.ndarray, voxel_size: float = VOXEL_SIZE):
    import open3d as o3d

    reg = o3d.pipelines.registration
    return reg.registration_icp(
        source_down,
        target_down,
        voxel_size * 0.7,
        init_transform,
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=ICP_MAX_ITERATION,
        ),
    )


def write_registration_outputs(
    output_dir: Path,
    source_pcd,
    target_pcd,
    global_source_pcd,
    refined_source_pcd,
    global_result,
    icp_result,
    target_ply: Path,
    target_frame: str,
    bottle_stl: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    global_transform_path = output_dir / "bottle_global_transform.txt"
    icp_transform_path = output_dir / "bottle_global_icp_transform.txt"
    summary_path = output_dir / "bottle_icp_summary.txt"

    np.savetxt(global_transform_path, global_result.transformation, fmt="%.9f")
    np.savetxt(icp_transform_path, icp_result.transformation, fmt="%.9f")

    summary = [
        f"bottle_stl: {bottle_stl}",
        f"target_ply: {target_ply}",
        f"target_frame: {target_frame}",
        f"voxel_size: {VOXEL_SIZE}",
        f"model_sample_count: {MODEL_SAMPLE_COUNT}",
        f"model_even_radius: {MODEL_EVEN_RADIUS}",
        f"x_range: {X_RANGE}",
        f"y_range: {Y_RANGE}",
        f"z_range: {Z_RANGE}",
        f"target_points: {len(target_pcd.points)}",
        f"source_points: {len(source_pcd.points)}",
        f"global_fitness: {global_result.fitness:.9f}",
        f"global_inlier_rmse: {global_result.inlier_rmse:.9f}",
        f"icp_fitness: {icp_result.fitness:.9f}",
        f"icp_inlier_rmse: {icp_result.inlier_rmse:.9f}",
        f"global_transform: {global_transform_path}",
        f"icp_transform: {icp_transform_path}",
        "pointcloud_outputs_saved: false",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    return {
        "source_path": None,
        "target_path": None,
        "global_registered_path": None,
        "icp_registered_path": None,
        "global_transform_path": global_transform_path,
        "icp_transform_path": icp_transform_path,
        "summary_path": summary_path,
        "pointcloud_outputs_saved": False,
    }


def show_registration_result(target_pcd, global_source_pcd, refined_source_pcd):
    from wrs import mgm, wd

    target_points = np.asarray(target_pcd.points)
    if len(target_points) == 0:
        return
    center = (target_points.min(axis=0) + target_points.max(axis=0)) / 2.0
    extent = max(float(np.max(target_points.max(axis=0) - target_points.min(axis=0))), 1e-6)
    cam_dist = max(0.35, extent * 2.8)
    cam_pos = center + np.array([cam_dist, -cam_dist, max(cam_dist * 0.65, extent * 0.9)])

    base = wd.World(cam_pos=cam_pos, lookat_pos=center, w=1280, h=720)
    mgm.gen_frame(ax_length=max(extent * 0.35, 0.03), ax_radius=0.001).attach_to(base)
    mgm.gen_pointcloud(target_points, rgba=np.array([0.05, 0.45, 1.0, 0.45]), point_size=TARGET_POINT_SIZE).attach_to(base)
    mgm.gen_pointcloud(np.asarray(global_source_pcd.points), rgba=np.array([1.0, 0.75, 0.0, 0.85]), point_size=MODEL_POINT_SIZE).attach_to(base)
    mgm.gen_pointcloud(np.asarray(refined_source_pcd.points), rgba=np.array([1.0, 0.05, 0.02, 1.0]), point_size=MODEL_POINT_SIZE).attach_to(base)
    print("Viewer colors: target=blue, global=yellow, ICP refined=red")
    base.run()


def main() -> None:
    target_pcd, target_ply, capture_dir, target_frame = load_saved_capture_pointcloud()
    target_pcd = crop_pointcloud_by_range(target_pcd)
    source_pcd, bottle_stl = sample_bottle_surface_pointcloud()

    global_result, source_down, target_down = run_global_registration(source_pcd, target_pcd)
    icp_result = run_point_to_plane_icp(source_down, target_down, global_result.transformation, voxel_size=VOXEL_SIZE)

    global_source_pcd = None
    refined_source_pcd = None
    if SHOW_RESULT_VIEWER:
        global_source_pcd = transform_open3d_pointcloud(source_pcd, global_result.transformation)
        refined_source_pcd = transform_open3d_pointcloud(source_pcd, icp_result.transformation)

    output_dir = resolve_path(OUTPUT_DIR) if OUTPUT_DIR is not None else capture_dir / "bottle_icp"
    paths = write_registration_outputs(
        output_dir=output_dir,
        source_pcd=source_pcd,
        target_pcd=target_pcd,
        global_source_pcd=global_source_pcd,
        refined_source_pcd=refined_source_pcd,
        global_result=global_result,
        icp_result=icp_result,
        target_ply=target_ply,
        target_frame=target_frame,
        bottle_stl=bottle_stl,
    )

    print(f"Loaded target: {target_ply} ({target_frame})")
    print(f"Loaded bottle STL: {bottle_stl}")
    print(f"Target points: {len(target_pcd.points)}")
    print(f"Source points: {len(source_pcd.points)}")
    print(f"Global fitness/rmse: {global_result.fitness:.6f} / {global_result.inlier_rmse:.6f}")
    print(f"ICP fitness/rmse: {icp_result.fitness:.6f} / {icp_result.inlier_rmse:.6f}")
    print(f"Saved ICP transform: {paths['icp_transform_path']}")
    print(f"Saved summary: {paths['summary_path']}")

    if SHOW_RESULT_VIEWER:
        show_registration_result(target_pcd, global_source_pcd, refined_source_pcd)


if __name__ == "__main__":
    main()