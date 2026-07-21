"""
Prepare a saved Mech-Eye bottle point cloud for camera-frame completion.

Workflow:
    1. Load a saved capture from yanjiuyuan/captures.
    2. Convert it to the world frame.
    3. Remove the tabletop in the world frame.
    4. Transform the remaining points back to the camera frame.
    5. Run the PCN camera-frame completion network.
    6. Save the completed point cloud in the camera frame.

Run:
    python yanjiuyuan/bottle_completion_from_saved_capture.py

Adjust the constants below instead of passing command-line parameters.
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
    DEFAULT_OUTPUT_ROOT,
    load_ply_pointcloud,
)


CAPTURE_ROOT = DEFAULT_OUTPUT_ROOT
PCN_TRAIN_DIR = REPO_ROOT / "poind_cloud_completion" / "pcn_train"
PCN_CHECKPOINT = PCN_TRAIN_DIR / "log" / "train_exp_dy_aug_4000_camera" / "checkpoints" / "best_combo.pth"

# If CAPTURE_DIR is None, the newest folder under CAPTURE_ROOT is used.
# You may also set CAPTURE_PLY directly to a specific .ply file.
CAPTURE_DIR = None
CAPTURE_PLY = None

# If CAPTURE_PLY is a custom file name, set this to "camera" or "world".
# Otherwise the script infers the frame from standard saved-capture names.
CAPTURE_PLY_FRAME = None
PREFER_WORLD_FRAME_PLY = True

# Tabletop removal is done in the world frame. Z_RANGE=(0.04, None) matches
# bottle_icp_from_saved_capture.py and removes points near the table plane.
WORLD_X_RANGE = None
WORLD_Y_RANGE = None
WORLD_Z_RANGE = (0.04, None)

# Optional cleanup before the completion network.
VOXEL_DOWNSAMPLE_SIZE = None
REMOVE_STATISTICAL_OUTLIERS = False
OUTLIER_NB_NEIGHBORS = 30
OUTLIER_STD_RATIO = 2.0

# PCN camera-frame inference settings. These mirror
# poind_cloud_completion/pcn_train/infer_dy_camera.py.
PCN_INPUT_POINT_COUNT = 2048
PCN_DEVICE = "cuda:0"
PCN_USE_DENSE_OUTPUT_AS_COMPLETED = True
SAVE_COMPLETED_WORLD_DEBUG = True
RANDOM_SEED = 7

# Output and optional debug visualization.
OUTPUT_DIR = None
SHOW_DEBUG_VIEWER = False
SHOW_SIM_VIEWER = True
SHOW_SIM_INPUT_POINTS = True
SHOW_TABLE_REMOVED_WORLD_POINTS = True
RUN_MIRRORED_XY_COMPLETION = True
RUN_ORIGINAL_COMPLETION = False
POINT_SIZE = 0.001
COMPLETED_POINT_SIZE = 0.001
RUN_COMPLETION_NETWORK = True


Range3D = Optional[Tuple[Optional[float], Optional[float]]]
WORLD_TO_CAM = np.linalg.inv(CAM_TO_WORLD)
CAMERA_XY_MIRROR = np.diag([1.0, 1.0, -1.0, 1.0])


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def transform_open3d_pointcloud(pcd, homomat: np.ndarray):
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(homomat)
    return pcd_copy


def mirror_camera_pointcloud_across_xy(pcd):
    return transform_open3d_pointcloud(pcd, CAMERA_XY_MIRROR)


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


def infer_capture_frame(ply_path: Path, frame_override: Optional[str] = CAPTURE_PLY_FRAME) -> str:
    if frame_override is not None:
        frame = frame_override.lower()
        if frame not in {"camera", "world"}:
            raise ValueError('CAPTURE_PLY_FRAME must be None, "camera", or "world".')
        return frame
    if ply_path.name == "world_colored_pointcloud.ply":
        return "world"
    return "camera"


def resolve_capture_ply(
    capture_dir: Optional[Path] = CAPTURE_DIR,
    capture_ply: Optional[Path] = CAPTURE_PLY,
    prefer_world_frame: bool = PREFER_WORLD_FRAME_PLY,
) -> Tuple[Path, Path, str]:
    if capture_ply is not None:
        ply_path = resolve_path(capture_ply)
        if not ply_path.exists():
            raise FileNotFoundError(f"Capture PLY does not exist: {ply_path}")
        return ply_path, ply_path.parent, infer_capture_frame(ply_path)

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


def load_saved_capture_as_world():
    ply_path, capture_dir, source_frame = resolve_capture_ply()
    pcd = load_ply_pointcloud(ply_path)
    if source_frame == "camera":
        pcd_world = transform_open3d_pointcloud(pcd, CAM_TO_WORLD)
        world_frame = "world_from_camera"
    else:
        pcd_world = pcd
        world_frame = "world"
    return pcd_world, ply_path, capture_dir, source_frame, world_frame


def crop_pointcloud_by_world_range(
    pcd,
    x_range: Range3D = WORLD_X_RANGE,
    y_range: Range3D = WORLD_Y_RANGE,
    z_range: Range3D = WORLD_Z_RANGE,
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
        raise RuntimeError("Point cloud is empty after world-frame tabletop removal.")
    return pcd.select_by_index(indices)


def clean_partial_pointcloud(pcd):
    if VOXEL_DOWNSAMPLE_SIZE is not None and VOXEL_DOWNSAMPLE_SIZE > 0:
        pcd = pcd.voxel_down_sample(VOXEL_DOWNSAMPLE_SIZE)
    if REMOVE_STATISTICAL_OUTLIERS and len(pcd.points) > OUTLIER_NB_NEIGHBORS:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS,
            std_ratio=OUTLIER_STD_RATIO,
        )
    if len(pcd.points) == 0:
        raise RuntimeError("Point cloud is empty after cleanup.")
    return pcd


def numpy_to_open3d_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0))
    return pcd


def farthest_point_sample(points: np.ndarray, n_sample: int, random_seed: int = RANDOM_SEED) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    point_count = points.shape[0]
    if point_count <= n_sample:
        return points

    rng = np.random.default_rng(random_seed)
    distances = np.full(point_count, np.inf)
    sampled_idx = np.zeros(n_sample, dtype=np.int64)
    farthest = int(rng.integers(0, point_count))

    for i in range(n_sample):
        sampled_idx[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))

    return points[sampled_idx]


def resample_pointcloud_for_pcn(
    pcd,
    point_count: int = PCN_INPUT_POINT_COUNT,
    random_seed: int = RANDOM_SEED,
):
    if point_count is None or point_count <= 0:
        return pcd

    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise RuntimeError("Cannot resample an empty point cloud.")

    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    if len(points) > point_count:
        sampled_points = farthest_point_sample(points, point_count, random_seed=random_seed)
        sampled_colors = None
    else:
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(len(points), size=point_count, replace=True)
        sampled_points = points[indices]
        sampled_colors = colors[indices] if colors is not None else None
    return numpy_to_open3d_pointcloud(sampled_points, colors=sampled_colors)


def open3d_to_numpy(pcd):
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)
    return points, colors


def write_frontend_outputs(
    output_dir: Path,
    table_removed_world_pcd,
    completion_input_camera_pcd,
    capture_ply: Path,
    source_frame: str,
    world_frame: str,
    raw_world_count: int,
):
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    world_debug_path = output_dir / "bottle_table_removed_world_points.ply"
    camera_input_path = output_dir / "bottle_completion_input_camera_points.ply"
    camera_npy_path = output_dir / "bottle_completion_input_camera_points.npy"
    camera_npz_path = output_dir / "bottle_completion_input_camera_points.npz"
    cam_to_world_path = output_dir / "camera_to_world.txt"
    world_to_cam_path = output_dir / "world_to_camera.txt"
    summary_path = output_dir / "bottle_completion_frontend_summary.txt"

    o3d.io.write_point_cloud(str(world_debug_path), table_removed_world_pcd, write_ascii=False)
    o3d.io.write_point_cloud(str(camera_input_path), completion_input_camera_pcd, write_ascii=False)
    points, colors = open3d_to_numpy(completion_input_camera_pcd)
    np.save(camera_npy_path, points)
    if colors is None:
        np.savez_compressed(camera_npz_path, points=points)
    else:
        np.savez_compressed(camera_npz_path, points=points, colors=colors)
    np.savetxt(cam_to_world_path, CAM_TO_WORLD, fmt="%.9f")
    np.savetxt(world_to_cam_path, WORLD_TO_CAM, fmt="%.9f")

    summary = [
        f"capture_ply: {capture_ply}",
        f"source_frame: {source_frame}",
        f"world_frame_for_table_removal: {world_frame}",
        f"raw_world_points: {raw_world_count}",
        f"table_removed_world_points: {len(table_removed_world_pcd.points)}",
        f"completion_input_camera_points: {len(completion_input_camera_pcd.points)}",
        f"world_x_range: {WORLD_X_RANGE}",
        f"world_y_range: {WORLD_Y_RANGE}",
        f"world_z_range: {WORLD_Z_RANGE}",
        f"voxel_downsample_size: {VOXEL_DOWNSAMPLE_SIZE}",
        f"remove_statistical_outliers: {REMOVE_STATISTICAL_OUTLIERS}",
        f"outlier_nb_neighbors: {OUTLIER_NB_NEIGHBORS}",
        f"outlier_std_ratio: {OUTLIER_STD_RATIO}",
        f"pcn_input_point_count: {PCN_INPUT_POINT_COUNT}",
        f"world_debug_ply: {world_debug_path}",
        f"camera_input_ply: {camera_input_path}",
        f"camera_input_npy: {camera_npy_path}",
        f"camera_input_npz: {camera_npz_path}",
        f"camera_to_world: {cam_to_world_path}",
        f"world_to_camera: {world_to_cam_path}",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    return {
        "world_debug_path": world_debug_path,
        "camera_input_path": camera_input_path,
        "camera_npy_path": camera_npy_path,
        "camera_npz_path": camera_npz_path,
        "cam_to_world_path": cam_to_world_path,
        "world_to_cam_path": world_to_cam_path,
        "summary_path": summary_path,
    }


def preprocess_for_pcn(points: np.ndarray) -> Tuple[np.ndarray, float]:
    scale = float(np.max(np.linalg.norm(points, axis=1)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("Invalid PCN normalization scale computed from camera-frame points.")
    return points / scale, scale


def load_pcn_model(checkpoint_path: Path, device_name: str = PCN_DEVICE):
    import torch

    pcn_train_dir = resolve_path(PCN_TRAIN_DIR)
    if str(pcn_train_dir) not in sys.path:
        sys.path.insert(0, str(pcn_train_dir))
    from models.pcn_dynamic_frame import PCNDynamicFrame

    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PCN checkpoint does not exist: {checkpoint_path}")

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: {device_name} requested but CUDA is not available; using cpu.")
        device_name = "cpu"
    device = torch.device(device_name)

    model = PCNDynamicFrame().to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model, device, checkpoint_path


def write_camera_points(points: np.ndarray, ply_path: Path, npy_path: Optional[Path] = None):
    import open3d as o3d

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = numpy_to_open3d_pointcloud(np.asarray(points, dtype=np.float64))
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)
    if npy_path is not None:
        np.save(npy_path, np.asarray(points, dtype=np.float32))
    return pcd


def run_point_completion_network(
    partial_camera_pcd,
    output_dir: Path,
    checkpoint_path: Path = PCN_CHECKPOINT,
    device_name: str = PCN_DEVICE,
    output_prefix: str = "bottle_completed",
):
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    input_points, _ = open3d_to_numpy(partial_camera_pcd)
    if input_points.shape[0] != PCN_INPUT_POINT_COUNT:
        raise RuntimeError(
            f"PCN input point count is {input_points.shape[0]}, expected {PCN_INPUT_POINT_COUNT}. "
            "Call resample_pointcloud_for_pcn before inference."
        )

    model, device, checkpoint_path = load_pcn_model(checkpoint_path, device_name=device_name)
    points_norm, scale = preprocess_for_pcn(input_points)
    input_tensor = torch.from_numpy(points_norm.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        coarse_pred, dense_pred = model(input_tensor)

    coarse_points = coarse_pred.squeeze(0).detach().cpu().numpy() * scale
    dense_points = dense_pred.squeeze(0).detach().cpu().numpy() * scale
    completed_points = dense_points if PCN_USE_DENSE_OUTPUT_AS_COMPLETED else coarse_points

    coarse_ply_path = output_dir / f"{output_prefix}_coarse_camera_points.ply"
    coarse_npy_path = output_dir / f"{output_prefix}_coarse_camera_points.npy"
    dense_ply_path = output_dir / f"{output_prefix}_dense_camera_points.ply"
    dense_npy_path = output_dir / f"{output_prefix}_dense_camera_points.npy"
    completed_ply_path = output_dir / f"{output_prefix}_camera_points.ply"
    completed_npy_path = output_dir / f"{output_prefix}_camera_points.npy"
    world_debug_path = output_dir / f"{output_prefix}_world_debug_points.ply"
    summary_path = output_dir / f"{output_prefix}_network_summary.txt"

    write_camera_points(coarse_points, coarse_ply_path, coarse_npy_path)
    write_camera_points(dense_points, dense_ply_path, dense_npy_path)
    completed_pcd = write_camera_points(completed_points, completed_ply_path, completed_npy_path)

    if SAVE_COMPLETED_WORLD_DEBUG:
        import open3d as o3d

        completed_world_pcd = transform_open3d_pointcloud(completed_pcd, CAM_TO_WORLD)
        o3d.io.write_point_cloud(str(world_debug_path), completed_world_pcd, write_ascii=False)

    summary = [
        "infer_reference: poind_cloud_completion/pcn_train/infer_dy_camera.py",
        f"checkpoint: {checkpoint_path}",
        f"device: {device}",
        f"pcn_input_points: {input_points.shape[0]}",
        f"normalization_scale: {scale:.9f}",
        f"coarse_points: {coarse_points.shape[0]}",
        f"dense_points: {dense_points.shape[0]}",
        f"use_dense_output_as_completed: {PCN_USE_DENSE_OUTPUT_AS_COMPLETED}",
        f"completed_camera_ply: {completed_ply_path}",
        f"completed_camera_npy: {completed_npy_path}",
        f"coarse_camera_ply: {coarse_ply_path}",
        f"dense_camera_ply: {dense_ply_path}",
        f"completed_world_debug_ply: {world_debug_path if SAVE_COMPLETED_WORLD_DEBUG else None}",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    return {
        "completed_camera_ply": completed_ply_path,
        "completed_camera_npy": completed_npy_path,
        "coarse_camera_ply": coarse_ply_path,
        "coarse_camera_npy": coarse_npy_path,
        "dense_camera_ply": dense_ply_path,
        "dense_camera_npy": dense_npy_path,
        "completed_world_debug_ply": world_debug_path if SAVE_COMPLETED_WORLD_DEBUG else None,
        "summary_path": summary_path,
        "scale": scale,
        "device": str(device),
        "completed_camera_pcd": completed_pcd,
    }


def compute_view_from_world_points(*point_arrays: np.ndarray):
    valid_arrays = [points for points in point_arrays if points is not None and len(points) > 0]
    if not valid_arrays:
        return np.array([2.0, -1.6, 1.2]), np.array([0.4, -0.25, 0.3]), 0.25

    points = np.vstack(valid_arrays)
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    extent = max(float(np.max(points.max(axis=0) - points.min(axis=0))), 1e-6)
    cam_dist = max(0.65, extent * 3.0)
    cam_pos = center + np.array([cam_dist, -cam_dist, max(cam_dist * 0.7, extent * 1.1)])
    return cam_pos, center, extent


def describe_points(label: str, points: np.ndarray) -> str:
    points = np.asarray(points)
    if len(points) == 0:
        return f"{label}: empty"
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    return (
        f"{label}: count={len(points)}, "
        f"min={np.array2string(min_corner, precision=4)}, "
        f"max={np.array2string(max_corner, precision=4)}, "
        f"center={np.array2string(center, precision=4)}"
    )


def show_completion_in_camera_frame(
    table_removed_world_pcd,
    partial_camera_pcd,
    completed_camera_pcd=None,
    mirrored_partial_camera_pcd=None,
    mirrored_completed_camera_pcd=None,
):
    from wrs import mgm, wd

    display_completed_pcd = mirrored_completed_camera_pcd if mirrored_completed_camera_pcd is not None else completed_camera_pcd
    display_input_pcd = mirrored_partial_camera_pcd if mirrored_partial_camera_pcd is not None else partial_camera_pcd
    if display_completed_pcd is None:
        print("No completed point cloud available for display.")
        return

    input_points = np.asarray(display_input_pcd.points)
    completed_points = np.asarray(display_completed_pcd.points)
    cam_pos, lookat_pos, extent = compute_view_from_world_points(input_points, completed_points)
    base = wd.World(cam_pos=cam_pos, lookat_pos=lookat_pos, w=1280, h=720)
    mgm.gen_frame(ax_length=max(extent * 0.35, 0.05), ax_radius=0.0015).attach_to(base)

    mgm.gen_pointcloud(
        input_points,
        rgba=np.array([0.72, 0.0, 1.0, 0.85]),
        point_size=POINT_SIZE,
    ).attach_to(base)
    mgm.gen_pointcloud(
        completed_points,
        rgba=np.array([1.0, 0.62, 0.0, 0.95]),
        point_size=COMPLETED_POINT_SIZE,
    ).attach_to(base)

    input_label = "xy_mirrored_input_camera" if mirrored_partial_camera_pcd is not None else "pcn_input_camera"
    completed_label = "xy_mirrored_completed_camera" if mirrored_completed_camera_pcd is not None else "completed_camera"
    print("Camera-frame viewer colors: purple=downsampled PCN input, orange=completion result")
    print(describe_points(input_label, input_points))
    print(describe_points(completed_label, completed_points))
    base.run()


def show_debug_result(table_removed_world_pcd, completion_input_camera_pcd):
    from wrs import mgm, wd

    world_points = np.asarray(table_removed_world_pcd.points)
    if len(world_points) == 0:
        return
    camera_points_in_world = transform_open3d_pointcloud(completion_input_camera_pcd, CAM_TO_WORLD)
    input_world_points = np.asarray(camera_points_in_world.points)
    cam_pos, lookat_pos, extent = compute_view_from_world_points(world_points, input_world_points)

    base = wd.World(cam_pos=cam_pos, lookat_pos=lookat_pos, w=1280, h=720)
    mgm.gen_frame(ax_length=max(extent * 0.35, 0.03), ax_radius=0.001).attach_to(base)
    mgm.gen_pointcloud(world_points, rgba=np.array([0.05, 0.45, 1.0, 0.55]), point_size=POINT_SIZE).attach_to(base)
    mgm.gen_pointcloud(
        input_world_points,
        rgba=np.array([1.0, 0.1, 0.02, 0.8]),
        point_size=POINT_SIZE,
    ).attach_to(base)
    print("Viewer colors: blue=table-removed world points, red=camera-frame completion input transformed back for checking")
    base.run()


def main() -> None:
    table_scene_world_pcd, capture_ply, capture_dir, source_frame, world_frame = load_saved_capture_as_world()
    raw_world_count = len(table_scene_world_pcd.points)

    table_removed_world_pcd = crop_pointcloud_by_world_range(table_scene_world_pcd)
    table_removed_world_pcd = clean_partial_pointcloud(table_removed_world_pcd)
    table_removed_camera_pcd = transform_open3d_pointcloud(table_removed_world_pcd, WORLD_TO_CAM)
    completion_input_camera_pcd = resample_pointcloud_for_pcn(table_removed_camera_pcd)

    output_dir = resolve_path(OUTPUT_DIR) if OUTPUT_DIR is not None else capture_dir / "bottle_completion_input"
    paths = write_frontend_outputs(
        output_dir=output_dir,
        table_removed_world_pcd=table_removed_world_pcd,
        completion_input_camera_pcd=completion_input_camera_pcd,
        capture_ply=capture_ply,
        source_frame=source_frame,
        world_frame=world_frame,
        raw_world_count=raw_world_count,
    )

    print(f"Loaded capture: {capture_ply} ({source_frame})")
    print(f"World frame used for table removal: {world_frame}")
    print(f"Raw world points: {raw_world_count}")
    print(f"Table-removed world points: {len(table_removed_world_pcd.points)}")
    print(f"Camera-frame completion input points: {len(completion_input_camera_pcd.points)}")
    print(f"Saved camera-frame PLY: {paths['camera_input_path']}")
    print(f"Saved camera-frame NPY: {paths['camera_npy_path']}")
    print(f"Saved camera-frame NPZ: {paths['camera_npz_path']}")
    print(f"Saved summary: {paths['summary_path']}")

    mirrored_completion_input_camera_pcd = None
    mirrored_completion_paths = None
    if RUN_MIRRORED_XY_COMPLETION:
        mirrored_completion_input_camera_pcd = mirror_camera_pointcloud_across_xy(completion_input_camera_pcd)
        mirrored_input_path = output_dir / "bottle_completion_input_xy_mirrored_camera_points.ply"
        mirrored_input_npy_path = output_dir / "bottle_completion_input_xy_mirrored_camera_points.npy"
        write_camera_points(np.asarray(mirrored_completion_input_camera_pcd.points), mirrored_input_path, mirrored_input_npy_path)
        print(f"Saved xy-mirrored camera-frame input PLY: {mirrored_input_path}")
        print(f"Saved xy-mirrored camera-frame input NPY: {mirrored_input_npy_path}")

    completion_paths = None
    if RUN_COMPLETION_NETWORK:
        if RUN_ORIGINAL_COMPLETION:
            completion_paths = run_point_completion_network(completion_input_camera_pcd, output_dir)
            print(f"PCN device: {completion_paths['device']}")
            print(f"PCN normalization scale: {completion_paths['scale']:.6f}")
            print(f"Saved completed camera-frame PLY: {completion_paths['completed_camera_ply']}")
            print(f"Saved completed camera-frame NPY: {completion_paths['completed_camera_npy']}")
            print(f"Saved PCN summary: {completion_paths['summary_path']}")

        if mirrored_completion_input_camera_pcd is not None:
            mirrored_completion_paths = run_point_completion_network(
                mirrored_completion_input_camera_pcd,
                output_dir,
                output_prefix="bottle_xy_mirrored_completed",
            )
            print(f"PCN device: {mirrored_completion_paths['device']}")
            print(f"XY-mirrored PCN normalization scale: {mirrored_completion_paths['scale']:.6f}")
            print(f"Saved xy-mirrored completed camera-frame PLY: {mirrored_completion_paths['completed_camera_ply']}")
            print(f"Saved xy-mirrored completed camera-frame NPY: {mirrored_completion_paths['completed_camera_npy']}")
            print(f"Saved xy-mirrored PCN summary: {mirrored_completion_paths['summary_path']}")

    if SHOW_SIM_VIEWER and (completion_paths is not None or mirrored_completion_paths is not None):
        show_completion_in_camera_frame(
            table_removed_world_pcd,
            completion_input_camera_pcd,
            completion_paths["completed_camera_pcd"] if completion_paths is not None else None,
            mirrored_completion_input_camera_pcd,
            mirrored_completion_paths["completed_camera_pcd"] if mirrored_completion_paths is not None else None,
        )

    if SHOW_DEBUG_VIEWER:
        show_debug_result(table_removed_world_pcd, completion_input_camera_pcd)


if __name__ == "__main__":
    main()