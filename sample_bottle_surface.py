"""
Generate bottle template point clouds from bottle.stl.

Outputs:
  1. Full sampled surface template.
  2. VirtualDepthCamera top-view visible template.
  3. VirtualDepthCamera front-view visible template.
  4. VirtualDepthCamera left/right-view visible templates.

Default usage:
    python yanjiuyuan/sample_bottle_surface.py

Adjust the constants below to change model path, point counts, virtual camera
settings, output paths, and display settings.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "bottle.stl"
MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_OUTPUT = MODEL_DIR / "bottle_surface_points.ply"
VIEW_OUTPUTS = {
    "top": MODEL_DIR / "bottle_top_view_points.ply",
    "front": MODEL_DIR / "bottle_front_view_points.ply",
    "left": MODEL_DIR / "bottle_left_view_points.ply",
    "right": MODEL_DIR / "bottle_right_view_points.ply",
}

MODEL_PATH = DEFAULT_MODEL
OUTPUT_PLY = DEFAULT_OUTPUT

# Sampling settings. Set EVEN_RADIUS to a float such as 0.002 for WRS even sampling.
# EVEN_RADIUS=None returns exactly N_SAMPLES points.
GENERATE_SURFACE_TEMPLATE = True
N_SAMPLES = 10000
EVEN_RADIUS = None
INCLUDE_NORMALS = True
CENTER_BOTTOM = False

# VirtualDepthCamera visible-surface templates.
GENERATE_VIEW_TEMPLATES = True
VIEW_TEMPLATE_NAMES = ("top", "front", "left", "right")
DEPTH_RESOLUTION = (640, 640)
DEPTH_FOV = 35
DEPTH_CAMERA_DISTANCE_FACTOR = 3.0
DEPTH_MIN_CAMERA_DISTANCE = 0.30
DEPTH_FAR_MARGIN_FACTOR = 4.0
VIEW_TEMPLATE_BBOX_MARGIN = 0.005
VIEW_TEMPLATE_VOXEL_SIZE = 0.0015

# Display settings.
SHOW_VIEWER = False
MODEL_RGB = np.array([0.42, 0.62, 0.95])
MODEL_ALPHA = 0.25
POINT_RGBA = np.array([1.0, 0.08, 0.02, 1.0])
VIEW_POINT_RGBA = {
    "top": np.array([0.0, 0.75, 1.0, 1.0]),
    "front": np.array([0.0, 0.85, 0.2, 1.0]),
    "left": np.array([1.0, 0.70, 0.0, 1.0]),
    "right": np.array([0.85, 0.25, 1.0, 1.0]),
}
POINT_SIZE = 0.002
SHOW_MODEL_WIREFRAME = False
SHOW_WORLD_FRAME = True
AUTO_ROTATE = False
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


@dataclass(frozen=True)
class ViewCameraSpec:
    name: str
    cam_pos: np.ndarray
    lookat_pos: np.ndarray
    up: np.ndarray


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def center_model_xy_and_bottom(model) -> None:
    bounds = np.asarray(model.trm_mesh.bounds, dtype=float)
    center = (bounds[0] + bounds[1]) / 2.0
    model.pos = np.array([-center[0], -center[1], -bounds[0, 2]], dtype=float)


def load_geometric_model(
    model_path: Path,
    rgb: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    center_bottom: bool = False,
):
    import wrs.modeling.geometric_model as mgm

    model_path = resolve_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    model = mgm.GeometricModel(
        initor=str(model_path),
        name=model_path.stem,
        toggle_twosided=True,
        rgb=rgb,
        alpha=alpha,
    )
    if model.trm_mesh is None:
        raise RuntimeError(f"Unable to load a triangle mesh from: {model_path}")
    if center_bottom:
        center_model_xy_and_bottom(model)
    return model, model_path


def write_points_to_ply(points: np.ndarray, ply_path: Path, normals: Optional[np.ndarray] = None) -> None:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")

    if normals is not None:
        normals = np.asarray(normals, dtype=float)
        if normals.shape != points.shape:
            raise ValueError("normals must have the same shape as points")

    ply_path = resolve_path(ply_path)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with ply_path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        if normals is not None:
            file.write("property float nx\n")
            file.write("property float ny\n")
            file.write("property float nz\n")
        file.write("end_header\n")

        if normals is None:
            for point in points:
                file.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
        else:
            for point, normal in zip(points, normals):
                file.write(
                    f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                    f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n"
                )


def sample_surface_from_model(
    model,
    n_samples: Optional[int] = N_SAMPLES,
    even_radius: Optional[float] = EVEN_RADIUS,
    include_normals: bool = INCLUDE_NORMALS,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Sample surface points with WRS GeometricModel.sample_surface."""
    sample_count = None if n_samples is None or n_samples <= 0 else n_samples
    if sample_count is None and even_radius is None:
        raise ValueError("Set N_SAMPLES to a positive value, or set EVEN_RADIUS when N_SAMPLES <= 0.")

    if include_normals:
        points, normals = model.sample_surface(radius=even_radius, n_samples=sample_count, toggle_option="normals")
    else:
        points = model.sample_surface(radius=even_radius, n_samples=sample_count)
        normals = None
    return points, normals


def sample_surface_points(
    model_path: Path,
    output_path: Path,
    n_samples: Optional[int] = N_SAMPLES,
    even_radius: Optional[float] = EVEN_RADIUS,
    include_normals: bool = INCLUDE_NORMALS,
    center_bottom: bool = CENTER_BOTTOM,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Sample surface points from a model and save them as a PLY file."""
    model, _ = load_geometric_model(model_path, center_bottom=center_bottom)
    points, normals = sample_surface_from_model(
        model,
        n_samples=n_samples,
        even_radius=even_radius,
        include_normals=include_normals,
    )
    write_points_to_ply(points, output_path, normals=normals)
    return points, normals


def compute_model_bounds(model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    bounds = np.asarray(model.trm_mesh.bounds, dtype=float)
    min_corner = bounds[0]
    max_corner = bounds[1]
    center = (min_corner + max_corner) / 2.0
    extents = max_corner - min_corner
    max_extent = max(float(np.max(extents)), 1e-6)
    return min_corner, max_corner, center, extents, max_extent


def compute_camera_from_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    extents = max_corner - min_corner
    max_extent = max(float(np.max(extents)), 1e-6)
    cam_dist = max(0.35, max_extent * 2.8)
    cam_pos = center + np.array([cam_dist, -cam_dist, max(cam_dist * 0.65, max_extent * 0.9)])
    return cam_pos, center, max_extent


def build_view_camera_specs(model) -> tuple[Dict[str, ViewCameraSpec], float]:
    _min_corner, _max_corner, center, _extents, max_extent = compute_model_bounds(model)
    cam_dist = max(DEPTH_MIN_CAMERA_DISTANCE, max_extent * DEPTH_CAMERA_DISTANCE_FACTOR)
    depth_far = cam_dist + max_extent * DEPTH_FAR_MARGIN_FACTOR
    specs = {
        "top": ViewCameraSpec("top", center + np.array([0.0, 0.0, cam_dist]), center, np.array([0.0, 1.0, 0.0])),
        "front": ViewCameraSpec("front", center + np.array([0.0, -cam_dist, 0.0]), center, np.array([0.0, 0.0, 1.0])),
        "left": ViewCameraSpec("left", center + np.array([-cam_dist, 0.0, 0.0]), center, np.array([0.0, 0.0, 1.0])),
        "right": ViewCameraSpec("right", center + np.array([cam_dist, 0.0, 0.0]), center, np.array([0.0, 0.0, 1.0])),
    }
    return specs, depth_far


def filter_points_to_model_bounds(points: np.ndarray, model, margin: float = VIEW_TEMPLATE_BBOX_MARGIN) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points.reshape(0, 3)
    min_corner, max_corner, _center, _extents, _max_extent = compute_model_bounds(model)
    min_corner = min_corner - margin
    max_corner = max_corner + margin
    mask = np.all((points >= min_corner) & (points <= max_corner), axis=1)
    return points[mask]


def voxel_downsample_points(points: np.ndarray, voxel_size: Optional[float] = VIEW_TEMPLATE_VOXEL_SIZE) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if voxel_size is None or voxel_size <= 0 or len(points) == 0:
        return points
    try:
        import open3d as o3d
    except ImportError:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(float(voxel_size))
    return np.asarray(pcd.points, dtype=float)




def ensure_optional_cv2_for_panda_utils() -> None:
    try:
        import cv2  # noqa: F401
    except ImportError:
        # VirtualDepthCamera does not use cv2 directly, but panda3d_utils imports it at module load time.
        sys.modules.setdefault("cv2", types.SimpleNamespace())
def generate_view_templates(model, view_names: Iterable[str] = VIEW_TEMPLATE_NAMES) -> Dict[str, np.ndarray]:
    import wrs.visualization.panda.world as wd
    ensure_optional_cv2_for_panda_utils()
    from wrs.visualization.panda.panda3d_utils import VirtualDepthCamera

    specs, depth_far = build_view_camera_specs(model)
    _min_corner, _max_corner, center, _extents, max_extent = compute_model_bounds(model)
    cam_pos = center + np.array([max_extent * 2.0, -max_extent * 2.0, max_extent * 1.5])
    base = wd.World(
        cam_pos=cam_pos,
        lookat_pos=center,
        w=WINDOW_WIDTH,
        h=WINDOW_HEIGHT,
        auto_rotate=AUTO_ROTATE,
    )
    model.attach_to(base)
    base.graphicsEngine.renderFrame()

    view_points: Dict[str, np.ndarray] = {}
    for view_name in view_names:
        if view_name not in specs:
            raise ValueError(f"Unknown view template name: {view_name}")
        spec = specs[view_name]
        camera = VirtualDepthCamera(
            cam_pos=spec.cam_pos,
            lookat_pos=spec.lookat_pos,
            resolution=DEPTH_RESOLUTION,
            fov=DEPTH_FOV,
            w_base=base,
            depth_far=depth_far,
        )
        camera.look_at(spec.lookat_pos, up=spec.up)
        points = camera.get_point_cloud(filter_zeros=True)
        camera.remove()
        points = filter_points_to_model_bounds(points, model, margin=VIEW_TEMPLATE_BBOX_MARGIN)
        points = voxel_downsample_points(points, voxel_size=VIEW_TEMPLATE_VOXEL_SIZE)
        if len(points) == 0:
            raise RuntimeError(f"VirtualDepthCamera generated no points for {view_name} view.")
        output_path = VIEW_OUTPUTS[view_name]
        write_points_to_ply(points, output_path, normals=None)
        view_points[view_name] = points
        print(f"Wrote {view_name} view template: {resolve_path(output_path)} ({len(points)} points)")

    if SHOW_VIEWER:
        draw_model_and_points_on_base(base, model, view_points)
    return view_points


def draw_model_and_points_on_base(base, model, view_points: Dict[str, np.ndarray]) -> None:
    import wrs.modeling.geometric_model as mgm

    model.alpha = MODEL_ALPHA
    if SHOW_MODEL_WIREFRAME:
        model.pdndp_core.setRenderModeWireframe()
        model.pdndp_core.setLightOff()

    if SHOW_WORLD_FRAME:
        _min_corner, _max_corner, _center, _extents, max_extent = compute_model_bounds(model)
        frame_length = max(max_extent * 0.35, 0.03)
        frame_radius = max(frame_length * 0.015, 0.0005)
        mgm.gen_frame(ax_length=frame_length, ax_radius=frame_radius).attach_to(base)

    for view_name, points in view_points.items():
        rgba = VIEW_POINT_RGBA.get(view_name, POINT_RGBA)
        mgm.gen_pointcloud(points=points, rgba=rgba, point_size=POINT_SIZE).attach_to(base)

    print("Viewer opened: translucent model + VirtualDepthCamera view templates.")
    base.run()


def draw_model_and_points(model, points: np.ndarray) -> None:
    import wrs.modeling.geometric_model as mgm
    import wrs.visualization.panda.world as wd

    cam_pos, lookat_pos, max_extent = compute_camera_from_points(points)
    base = wd.World(
        cam_pos=cam_pos,
        lookat_pos=lookat_pos,
        w=WINDOW_WIDTH,
        h=WINDOW_HEIGHT,
        auto_rotate=AUTO_ROTATE,
    )
    model.attach_to(base)
    model.alpha = MODEL_ALPHA

    if SHOW_WORLD_FRAME:
        frame_length = max(max_extent * 0.35, 0.03)
        frame_radius = max(frame_length * 0.015, 0.0005)
        mgm.gen_frame(ax_length=frame_length, ax_radius=frame_radius).attach_to(base)

    if SHOW_MODEL_WIREFRAME:
        model.pdndp_core.setRenderModeWireframe()
        model.pdndp_core.setLightOff()

    pointcloud = mgm.gen_pointcloud(points=points, rgba=POINT_RGBA, point_size=POINT_SIZE)
    pointcloud.attach_to(base)

    print("Viewer opened: translucent model + sampled surface points.")
    base.run()


def main() -> None:
    model, model_path = load_geometric_model(
        MODEL_PATH,
        rgb=MODEL_RGB,
        alpha=1.0,
        center_bottom=CENTER_BOTTOM,
    )

    surface_points = None
    surface_normals = None
    if GENERATE_SURFACE_TEMPLATE:
        surface_points, surface_normals = sample_surface_from_model(
            model,
            n_samples=N_SAMPLES,
            even_radius=EVEN_RADIUS,
            include_normals=INCLUDE_NORMALS,
        )
        write_points_to_ply(surface_points, OUTPUT_PLY, normals=surface_normals)
        print(f"Loaded model: {model_path}")
        print(f"Sampled {len(surface_points)} surface points.")
        print(f"Normals saved: {surface_normals is not None}")
        print(f"Wrote surface PLY: {resolve_path(OUTPUT_PLY)}")

    if GENERATE_VIEW_TEMPLATES:
        generate_view_templates(model, VIEW_TEMPLATE_NAMES)
    elif SHOW_VIEWER and surface_points is not None:
        draw_model_and_points(model, surface_points)


if __name__ == "__main__":
    main()
