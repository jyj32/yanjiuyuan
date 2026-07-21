"""
Sample surface points from box.STL, keeping only high faces whose normals are
parallel to the Z axis, save them as a PLY point cloud, and optionally draw the
model and sampled points in a WRS/Panda3D viewer.

Default usage:
    python yanjiuyuan/sample_box_surface.py

Adjust the constants below to change the model path, sample count, tolerance,
point size, colors, and other display settings.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yanjiuyuan.constants import (  # noqa: E402
    BOX_MODEL_PATH,
    BOX_TEMPLATE_CENTER_BOTTOM,
    BOX_TEMPLATE_INCLUDE_NORMALS,
    BOX_TEMPLATE_N_SAMPLES,
    BOX_TEMPLATE_NORMAL_Z_TOLERANCE,
    BOX_TEMPLATE_PLY,
    BOX_TEMPLATE_RANDOM_SEED,
    BOX_TEMPLATE_SURFACE_Z_MIN,
)
from yanjiuyuan.sample_bottle_surface import (  # noqa: E402
    center_model_xy_and_bottom,
    compute_camera_from_points,
    resolve_path,
    write_points_to_ply,
)


DEFAULT_MODEL = BOX_MODEL_PATH
DEFAULT_OUTPUT = BOX_TEMPLATE_PLY

MODEL_PATH = DEFAULT_MODEL
OUTPUT_PLY = DEFAULT_OUTPUT

# Sampling settings. NORMAL_Z_TOLERANCE is applied to abs(unit_normal dot Z).
N_SAMPLES = BOX_TEMPLATE_N_SAMPLES
NORMAL_Z_TOLERANCE = BOX_TEMPLATE_NORMAL_Z_TOLERANCE
SURFACE_Z_MIN = BOX_TEMPLATE_SURFACE_Z_MIN
INCLUDE_NORMALS = BOX_TEMPLATE_INCLUDE_NORMALS
CENTER_BOTTOM = BOX_TEMPLATE_CENTER_BOTTOM
RANDOM_SEED: Optional[int] = BOX_TEMPLATE_RANDOM_SEED

# Display settings.
SHOW_VIEWER = True
MODEL_RGB = np.array([0.42, 0.62, 0.95])
MODEL_ALPHA = 0.25
POINT_RGBA = np.array([1.0, 0.08, 0.02, 1.0])
POINT_SIZE = 0.002
SHOW_MODEL_WIREFRAME = False
SHOW_WORLD_FRAME = True
AUTO_ROTATE = False
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


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


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    normalized = np.zeros_like(vectors, dtype=float)
    valid = lengths > 0.0
    normalized[valid] = vectors[valid] / lengths[valid, None]
    return normalized


def transformed_vertices_and_normals(model) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(model.trm_mesh.vertices, dtype=float)
    face_normals = normalize_vectors(np.asarray(model.trm_mesh.face_normals, dtype=float))
    rotmat = np.asarray(model.rotmat, dtype=float)
    pos = np.asarray(model.pos, dtype=float)

    transformed_vertices = vertices @ rotmat.T + pos
    transformed_normals = normalize_vectors(face_normals @ rotmat.T)
    return transformed_vertices, transformed_normals


def find_z_parallel_face_ids(face_normals: np.ndarray, tolerance: float = NORMAL_Z_TOLERANCE) -> np.ndarray:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    unit_normals = normalize_vectors(face_normals)
    z_alignment = np.abs(unit_normals[:, 2])
    return np.flatnonzero(z_alignment >= 1.0 - tolerance)


def filter_face_ids_by_center_z(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    z_min: Optional[float] = SURFACE_Z_MIN,
) -> np.ndarray:
    if z_min is None:
        return face_ids

    triangles = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int)[face_ids]]
    centers = triangles.mean(axis=1)
    filtered_ids = face_ids[centers[:, 2] > z_min]
    if len(filtered_ids) == 0:
        raise RuntimeError(f"No selected faces have centers with z > {z_min}.")
    return filtered_ids


def sample_points_on_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_ids: np.ndarray,
    n_samples: int = N_SAMPLES,
    include_normals: bool = INCLUDE_NORMALS,
    random_seed: Optional[int] = RANDOM_SEED,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    faces = np.asarray(faces, dtype=int)
    face_ids = np.asarray(face_ids, dtype=int)
    triangles = np.asarray(vertices, dtype=float)[faces[face_ids]]
    normals = normalize_vectors(np.asarray(face_normals, dtype=float)[face_ids])

    edge01 = triangles[:, 1] - triangles[:, 0]
    edge02 = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge01, edge02), axis=1)
    valid = areas > 0.0
    if not np.any(valid):
        raise RuntimeError("All selected faces have zero area.")

    triangles = triangles[valid]
    normals = normals[valid]
    areas = areas[valid]
    probabilities = areas / areas.sum()

    rng = np.random.default_rng(random_seed)
    chosen = rng.choice(len(triangles), size=n_samples, p=probabilities)
    chosen_triangles = triangles[chosen]

    r0 = rng.random(n_samples)
    r1 = rng.random(n_samples)
    sqrt_r0 = np.sqrt(r0)
    bary0 = 1.0 - sqrt_r0
    bary1 = sqrt_r0 * (1.0 - r1)
    bary2 = sqrt_r0 * r1

    points = (
        chosen_triangles[:, 0] * bary0[:, None]
        + chosen_triangles[:, 1] * bary1[:, None]
        + chosen_triangles[:, 2] * bary2[:, None]
    )
    point_normals = normals[chosen] if include_normals else None
    return points, point_normals


def sample_z_parallel_surface_from_model(
    model,
    n_samples: int = N_SAMPLES,
    tolerance: float = NORMAL_Z_TOLERANCE,
    z_min: Optional[float] = SURFACE_Z_MIN,
    include_normals: bool = INCLUDE_NORMALS,
    random_seed: Optional[int] = RANDOM_SEED,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Sample high mesh faces whose normals are parallel to +/-Z."""
    vertices, face_normals = transformed_vertices_and_normals(model)
    z_parallel_face_ids = find_z_parallel_face_ids(face_normals, tolerance=tolerance)
    if len(z_parallel_face_ids) == 0:
        raise RuntimeError(
            "No faces have normals parallel to the Z axis. "
            "Increase NORMAL_Z_TOLERANCE if the mesh is slightly tilted."
        )
    selected_face_ids = filter_face_ids_by_center_z(
        vertices=vertices,
        faces=model.trm_mesh.faces,
        face_ids=z_parallel_face_ids,
        z_min=z_min,
    )

    points, normals = sample_points_on_faces(
        vertices=vertices,
        faces=model.trm_mesh.faces,
        face_normals=face_normals,
        face_ids=selected_face_ids,
        n_samples=n_samples,
        include_normals=include_normals,
        random_seed=random_seed,
    )
    return points, normals, selected_face_ids


def sample_surface_points(
    model_path: Path,
    output_path: Path,
    n_samples: int = N_SAMPLES,
    tolerance: float = NORMAL_Z_TOLERANCE,
    z_min: Optional[float] = SURFACE_Z_MIN,
    include_normals: bool = INCLUDE_NORMALS,
    center_bottom: bool = CENTER_BOTTOM,
    random_seed: Optional[int] = RANDOM_SEED,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Sample high Z-parallel box faces and save them as a PLY file."""
    model, _ = load_geometric_model(model_path, center_bottom=center_bottom)
    points, normals, face_ids = sample_z_parallel_surface_from_model(
        model,
        n_samples=n_samples,
        tolerance=tolerance,
        z_min=z_min,
        include_normals=include_normals,
        random_seed=random_seed,
    )
    write_points_to_ply(points, output_path, normals=normals)
    return points, normals, face_ids


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

    if SHOW_WORLD_FRAME:
        frame_length = max(max_extent * 0.35, 0.03)
        frame_radius = max(frame_length * 0.015, 0.0005)
        mgm.gen_frame(ax_length=frame_length, ax_radius=frame_radius).attach_to(base)

    if SHOW_MODEL_WIREFRAME:
        model.pdndp_core.setRenderModeWireframe()
        model.pdndp_core.setLightOff()
    model.attach_to(base)

    pointcloud = mgm.gen_pointcloud(points=points, rgba=POINT_RGBA, point_size=POINT_SIZE)
    pointcloud.attach_to(base)

    print("Viewer opened: translucent model + sampled Z-parallel surface points.")
    base.run()


def main() -> None:
    model, model_path = load_geometric_model(
        MODEL_PATH,
        rgb=MODEL_RGB,
        alpha=MODEL_ALPHA,
        center_bottom=CENTER_BOTTOM,
    )
    points, normals, face_ids = sample_z_parallel_surface_from_model(
        model,
        n_samples=N_SAMPLES,
        tolerance=NORMAL_Z_TOLERANCE,
        z_min=SURFACE_Z_MIN,
        include_normals=INCLUDE_NORMALS,
        random_seed=RANDOM_SEED,
    )
    write_points_to_ply(points, OUTPUT_PLY, normals=normals)

    print(f"Loaded model: {model_path}")
    print(f"Selected {len(face_ids)} faces whose normals are parallel to +/-Z and center z > {SURFACE_Z_MIN}.")
    print(f"Sampled {len(points)} surface points.")
    print(f"Normals saved: {normals is not None}")
    print(f"Wrote PLY: {resolve_path(OUTPUT_PLY)}")

    if SHOW_VIEWER:
        draw_model_and_points(model, points)


if __name__ == "__main__":
    main()
