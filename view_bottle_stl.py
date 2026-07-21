"""
View the bottle STL model in a WRS/Panda3D scene.

Usage:
    python yanjiuyuan/view_bottle_stl.py

Optional examples:
    python yanjiuyuan/view_bottle_stl.py --wireframe
    python yanjiuyuan/view_bottle_stl.py --keep-pose --auto-rotate
    python yanjiuyuan/view_bottle_stl.py --model path/to/other_model.stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "bottle.stl"
MODEL_DIR = Path(__file__).resolve().parent / "models"
TEMPLATE_PATHS = {
    "surface": MODEL_DIR / "bottle_surface_points.ply",
    "top": MODEL_DIR / "bottle_top_view_points.ply",
    "front": MODEL_DIR / "bottle_front_view_points.ply",
    "left": MODEL_DIR / "bottle_left_view_points.ply",
    "right": MODEL_DIR / "bottle_right_view_points.ply",
}
TEMPLATE_ORDER = ("surface", "top", "front", "left", "right")
TEMPLATE_RGBA = {
    "surface": np.array([1.0, 0.08, 0.02, 1.0]),
    "top": np.array([0.0, 0.75, 1.0, 1.0]),
    "front": np.array([0.0, 0.85, 0.2, 1.0]),
    "left": np.array([1.0, 0.70, 0.0, 1.0]),
    "right": np.array([0.85, 0.25, 1.0, 1.0]),
}


def parse_vec3(value: str) -> np.ndarray:
    values = [item.strip() for item in value.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values, e.g. 1,0,0.5")
    try:
        return np.array([float(item) for item in values], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("all vector values must be numbers") from exc


def parse_rgb(value: str) -> np.ndarray:
    rgb = parse_vec3(value)
    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise argparse.ArgumentTypeError("RGB values must be in the range [0, 1]")
    return rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View yanjiuyuan/models/bottle.stl.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="STL model path.")
    parser.add_argument(
        "--keep-pose",
        action="store_true",
        help="Keep the STL coordinates instead of centering XY and placing the bottom at Z=0.",
    )
    parser.add_argument("--rgb", type=parse_rgb, default=np.array([0.42, 0.62, 0.95]), help="Model color as r,g,b.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Model opacity in the range [0, 1].")
    parser.add_argument("--wireframe", action="store_true", help="Display the model as wireframe.")
    parser.add_argument("--show-local-frame", action="store_true", help="Show the model local coordinate frame.")
    parser.add_argument("--templates", action=argparse.BooleanOptionalAction, default=True, help="Show generated bottle template point clouds and switch them with Space.")
    parser.add_argument("--template", choices=TEMPLATE_ORDER, default="surface", help="Initial template point cloud to show.")
    parser.add_argument("--template-point-size", type=float, default=0.002, help="Template point cloud point size.")
    parser.add_argument("--hide-world-frame", action="store_true", help="Hide the world coordinate frame.")
    parser.add_argument("--auto-rotate", action="store_true", help="Slowly rotate the camera around the model.")
    parser.add_argument("--cam-pos", type=parse_vec3, default=None, help="Camera position as x,y,z.")
    parser.add_argument("--lookat-pos", type=parse_vec3, default=None, help="Camera look-at position as x,y,z.")
    parser.add_argument("--cam-dist", type=float, default=None, help="Auto-camera distance override.")
    parser.add_argument("--width", type=int, default=1280, help="Viewer window width.")
    parser.add_argument("--height", type=int, default=720, help="Viewer window height.")
    return parser.parse_args()


def resolve_model_path(model_path: Path) -> Path:
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    return model_path


def compute_view_pose(bounds: np.ndarray, keep_pose: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    min_corner = bounds[0]
    max_corner = bounds[1]
    center = (min_corner + max_corner) / 2.0
    extents = max_corner - min_corner
    max_extent = max(float(np.max(extents)), 1e-6)

    if keep_pose:
        model_pos = np.zeros(3)
        lookat_pos = center
    else:
        model_pos = np.array([-center[0], -center[1], -min_corner[2]], dtype=float)
        lookat_pos = np.array([0.0, 0.0, extents[2] / 2.0], dtype=float)

    return model_pos, lookat_pos, extents, max_extent




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
    return np.asarray(points, dtype=float)


def load_template_points(ply_path: Path) -> np.ndarray:
    ply_path = resolve_model_path(ply_path)
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points, dtype=float)
        if len(points) > 0:
            return points
    except ImportError:
        pass
    return read_ascii_ply_points(ply_path)


class TemplateSwitcher:
    def __init__(self, base, template_models: list[tuple[str, object]], initial_index: int = 0):
        self.base = base
        self.template_models = template_models
        self.index = int(initial_index) % max(1, len(template_models))
        self.current = None
        if self.template_models:
            self.show(self.index)
            base.accept("space", self.next)

    def show(self, index: int) -> None:
        if self.current is not None:
            self.current.detach()
        self.index = index % len(self.template_models)
        name, model = self.template_models[self.index]
        self.current = model
        model.attach_to(self.base)
        print(f"Template [{self.index + 1}/{len(self.template_models)}]: {name}. Press Space to switch.")

    def next(self) -> None:
        if self.template_models:
            self.show(self.index + 1)


def main() -> None:
    args = parse_args()
    if args.alpha < 0.0 or args.alpha > 1.0:
        raise ValueError("--alpha must be in the range [0, 1]")

    import wrs.basis.robot_math as rm
    import wrs.modeling.geometric_model as mgm
    import wrs.visualization.panda.world as wd

    model_path = resolve_model_path(args.model)
    bottle = mgm.GeometricModel(
        initor=str(model_path),
        name=model_path.stem,
        toggle_twosided=True,
        rgb=args.rgb,
        alpha=args.alpha,
    )
    if bottle.trm_mesh is None:
        raise RuntimeError(f"Unable to load a triangle mesh from: {model_path}")

    bounds = np.asarray(bottle.trm_mesh.bounds, dtype=float)
    model_pos, auto_lookat, extents, max_extent = compute_view_pose(bounds, args.keep_pose)
    bottle.pos = model_pos

    lookat_pos = args.lookat_pos if args.lookat_pos is not None else auto_lookat
    cam_dist = args.cam_dist if args.cam_dist is not None else max(0.35, max_extent * 2.8)
    cam_pos = args.cam_pos
    if cam_pos is None:
        cam_pos = lookat_pos + np.array([cam_dist, -cam_dist, max(cam_dist * 0.65, max_extent * 0.9)])

    base = wd.World(
        cam_pos=cam_pos,
        lookat_pos=lookat_pos,
        w=args.width,
        h=args.height,
        auto_rotate=args.auto_rotate,
    )

    if not args.hide_world_frame:
        frame_length = max(max_extent * 0.35, 0.03)
        frame_radius = max(frame_length * 0.015, 0.0005)
        mgm.gen_frame(ax_length=frame_length, ax_radius=frame_radius).attach_to(base)

    if args.wireframe:
        bottle.pdndp_core.setRenderModeWireframe()
        bottle.pdndp_core.setLightOff()
    if args.show_local_frame:
        bottle.show_local_frame()

    bottle.attach_to(base)

    template_models = []
    if args.templates:
        for template_name in TEMPLATE_ORDER:
            template_path = TEMPLATE_PATHS[template_name]
            if not template_path.exists():
                print(f"Template missing: {template_name} ({template_path})")
                continue
            template_points = load_template_points(template_path) + model_pos
            if len(template_points) == 0:
                print(f"Template empty: {template_name} ({template_path})")
                continue
            pointcloud = mgm.gen_pointcloud(
                points=template_points,
                rgba=TEMPLATE_RGBA.get(template_name, np.array([1.0, 0.08, 0.02, 1.0])),
                point_size=args.template_point_size,
            )
            template_models.append((template_name, pointcloud))
        initial_template_index = next(
            (idx for idx, (name, _model) in enumerate(template_models) if name == args.template),
            0,
        )
        template_switcher = TemplateSwitcher(base, template_models, initial_template_index)
    else:
        template_switcher = None

    print(f"Loaded model: {model_path}")
    print(f"Bounds min: {rm.np.array2string(bounds[0], precision=5)}")
    print(f"Bounds max: {rm.np.array2string(bounds[1], precision=5)}")
    print(f"Extents: {rm.np.array2string(extents, precision=5)}")
    if args.templates:
        print(f"Templates loaded: {len(template_models)}. Press Space to switch template point clouds.")
    print("Use the mouse to rotate, pan, and zoom the WRS viewer.")
    base.run()


if __name__ == "__main__":
    main()
