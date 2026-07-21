"""Visualize the concave collision model generated for the detected box."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from yanjiuyuan.box_collision import (  # noqa: E402
    describe_panel_specs,
    load_homomat,
    local_probe_points,
    make_box_panel_specs,
    make_concave_box_collision_obstacles,
    make_detected_box_visual_model,
    transform_local_pose,
)
from yanjiuyuan.constants import BOX_CAPTURE_ROOT  # noqa: E402


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def find_latest_object_summary() -> Path:
    candidates = list(BOX_CAPTURE_ROOT.glob("*/box_object_extraction/box_object_extraction_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No box_object_extraction_summary.json found under {BOX_CAPTURE_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def box_transform_from_summary(summary_path: Path) -> Path:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    path = Path(summary["box_transform_used"])
    if not path.exists():
        raise FileNotFoundError(f"box_transform_used does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize concave box collision panels for WRS planning.")
    parser.add_argument("--box-transform", type=Path, default=None, help="Path to detected box 4x4 transform txt.")
    parser.add_argument("--object-summary", type=Path, default=None, help="box_object_extraction_summary.json.")
    parser.add_argument("--show-cdprim", action="store_true", help="Show Panda collision primitives for every panel.")
    parser.add_argument("--probe-radius", type=float, default=0.012)
    parser.add_argument("--no-probes", action="store_true")
    parser.add_argument("--no-run", action="store_true", help="Build and print diagnostics without opening the WRS window.")
    return parser.parse_args()


def resolve_box_transform_path(args: argparse.Namespace) -> Path:
    box_transform = resolve_path(args.box_transform)
    if box_transform is not None:
        if not box_transform.exists():
            raise FileNotFoundError(f"Box transform not found: {box_transform}")
        return box_transform
    object_summary = resolve_path(args.object_summary)
    if object_summary is None:
        object_summary = find_latest_object_summary()
    if not object_summary.exists():
        raise FileNotFoundError(f"Object summary not found: {object_summary}")
    return box_transform_from_summary(object_summary)


def _probe_cdprim_fn(radius: float):
    def _make_cdprim(name: str, ex_radius: float):
        from panda3d.core import CollisionBox, CollisionNode, LPoint3, NodePath

        half = float(radius) + (0.0 if ex_radius is None else float(ex_radius))
        cnode = CollisionNode(f"{name}_cnode")
        cnode.addSolid(CollisionBox(center=LPoint3(0.0, 0.0, 0.0), x=half, y=half, z=half))
        return NodePath(cnode)

    return _make_cdprim


def make_probe_sphere(name: str, pos: np.ndarray, collided: bool, radius: float):
    from wrs import mcm, mgm

    rgb = np.array([1.0, 0.12, 0.08]) if collided else np.array([0.0, 0.78, 0.26])
    sphere_sgm = mgm.gen_sphere(pos=np.zeros(3), radius=radius, rgb=rgb, alpha=0.9)
    sphere = mcm.CollisionModel(
        sphere_sgm,
        name=name,
        cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
        ex_radius=0.0,
        userdef_cdprim_fn=_probe_cdprim_fn(radius),
        rgb=rgb,
        alpha=0.9,
    )
    sphere.pos = pos
    return sphere


def local_probe_collides(local_pos: np.ndarray, radius: float) -> bool:
    local_pos = np.asarray(local_pos, dtype=float)
    for spec in make_box_panel_specs():
        half_lengths = spec.local_lengths * 0.5 + float(radius)
        delta = np.abs(local_pos - spec.local_center)
        if bool(np.all(delta <= half_lengths)):
            return True
    return False


def probe_collision(box_homomat: np.ndarray, panels: list[object], radius: float) -> list[object]:
    probes = []
    for name, local_pos in local_probe_points().items():
        world_pos, _rotmat = transform_local_pose(box_homomat, local_pos)
        collided = local_probe_collides(local_pos, radius)
        probe = make_probe_sphere(name, world_pos, collided=collided, radius=radius)
        probes.append(probe)
        print(f"[box_collision] probe {name}: local={np.round(local_pos, 5).tolist()} collided={collided}")
    return probes


def main() -> None:
    args = parse_args()
    box_transform_path = resolve_box_transform_path(args)
    box_homomat = load_homomat(box_transform_path)

    print(f"[box_collision] box transform: {box_transform_path}")
    print("[box_collision] generated concave collision panels:")
    for line in describe_panel_specs():
        print(f"  {line}")

    panels = make_concave_box_collision_obstacles(box_homomat, show_cdprim=args.show_cdprim)
    print(f"[box_collision] panel count: {len(panels)}")

    probes = [] if args.no_probes else probe_collision(box_homomat, panels, radius=args.probe_radius)
    if args.no_run:
        return

    from wrs import mgm, wd

    base = wd.World(cam_pos=[1.8, -1.45, 1.0], lookat_pos=box_homomat[:3, 3] + np.array([0.0, 0.0, 0.12]), w=1280, h=720)
    mgm.gen_frame(ax_length=0.25, ax_radius=0.004).attach_to(base)
    mgm.gen_frame(pos=box_homomat[:3, 3], rotmat=box_homomat[:3, :3], ax_length=0.16, ax_radius=0.003).attach_to(base)

    make_detected_box_visual_model(box_homomat).attach_to(base)
    for panel in panels:
        panel.attach_to(base)
    for probe in probes:
        probe.attach_to(base)

    import wrs.modeling.collision_model as mcm
    import os

    from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
    def get_bottle_cdprim(name="bottle_cdprim", ex_radius=0):
        pdcnd = CollisionNode(name+ "_cnode")
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.01), x=.065 + ex_radius, y=.044 + ex_radius, z=.01 + ex_radius))
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.03), x=.072 + ex_radius, y=.045 + ex_radius, z=.01 + ex_radius))
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.05), x=.075 + ex_radius, y=.045 + ex_radius, z=.01 + ex_radius))
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.07), x=.0755 + ex_radius, y=.045 + ex_radius, z=.01 + ex_radius))
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.12), x=.077 + ex_radius, y=.045 + ex_radius, z=.04 + ex_radius))
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.17), x=.0755 + ex_radius, y=.045 + ex_radius, z=.01 + ex_radius))
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.19), x=.074 + ex_radius, y=.045 + ex_radius, z=.01 + ex_radius))
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.21), x=.071 + ex_radius, y=.044 + ex_radius, z=.01 + ex_radius))
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.23), x=.062 + ex_radius, y=.042 + ex_radius, z=.01 + ex_radius))
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.25), x=.044 + ex_radius, y=.033 + ex_radius, z=.01 + ex_radius))
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.275), x=.029 + ex_radius, y=.029 + ex_radius, z=.015 + ex_radius))
        pdcnd.addSolid(CollisionBox(Point3(0, 0, 0.305), x=.032 + ex_radius, y=.032 + ex_radius, z=.015 + ex_radius))
        cdprim = NodePath(name+"_cdprim")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    bottle = mcm.CollisionModel(
        initor=os.path.join(os.path.dirname(__file__), "models", "bottle.STL"),
        cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
        userdef_cdprim_fn=get_bottle_cdprim
    )
    # 瓶子碰撞体
    bottle.show_cdprim()
    bottle.alpha = .1
    bottle.attach_to(base)
    base.run()

    print("[box_collision] opening WRS visualization.")
    print("[box_collision] blue panels are collision walls/bottom; green probes are free; red probes collide.")
    base.run()


if __name__ == "__main__":
    main()