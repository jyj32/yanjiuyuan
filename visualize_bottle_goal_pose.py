"""Lightweight visualization of pick/place goal poses for the bottle ICP pipeline."""

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

from yanjiuyuan import sim_pick_and_place as sim_pick  # noqa: E402
from yanjiuyuan.box_collision import (  # noqa: E402
    load_homomat,
    make_concave_box_collision_obstacles,
    make_detected_box_visual_model,
)
from yanjiuyuan.constants import BOX_CAPTURE_ROOT, BOTTLE_ROBOT_SIDE_PLACE_POS, MODEL_DIR  # noqa: E402
from yanjiuyuan.sim_bottle_pick_place_from_box_object_icp import homomat_to_pose  # noqa: E402

DEFAULT_OBJECT_MODEL_PATH = MODEL_DIR / "bottle.stl"


def parse_vec3(value: str) -> np.ndarray:
    items = [item.strip() for item in value.split(",")]
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    try:
        result = np.array([float(item) for item in items], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not np.all(np.isfinite(result)):
        raise argparse.ArgumentTypeError("vector contains NaN or inf")
    return result


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def find_latest_object_summary() -> Path:
    candidates = list(BOX_CAPTURE_ROOT.glob("*/box_object_extraction/box_object_extraction_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No box_object_extraction_summary.json found under {BOX_CAPTURE_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_goal_inputs(summary_path: Path) -> tuple[Path, Path, Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bottle_summary = summary.get("bottle_icp")
    if bottle_summary is None:
        raise RuntimeError(f"Summary has no bottle_icp result: {summary_path}")
    bottle_transform_path = Path(bottle_summary["icp_transform_path"])
    box_transform_path = Path(summary["box_transform_used"])
    bottle_model_path = Path(bottle_summary.get("bottle_stl") or DEFAULT_OBJECT_MODEL_PATH)
    for label, path in (
        ("bottle transform", bottle_transform_path),
        ("box transform", box_transform_path),
        ("bottle model", bottle_model_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    return bottle_transform_path, box_transform_path, bottle_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw bottle pick pose and place goal pose without planning.")
    parser.add_argument("--object-summary", type=Path, default=None, help="box_object_extraction_summary.json.")
    parser.add_argument("--place-pos", type=parse_vec3, default=None, help="Override place world XYZ.")
    parser.add_argument("--place-rpy-deg", type=parse_vec3, default=None, help="Override place world RPY in degrees.")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--no-box-panels", action="store_true")
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--no-run", action="store_true", help="Print pose diagnostics without opening WRS.")
    return parser.parse_args()


def make_place_pose(args: argparse.Namespace, pick_pose: tuple[np.ndarray, np.ndarray]) -> tuple[tuple[np.ndarray, np.ndarray], str, str]:
    if args.place_pos is None:
        place_pos = np.asarray(BOTTLE_ROBOT_SIDE_PLACE_POS, dtype=float).copy()
        place_pos_source = f"constants.BOTTLE_ROBOT_SIDE_PLACE_POS {sim_pick.format_vec(place_pos, digits=3)}"
    else:
        place_pos = np.asarray(args.place_pos, dtype=float)
        place_pos_source = "--place-pos"
    if args.place_rpy_deg is None:
        place_rotmat = pick_pose[1].copy()
        place_rot_source = "keep ICP pick orientation"
    else:
        _pos, place_rotmat = sim_pick.pose_from_pos_rpy(place_pos, args.place_rpy_deg)
        place_rot_source = f"--place-rpy-deg {sim_pick.format_vec(args.place_rpy_deg, digits=3)}"
    return (place_pos, place_rotmat), place_pos_source, place_rot_source


def attach_model(model, base) -> None:
    try:
        model.copy().attach_to(base)
    except Exception:
        model.attach_to(base)


def main() -> None:
    args = parse_args()
    object_summary = resolve_path(args.object_summary) if args.object_summary is not None else find_latest_object_summary()
    bottle_transform_path, box_transform_path, bottle_model_path = read_goal_inputs(object_summary)

    bottle_homomat = load_homomat(bottle_transform_path)
    box_homomat = load_homomat(box_transform_path)
    pick_pose = homomat_to_pose(bottle_homomat)
    place_pose, place_pos_source, place_rot_source = make_place_pose(args, pick_pose)

    print(f"[goal_pose] object summary: {object_summary}")
    print(f"[goal_pose] bottle transform: {bottle_transform_path}")
    print(f"[goal_pose] box transform: {box_transform_path}")
    print(f"[goal_pose] bottle model: {bottle_model_path}")
    print(f"[goal_pose] pick pos: {sim_pick.format_vec(pick_pose[0], digits=6)}")
    print(
        f"[goal_pose] place pos: {sim_pick.format_vec(place_pose[0], digits=6)} "
        f"({place_pos_source}), rot={place_rot_source}"
    )
    if args.no_run:
        return

    from wrs import mgm, wd

    lookat = (pick_pose[0] + place_pose[0]) * 0.5 + np.array([0.0, 0.0, 0.12])
    base = wd.World(cam_pos=[1.8, -1.55, 1.05], lookat_pos=lookat, w=1280, h=720)
    mgm.gen_frame(ax_length=0.25, ax_radius=0.004).attach_to(base)

    if not args.no_robot:
        robot = sim_pick.make_robot()
        robot.gen_meshmodel(
            alpha=0.55,
            toggle_tcp_frame=True,
            toggle_flange_frame=True,
            toggle_jnt_frames=False,
        ).attach_to(base)

    if not args.no_table:
        attach_model(sim_pick.make_table_obstacle(), base)

    attach_model(make_detected_box_visual_model(box_homomat), base)
    if not args.no_box_panels:
        for panel in make_concave_box_collision_obstacles(box_homomat):
            attach_model(panel, base)

    sim_pick.make_object_model(
        bottle_model_path,
        pick_pose,
        name="goal_pick_bottle",
        alpha=0.55,
        rgb=np.array([0.25, 0.62, 1.0]),
    ).attach_to(base)
    sim_pick.make_object_model(
        bottle_model_path,
        place_pose,
        name="goal_place_bottle",
        alpha=0.42,
        rgb=np.array([0.1, 0.9, 0.38]),
    ).attach_to(base)

    mgm.gen_frame(pos=pick_pose[0], rotmat=pick_pose[1], ax_length=0.11, ax_radius=0.0025).attach_to(base)
    mgm.gen_frame(pos=place_pose[0], rotmat=place_pose[1], ax_length=0.11, ax_radius=0.0025).attach_to(base)
    mgm.gen_arrow(
        spos=pick_pose[0],
        epos=place_pose[0],
        rgb=np.array([1.0, 0.72, 0.08]),
        alpha=0.9,
        stick_radius=0.004,
    ).attach_to(base)

    print("[goal_pose] opening WRS goal-pose visualization.")
    print("[goal_pose] robot is shown at DEFAULT_HOME_CONF; blue bottle: pick pose; green bottle: place goal pose.")
    base.run()


if __name__ == "__main__":
    main()