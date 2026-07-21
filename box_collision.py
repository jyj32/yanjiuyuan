"""Concave collision helpers for the detected blue box.

The box is an open container. A single AABB makes the inner cavity solid, so
planning must use separate collision panels for the bottom and four side walls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import open3d as o3d
from panda3d.core import CollisionBox, CollisionNode, NodePath, Point3
from wrs import mcm, mgm
REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from yanjiuyuan.constants import (  # noqa: E402
    BOX_MODEL_PATH,
    BOX_OBJECT_CONCAVE_X_RANGE,
    BOX_OBJECT_CONCAVE_Y_RANGE,
    BOX_OBJECT_CONCAVE_Z_RANGE,
)

DEFAULT_BOX_OUTER_BOUNDS = np.array(
    [
        [-0.30000001, -0.2, 0.0],
        [0.30000001, 0.2, 0.23],
    ],
    dtype=float,
)
DEFAULT_PANEL_EX_RADIUS = 0.01
PANEL_ALPHA = 0.34
VISUAL_BOX_ALPHA = 1 # 箱子的不透明度


@dataclass(frozen=True)
class BoxPanelSpec:
    name: str
    local_center: np.ndarray
    local_lengths: np.ndarray
    rgb: np.ndarray


def load_homomat(path: Path) -> np.ndarray:
    homomat = np.asarray(np.loadtxt(path), dtype=float)
    if homomat.shape != (4, 4):
        raise ValueError(f"Box transform must be 4x4, got {homomat.shape}: {path}")
    if not np.all(np.isfinite(homomat)):
        raise ValueError(f"Box transform contains NaN or inf: {path}")
    return homomat


def load_box_outer_bounds(model_path: Path = BOX_MODEL_PATH) -> np.ndarray:
    try:

        mesh = o3d.io.read_triangle_mesh(str(model_path))
        if not mesh.is_empty():
            return np.vstack([mesh.get_min_bound(), mesh.get_max_bound()]).astype(float)
    except Exception:
        pass
    return DEFAULT_BOX_OUTER_BOUNDS.copy()


def _bounds_from_ranges(x_range: Iterable[float], y_range: Iterable[float], z_range: Iterable[float]) -> np.ndarray:
    return np.array(
        [
            [float(x_range[0]), float(y_range[0]), float(z_range[0])],
            [float(x_range[1]), float(y_range[1]), float(z_range[1])],
        ],
        dtype=float,
    )


def make_box_panel_specs(
    outer_bounds: np.ndarray | None = None,
    inner_bounds: np.ndarray | None = None,
) -> list[BoxPanelSpec]:
    outer_bounds = load_box_outer_bounds() if outer_bounds is None else np.asarray(outer_bounds, dtype=float)
    if outer_bounds.shape != (2, 3):
        raise ValueError(f"outer_bounds must be shaped (2, 3), got {outer_bounds.shape}")
    outer_min = outer_bounds[0]
    outer_max = outer_bounds[1]

    if inner_bounds is None:
        raw_inner = _bounds_from_ranges(
            BOX_OBJECT_CONCAVE_X_RANGE,
            BOX_OBJECT_CONCAVE_Y_RANGE,
            BOX_OBJECT_CONCAVE_Z_RANGE,
        )
    else:
        raw_inner = np.asarray(inner_bounds, dtype=float)
    if raw_inner.shape != (2, 3):
        raise ValueError(f"inner_bounds must be shaped (2, 3), got {raw_inner.shape}")

    inner_min = np.maximum(raw_inner[0], outer_min)
    inner_max = np.minimum(raw_inner[1], outer_max)
    # The concave z max used by point filtering may sit slightly above the STL.
    inner_max[2] = outer_max[2]

    if np.any(inner_min[:2] <= outer_min[:2]) or np.any(inner_max[:2] >= outer_max[:2]):
        raise ValueError(f"Inner XY bounds must lie inside outer bounds: outer={outer_bounds}, inner={raw_inner}")
    if not (outer_min[2] < inner_min[2] < outer_max[2]):
        raise ValueError(f"Inner z min must be between box bottom/top: outer={outer_bounds}, inner={raw_inner}")

    outer_size = outer_max - outer_min
    inner_size = inner_max - inner_min
    bottom_thickness = inner_min[2] - outer_min[2]
    wall_height = outer_max[2] - inner_min[2]
    wall_z = (inner_min[2] + outer_max[2]) * 0.5
    bottom_z = (outer_min[2] + inner_min[2]) * 0.5

    x_minus_t = inner_min[0] - outer_min[0]
    x_plus_t = outer_max[0] - inner_max[0]
    y_minus_t = inner_min[1] - outer_min[1]
    y_plus_t = outer_max[1] - inner_max[1]

    return [
        BoxPanelSpec(
            name="bottom",
            local_center=np.array([(outer_min[0] + outer_max[0]) * 0.5, (outer_min[1] + outer_max[1]) * 0.5, bottom_z]),
            local_lengths=np.array([outer_size[0], outer_size[1], bottom_thickness]),
            rgb=np.array([0.15, 0.42, 0.95]),
        ),
        BoxPanelSpec(
            name="x_minus_wall",
            local_center=np.array([(outer_min[0] + inner_min[0]) * 0.5, (outer_min[1] + outer_max[1]) * 0.5, wall_z]),
            local_lengths=np.array([x_minus_t, outer_size[1], wall_height]),
            rgb=np.array([0.1, 0.62, 0.95]),
        ),
        BoxPanelSpec(
            name="x_plus_wall",
            local_center=np.array([(inner_max[0] + outer_max[0]) * 0.5, (outer_min[1] + outer_max[1]) * 0.5, wall_z]),
            local_lengths=np.array([x_plus_t, outer_size[1], wall_height]),
            rgb=np.array([0.1, 0.62, 0.95]),
        ),
        BoxPanelSpec(
            name="y_minus_wall",
            local_center=np.array([(inner_min[0] + inner_max[0]) * 0.5, (outer_min[1] + inner_min[1]) * 0.5, wall_z]),
            local_lengths=np.array([inner_size[0], y_minus_t, wall_height]),
            rgb=np.array([0.0, 0.76, 0.78]),
        ),
        BoxPanelSpec(
            name="y_plus_wall",
            local_center=np.array([(inner_min[0] + inner_max[0]) * 0.5, (inner_max[1] + outer_max[1]) * 0.5, wall_z]),
            local_lengths=np.array([inner_size[0], y_plus_t, wall_height]),
            rgb=np.array([0.0, 0.76, 0.78]),
        ),
    ]

# 蓝色箱子碰撞体模型
def make_hollow_box_cdprim(name="hollow_box_cdprim", ex_radius=-0.005):

    pdcnd = CollisionNode(name + "_cnode")
    Lx, Ly, Lz = 0.6, 0.4, 0.26 # 箱子高度原长0.23m
    t = 0.01  # 壁厚
    # 底面
    pdcnd.addSolid(CollisionBox(Point3(0, 0, t / 2), x=Lx / 2 + ex_radius, y=Ly / 2 + ex_radius, z=t / 2 + ex_radius))
    # 前面
    pdcnd.addSolid(CollisionBox(Point3(0, Ly / 2 - t / 2, Lz / 2), x=Lx / 2 + ex_radius, y=t / 2 + ex_radius,
                                z=Lz / 2  + ex_radius))
    # # 后面
    pdcnd.addSolid(CollisionBox(Point3(0, -Ly / 2 + t / 2, Lz / 2), x=Lx / 2 + ex_radius, y=t / 2 + ex_radius,
                                z=Lz / 2  + ex_radius))
    # # 左面
    pdcnd.addSolid(CollisionBox(Point3(-Lx / 2 + t / 2, 0, Lz / 2), x=t / 2 + ex_radius, y=Ly / 2 + ex_radius,
                                z=Lz / 2  + ex_radius))
    # # 右面
    pdcnd.addSolid(CollisionBox(Point3(Lx / 2 - t / 2, 0, Lz / 2), x=t / 2 + ex_radius, y=Ly / 2 + ex_radius,
                                z=Lz / 2  + ex_radius))
    cdprim = NodePath(name + "_cdprim")
    cdprim.attachNewNode(pdcnd)
    return cdprim

def transform_local_pose(box_homomat: np.ndarray, local_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    box_homomat = np.asarray(box_homomat, dtype=float)
    if box_homomat.shape != (4, 4):
        raise ValueError(f"box_homomat must be 4x4, got {box_homomat.shape}")
    rotmat = box_homomat[:3, :3]
    pos = rotmat.dot(np.asarray(local_center, dtype=float)) + box_homomat[:3, 3]
    return pos, rotmat


def make_panel_collision_model(
    spec: BoxPanelSpec,
    box_homomat: np.ndarray,
    ex_radius: float = DEFAULT_PANEL_EX_RADIUS,
    alpha: float = PANEL_ALPHA,
):


    panel_sgm = mgm.gen_box(xyz_lengths=spec.local_lengths, rgb=spec.rgb, alpha=alpha)
    panel = mcm.CollisionModel(
        panel_sgm,
        name=f"detected_box_{spec.name}",
        cdprim_type=mcm.const.CDPrimType.AABB,
        ex_radius=ex_radius,
        rgb=spec.rgb,
        alpha=alpha,
    )
    panel._name = f"detected_box_{spec.name}"
    panel.pose = transform_local_pose(box_homomat, spec.local_center)
    return panel

def make_concave_box_collision_obstacles(
    box_homomat: np.ndarray,
    show_cdprim: bool = True,
    ex_radius: float = DEFAULT_PANEL_EX_RADIUS,
) -> list[object]:


    box = mcm.CollisionModel(
        initor=str(BOX_MODEL_PATH),
        name="detected_hollow_box_collision",
        cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
        ex_radius=ex_radius,
        userdef_cdprim_fn=make_hollow_box_cdprim,
        rgb=np.array([0.15, 0.42, 0.95]),
        alpha=PANEL_ALPHA,
    )
    box.homomat = box_homomat
    if show_cdprim:
        box.show_cdprim()
    return [box]


def make_detected_box_visual_model(box_homomat: np.ndarray, alpha: float = VISUAL_BOX_ALPHA):

    visual = mcm.CollisionModel(
        initor=str(BOX_MODEL_PATH),
        name="detected_box_visual",
        cdprim_type=mcm.const.CDPrimType.SURFACE_BALLS,
        ex_radius=0.008,
        rgb=np.array([0.45, 0.8, 1.0]),
        alpha=alpha,
    )
    visual.homomat = box_homomat
    return visual


def local_probe_points() -> dict[str, np.ndarray]:
    return {
        "free_inner_cavity": np.array([0.0, 0.0, 0.12]),
        "bottom_collision": np.array([0.0, 0.0, 0.015]),
        "x_wall_collision": np.array([-0.285, 0.0, 0.12]),
        "y_wall_collision": np.array([0.0, 0.18, 0.12]),
        "free_above_open_top": np.array([0.0, 0.0, 0.29]),
    }


def describe_panel_specs() -> list[str]:
    lines = []
    for spec in make_box_panel_specs():
        lines.append(
            f"{spec.name}: center={np.round(spec.local_center, 5).tolist()}, "
            f"lengths={np.round(spec.local_lengths, 5).tolist()}"
        )
    return lines