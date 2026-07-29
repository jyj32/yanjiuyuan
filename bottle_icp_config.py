"""Central bottle ICP parameters shared by offline and real pipelines."""

from __future__ import annotations

from pathlib import Path

from yanjiuyuan.mech_eye_ur7e_pointcloud_env import DEFAULT_OUTPUT_ROOT


BOTTLE_STL = Path(__file__).resolve().parent / "models" / "bottle.stl"
CAPTURE_ROOT = DEFAULT_OUTPUT_ROOT

# If CAPTURE_DIR is None, the newest folder under CAPTURE_ROOT is used.
# You may also set CAPTURE_PLY directly to a specific .ply file.
CAPTURE_DIR = None
CAPTURE_PLY = None
PREFER_WORLD_FRAME_PLY = True
TRANSFORM_CAMERA_PLY_TO_WORLD = True

# Crop the target point cloud before registration. Keep these as None until you
# want to isolate only the bottle region, for example X_RANGE=(0.2, 0.5).
X_RANGE = None
Y_RANGE = None
Z_RANGE = (0.04, None)

# Registration settings.
VOXEL_SIZE = 0.005
TEMPLATE_VOXEL_SIZE = 0.005
MODEL_SAMPLE_COUNT = 10000
MODEL_EVEN_RADIUS = None
GLOBAL_RANSAC_N = 3
ICP_MAX_ITERATION = 80

# Output and optional visualization.
OUTPUT_DIR = None
SHOW_RESULT_VIEWER = True
TARGET_POINT_SIZE = 0.002
MODEL_POINT_SIZE = 0.003
