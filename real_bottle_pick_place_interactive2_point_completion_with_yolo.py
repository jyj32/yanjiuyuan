"""
Real-environment interactive pipeline: live capture -> SAM point completion bottle pose ICP -> pick-and-place.

This program is intentionally separate from sim_bottle_pick_place_from_box_object_icp.py.
The sim script stays offline-only; this entry point owns the real robot/camera flow.

WRS key workflow:
  C: check robot/camera, sync robot state, capture a fresh Mech-Eye point cloud, detect the box.
  D: segment the bottle, complete the selected point cloud, run surface-template bottle ICP, show start and place poses.
  P: plan the pick-only approach/depart path and write an RTDE execution plan.
  O: execute the RTDE plan through pick_place_rtde_utils.execute_rtde_execution_plan.

After every O attempt, press C again before any new D/P/O.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from yanjiuyuan.constants import BOX_CAPTURE_ROOT, BOTTLE_ROBOT_SIDE_PLACE_POS, MODEL_DIR  # noqa: E402
from yanjiuyuan import box_object_pointcloud_sam_completion_template_icp_with_yolo as box_object_icp  # noqa: E402
from yanjiuyuan import connection_status as conn_status  # noqa: E402
from yanjiuyuan import pick_place_rtde_utils as rtde_utils  # noqa: E402
from yanjiuyuan import sim_pick_and_place as sim_pick  # noqa: E402
from yanjiuyuan import sync_real_ur7e_mech_eye_box_env as sync_scene  # noqa: E402
from yanjiuyuan.box_collision import (  # noqa: E402
    make_concave_box_collision_obstacles,
    make_detected_box_visual_model,
)


BOX_OBJECT_SCRIPT = Path(__file__).resolve().parent / "box_object_pointcloud_sam_completion_template_icp_with_yolo.py"
DEFAULT_OBJECT_MODEL_PATH = sim_pick.OBJECT_MODEL_PATH
DEFAULT_GRASP_PICKLE_PATH = sim_pick.GRASP_PICKLE_PATH
DEFAULT_GRASP_DIR = DEFAULT_GRASP_PICKLE_PATH.parent
DEFAULT_ROBOT_SIDE_PLACE_POS = np.array([0.55, -0.43, 0.08], dtype=float)
DEFAULT_SAM_TASK_CONFIG_PATH = Path(__file__).resolve().parent / "sam_task_config.json"
DEFAULT_ACTION_SEQUENCE_CONFIG_PATH = Path(__file__).resolve().parent / "action_sequence_config.json"
DEFAULT_COMPLETION_ADAPOINTR_SCRIPT = box_object_icp.completion_matching.DEFAULT_ADAPOINTR_SCRIPT
DEFAULT_COMPLETION_ADAPOINTR_CHECKPOINT = box_object_icp.completion_matching.DEFAULT_ADAPOINTR_CHECKPOINT
PUSH_J2_JOINT_INDEX = 1
DEFAULT_PUSH_J2_RETREAT_TARGET_DEG = -90.0
CAMERA_TO_WORLD = np.array(
    [
        [-0.998885, 0.022034, -0.041751, 0.647500],
        [-0.021796, -0.999744, -0.006148, 0.018000],
        [-0.041876, -0.005231, 0.999109, 1.267000],
        [0.000000, 0.000000, 0.000000, 1.000000],
    ],
    dtype=float,
)
sync_scene.CAM_TO_WORLD = CAMERA_TO_WORLD


@dataclass
class ObjectIcpResult:
    output_dir: Path
    summary_path: Path
    bottle_transform_path: Optional[Path]
    box_transform_path: Path
    bottle_model_path: Path
    global_registered_path: Optional[Path] = None
    bottle_homomat: Optional[np.ndarray] = None


@dataclass
class PlanningResult:
    selected_grasp_index: Optional[int]
    action_sequence: Optional[int]
    mot_data: Any
    pick_pose: tuple[np.ndarray, np.ndarray]
    place_pose: tuple[np.ndarray, np.ndarray]
    object_model_path: Path
    obstacle_names: list[str]
    place_pos_source: str
    place_rot_source: str
    rtde_plan: Optional[rtde_utils.RtdeExecutionPlan] = None
    rtde_plan_path: Optional[Path] = None


@dataclass
class RtdeObjectPose:
    pos: np.ndarray
    rotmat: np.ndarray


# Edit this block directly for the real setup. The program intentionally has no
# command-line tuning surface; run it as:
#     python yanjiuyuan\real_bottle_pick_place_interactive2_point_completion.py
REAL_PIPELINE_CONFIG = {
    # Capture and segmentation/ICP.
    "capture_root": BOX_CAPTURE_ROOT,
    "capture_dir": None,
    "ply": None,
    "image": None,
    "box_transform": None,
    "object_output_dir": None,
    "object_summary": None,
    "mask": None,
    "point": [],
    "sam_task_config": DEFAULT_SAM_TASK_CONFIG_PATH,
    "action_sequence_config": DEFAULT_ACTION_SEQUENCE_CONFIG_PATH,
    "segment_box": None,
    "auto_segment_box": False,
    "backend": "sam",
    "model": None,
    "keep": "best",
    "imgsz": 1024,
    "conf": 0.25,
    "iou": 0.9,
    "device": "0",
    "no_gui": False,
    "show_gui_with_points": False,
    # ---- YOLO 瓶子检测参数 ----
    "yolo_model": Path(__file__).resolve().parent / "models" / "bottle_detect.pt",
    "yolo_conf": 0.7,
    "yolo_iou": 0.5,
    "no_yolo": False,
    "show_object_viewer": False,
    "show_box_model": False,
    "bottle_template": "surface",
    "bottle_template_ply": None,
    "bottle_template_prompt_gui": False,
    "bottle_stl": DEFAULT_OBJECT_MODEL_PATH,
    "bottle_voxel": None,
    "bottle_template_voxel": None,
    "bottle_global_ransac_n": None,
    "bottle_global_ransac_attempts": None,
    "bottle_icp_max_iteration": None,

    # SAM point completion and final bottle pose estimation.
    "completion_matching": True,
    "completion_template": "surface",
    "completion_template_ply": None,
    "completion_adapointr_script": DEFAULT_COMPLETION_ADAPOINTR_SCRIPT,
    "completion_adapointr_checkpoint": DEFAULT_COMPLETION_ADAPOINTR_CHECKPOINT,
    "completion_output_prefix": "real_bottle_completion_surface",
    "completion_device": "cuda:0",
    "completion_global_scale": 0.4,
    "completion_num_points": 1024,
    "completion_num_query": 128,
    "completion_voxel_size": 0.005,
    "completion_template_voxel_size": 0.005,
    "completion_ransac_n": 3,
    "completion_ransac_attempts": 5,
    "completion_icp_max_iteration": 80,
    "completion_network_input_points": 2048,
    "completion_selected_outlier_nb_neighbors": 24,
    "completion_selected_outlier_std_ratio": 1.8,
    "completion_selected_outlier_min_keep_ratio": 0.65,
    "completion_bottle_icp": True,
    "completion_bottle_template": "surface",
    "completion_bottle_template_ply": None,
    "completion_bottle_target_voxel_size": 0.003,
    "completion_bottle_template_voxel_size": 0.003,

    # Robot, camera, and execution.
    "robot_ip": "192.168.125.30",
    "gp_port": "COM3",
    "mock": False,
    "depth_scale": 0.001,
    "depth_trunc": 3.0,
    "scene_max_points": 150000,
    "scene_point_size": 0.002,
    "save_capture_pointclouds": False,
    "rtde_plan_out": None,
    "execute_dry_run": False,
    "max_start_joint_error_deg": 5.0,
    "use_move_l_compliant": False,

    # Compliant pick approach. Tune these here, not from the command line.
    "compliant_force": 10.0,
    "compliant_vel": 0.06,
    "compliant_lateral_tolerance": 0.01,
    "compliant_lateral_stop_tolerance": "auto",
    "compliant_force_frame": "direction",
    "compliant_axes": None,
    "compliant_zero_ft_sensor": False,

    # Planning.
    "skip_plan": False,
    "interactive": None,
    "object_model": None,
    "place_pos": None,
    "place_rpy_deg": None,
    "start_conf_deg": None,
    "open_jaw": None,
    "approach_distance": None,
    "action_sequence": 1,
    "dry_run": False,
    "use_rrt": True,
    "no_env": False,
    "box_obstacle": True,
    "visualize_failure": False,
    "summary_out": None,
}


def make_runtime_config() -> SimpleNamespace:
    config = copy.deepcopy(REAL_PIPELINE_CONFIG)
    return SimpleNamespace(**config)
def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def normalize_paths(args: SimpleNamespace) -> None:
    for attr in (
        "capture_root",
        "capture_dir",
        "ply",
        "image",
        "box_transform",
        "object_output_dir",
        "object_summary",
        "mask",
        "sam_task_config",
        "action_sequence_config",
        "bottle_template_ply",
        "bottle_stl",
        "completion_template_ply",
        "completion_adapointr_script",
        "completion_adapointr_checkpoint",
        "completion_bottle_template_ply",
        "object_model",
        "rtde_plan_out",
        "yolo_model",
        "summary_out",
    ):
        setattr(args, attr, resolve_path(getattr(args, attr)))


def normalize_sam_task_label(label: Any) -> str:
    raw = str(label).strip().lower()
    if raw in {"1", "fg", "front", "pos", "positive", "+"}:
        return "fg"
    if raw in {"0", "bg", "back", "neg", "negative", "-"}:
        return "bg"
    raise ValueError(f"Unknown SAM point label {label!r}; use fg/bg or 1/0.")


def normalize_sam_task_point(point: Any) -> str:
    if isinstance(point, str):
        parts = [part.strip() for part in point.split(",")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"Bad SAM point {point!r}; expected X,Y or X,Y,LABEL.")
        x = int(round(float(parts[0])))
        y = int(round(float(parts[1])))
        label = normalize_sam_task_label(parts[2] if len(parts) == 3 else "fg")
        return f"{x},{y},{label}"
    if not isinstance(point, dict):
        raise ValueError(f"Bad SAM point {point!r}; use a string or an object with x/y/label.")
    if "x" not in point or "y" not in point:
        raise ValueError(f"Bad SAM point {point!r}; missing x or y.")
    x = int(round(float(point["x"])))
    y = int(round(float(point["y"])))
    label = normalize_sam_task_label(point.get("label", "fg"))
    return f"{x},{y},{label}"


def load_action_sequence_config(config_path: Optional[Path]) -> dict[str, Any]:
    if config_path is None:
        return {}
    config_path = resolve_path(Path(config_path))
    if config_path is None or not config_path.exists():
        print(f"[real_pipeline] Warning: action sequence config not found: {config_path}")
        return {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Action sequence config must be a JSON object: {config_path}")
    return data


def current_action_sequence_key(args: SimpleNamespace) -> Optional[str]:
    action_sequence = getattr(args, "action_sequence", None)
    if action_sequence is None:
        return None
    try:
        return str(int(action_sequence))
    except (TypeError, ValueError):
        return str(action_sequence).strip()


def merge_action_sequence_settings(default_settings: Any, sequence_settings: Any) -> dict[str, Any]:
    result = copy.deepcopy(default_settings) if isinstance(default_settings, dict) else {}
    if sequence_settings is None:
        return result
    if not isinstance(sequence_settings, dict):
        raise ValueError(f"Action sequence settings must be a JSON object, got {type(sequence_settings).__name__}.")
    for key, value in sequence_settings.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            merged = copy.deepcopy(result[key])
            merged.update(copy.deepcopy(value))
            result[key] = merged
        else:
            result[key] = copy.deepcopy(value)
    return result


def lookup_action_sequence_settings(data: dict[str, Any], sequence_key: str) -> dict[str, Any]:
    sequences = data.get("action_sequences", data.get("sequences"))
    if sequences is None:
        sequences = {key: value for key, value in data.items() if key != "default"}
    if isinstance(sequences, dict):
        for key, value in sequences.items():
            if str(key) == sequence_key:
                return merge_action_sequence_settings(data.get("default", {}), value)
        return merge_action_sequence_settings(data.get("default", {}), {})
    if isinstance(sequences, list):
        for item in sequences:
            if not isinstance(item, dict):
                continue
            item_key = item.get("action_sequence", item.get("sequence", item.get("id")))
            if item_key is not None and str(item_key) == sequence_key:
                return merge_action_sequence_settings(data.get("default", {}), item)
        return merge_action_sequence_settings(data.get("default", {}), {})
    raise ValueError("Action sequence config field 'action_sequences' must be a JSON object or list.")


def action_sequence_settings_for_args(args: SimpleNamespace) -> dict[str, Any]:
    sequence_key = current_action_sequence_key(args)
    if sequence_key is None:
        return {}
    data = load_action_sequence_config(getattr(args, "action_sequence_config", None))
    if not data:
        return {}
    return lookup_action_sequence_settings(data, sequence_key)


def is_nullish_json_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"})


def parse_vector3(value: Any, field_name: str) -> np.ndarray:
    if isinstance(value, dict):
        try:
            vector = np.array([value["x"], value["y"], value["z"]], dtype=float)
        except KeyError as exc:
            raise ValueError(f"{field_name} dict must contain x, y, and z.") from exc
    elif isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"{field_name} string must contain three comma-separated numbers.")
        vector = np.array([float(part) for part in parts], dtype=float)
    else:
        vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"{field_name} must contain exactly three numbers.")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"{field_name} cannot be a zero vector.")
    return vector / norm


def parse_point3(value: Any, field_name: str) -> np.ndarray:
    if isinstance(value, dict):
        try:
            vector = np.array([value["x"], value["y"], value["z"]], dtype=float)
        except KeyError as exc:
            raise ValueError(f"{field_name} dict must contain x, y, and z.") from exc
    elif isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"{field_name} string must contain three comma-separated numbers.")
        vector = np.array([float(part) for part in parts], dtype=float)
    else:
        vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"{field_name} must contain exactly three numbers.")
    return vector.reshape(3)


def parse_optional_point3(value: Any, field_name: str) -> Optional[np.ndarray]:
    if is_nullish_json_value(value):
        return None
    return parse_point3(value, field_name)


def parse_optional_positive_float(value: Any, field_name: str) -> Optional[float]:
    if is_nullish_json_value(value):
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive, got {parsed}.")
    return parsed


def parse_optional_float(value: Any, field_name: str) -> Optional[float]:
    if is_nullish_json_value(value):
        return None
    return float(value)

def parse_optional_bool(value: Any, field_name: str, default: bool = False) -> bool:
    if is_nullish_json_value(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"{field_name} must be a boolean value.")


def parse_joint_values_deg(value: Any, field_name: str) -> Optional[np.ndarray]:
    if is_nullish_json_value(value):
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        vector = np.array([float(part) for part in parts], dtype=float)
    else:
        vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 6:
        raise ValueError(f"{field_name} must contain exactly six joint values in degrees.")
    return np.radians(vector)


def nested_action_value(settings: dict[str, Any], nested_key: str, flat_keys: tuple[str, ...]) -> Any:
    nested = settings.get(nested_key, {})
    if nested is None:
        nested = {}
    if not isinstance(nested, dict):
        raise ValueError(f"Action sequence {nested_key} must be a JSON object.")
    for key in flat_keys:
        if key in nested and not is_nullish_json_value(nested[key]):
            return nested[key]
    for key in flat_keys:
        if key in settings and not is_nullish_json_value(settings[key]):
            return settings[key]
    for key in flat_keys:
        if key in nested:
            return nested[key]
    for key in flat_keys:
        if key in settings:
            return settings[key]
    return None


def action_sequence_push_settings(args: Optional[SimpleNamespace]) -> Optional[dict[str, Any]]:
    if args is None:
        return None
    settings = action_sequence_settings_for_args(args)
    direction_raw = nested_action_value(settings, "push", ("direction", "push_direction"))
    if is_nullish_json_value(direction_raw):
        return None
    action_sequence = current_action_sequence_key(args)
    direction = parse_vector3(direction_raw, f"action sequence {action_sequence} push.direction")
    distance = parse_optional_positive_float(
        nested_action_value(settings, "push", ("distance", "push_distance")),
        f"action sequence {action_sequence} push.distance",
    )
    if distance is None:
        raise ValueError(f"Action sequence {action_sequence} has push.direction but no push.distance.")
    lift_distance = parse_optional_positive_float(
        nested_action_value(
            settings,
            "push",
            ("lift_distance", "z_lift_distance", "post_push_lift_distance", "push_lift_distance"),
        ),
        f"action sequence {action_sequence} push.lift_distance",
    )
    if lift_distance is None:
        lift_distance = float(sim_pick.PICK_APPROACH_DEPART_DISTANCE)
    clearance_jnt_values = parse_joint_values_deg(
        nested_action_value(
            settings,
            "push",
            (
                "clearance_jnt_values_deg",
                "push_clearance_jnt_values_deg",
                "camera_clearance_jnt_values_deg",
                "camera_clearance_joint_values_deg",
                "clearance_joint_values_deg",
                "clearance_joints_deg",
            ),
        ),
        f"action sequence {action_sequence} push.clearance_jnt_values_deg",
    )
    clearance_tcp_pos = parse_optional_point3(
        nested_action_value(
            settings,
            "push",
            (
                "clearance_tcp_pos",
                "push_clearance_tcp_pos",
                "camera_clearance_tcp_pos",
                "clearance_position",
                "clearance_pos",
            ),
        ),
        f"action sequence {action_sequence} push.clearance_tcp_pos",
    )
    j2_retreat_target_deg = parse_optional_float(
        nested_action_value(
            settings,
            "push",
            (
                "j2_retreat_target_deg",
                "j2_backward_target_deg",
                "camera_clearance_j2_target_deg",
                "camera_clearance_j2_deg",
            ),
        ),
        f"action sequence {action_sequence} push.j2_retreat_target_deg",
    )
    j2_retreat_delta_deg = parse_optional_float(
        nested_action_value(
            settings,
            "push",
            (
                "j2_retreat_delta_deg",
                "j2_backward_delta_deg",
                "camera_clearance_j2_delta_deg",
            ),
        ),
        f"action sequence {action_sequence} push.j2_retreat_delta_deg",
    )
    if clearance_jnt_values is not None:
        clearance_mode = "joint"
    elif clearance_tcp_pos is not None:
        clearance_mode = "tcp"
    else:
        clearance_mode = "j2_retreat"
        if j2_retreat_target_deg is None and j2_retreat_delta_deg is None:
            j2_retreat_target_deg = DEFAULT_PUSH_J2_RETREAT_TARGET_DEG
        if j2_retreat_target_deg is not None and j2_retreat_delta_deg is not None:
            raise ValueError(
                f"Action sequence {action_sequence} push cannot set both "
                "j2_retreat_target_deg and j2_retreat_delta_deg."
            )

    return {
        "direction": direction,
        "distance": distance,
        "lift_distance": lift_distance,
        "clearance_mode": clearance_mode,
        "clearance_jnt_values": clearance_jnt_values,
        "clearance_jnt_values_deg": None if clearance_jnt_values is None else np.degrees(clearance_jnt_values),
        "clearance_tcp_pos": clearance_tcp_pos,
        "j2_retreat_joint_index": PUSH_J2_JOINT_INDEX,
        "j2_retreat_target_deg": j2_retreat_target_deg,
        "j2_retreat_delta_deg": j2_retreat_delta_deg,
        "open_after_push": True,
    }
def apply_action_sequence_start_settings(args: SimpleNamespace) -> Optional[int]:
    data = load_action_sequence_config(getattr(args, "action_sequence_config", None))
    for key in ("start_action_sequence", "initial_action_sequence", "action_sequence_start"):
        if key not in data:
            continue
        value = data[key]
        if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
            args.action_sequence = None
            print("[real_pipeline] start_action_sequence=None from action sequence config")
            return None
        args.action_sequence = int(value)
        print(f"[real_pipeline] start_action_sequence={args.action_sequence} from action sequence config")
        return int(args.action_sequence)
    return None


def raw_grasp_pickle_value_from_action_sequence_settings(settings: dict[str, Any]) -> Optional[Any]:
    for key in (
        "specified_grasp_pickle_file",
        "grasp_pickle_file",
        "grasp_pickle",
        "grasp_pickle_path",
        "grasp_file",
    ):
        if key in settings:
            return settings[key]
    return None


def resolve_grasp_pickle_path(value: Any, config_path: Optional[Path]) -> Path:
    if value is None:
        return DEFAULT_GRASP_PICKLE_PATH
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return DEFAULT_GRASP_PICKLE_PATH
    raw_path = Path(str(value).strip())
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidates: list[Path] = []
    if len(raw_path.parts) == 1:
        candidates.append((DEFAULT_GRASP_DIR / raw_path).resolve())
    if config_path is not None:
        config_path = resolve_path(Path(config_path))
        if config_path is not None:
            candidates.append((config_path.parent / raw_path).resolve())
    candidates.append(resolve_path(raw_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def action_sequence_grasp_pickle_path(args: SimpleNamespace) -> Path:
    settings = action_sequence_settings_for_args(args)
    value = raw_grasp_pickle_value_from_action_sequence_settings(settings)
    return resolve_grasp_pickle_path(value, getattr(args, "action_sequence_config", None))

def raw_points_from_action_sequence_settings(settings: dict[str, Any]) -> list[Any]:
    point_hint = settings.get("point_hint_segment", {})
    if point_hint is None:
        point_hint = {}
    if not isinstance(point_hint, dict):
        raise ValueError("Action sequence point_hint_segment must be a JSON object.")
    raw_points = point_hint.get("point_examples", point_hint.get("points"))
    if raw_points is None:
        raw_points = settings.get("point_examples", settings.get("points", []))
    if raw_points is None:
        return []
    if not isinstance(raw_points, list):
        raise ValueError("Action sequence SAM points must be a list.")
    return raw_points


def action_sequence_sam_task_settings(args: SimpleNamespace) -> Optional[dict[str, Any]]:
    sequence_key = current_action_sequence_key(args)
    if sequence_key is None:
        return None
    settings = action_sequence_settings_for_args(args)
    raw_points = raw_points_from_action_sequence_settings(settings)
    if not raw_points:
        return None
    points = [normalize_sam_task_point(point) for point in raw_points]
    return {
        "task": f"action_sequence_{sequence_key}",
        "mode": "click_no_gui",
        "points": points,
        "no_gui": True,
        "show_gui_with_points": False,
        "config_path": resolve_path(Path(getattr(args, "action_sequence_config"))),
    }


def action_sequence_template_id(settings: dict[str, Any]) -> Optional[str]:
    for key in ("template_pointcloud_id", "bottle_template", "template_id", "registration_template_id"):
        value = settings.get(key)
        if value is None:
            continue
        template_id = str(value).strip()
        if not template_id or template_id.lower() in {"none", "null"}:
            return None
        if template_id == "prompt":
            return template_id
        valid_choices = set(getattr(box_object_icp, "BOTTLE_TEMPLATE_CHOICES", ()))
        if valid_choices and template_id not in valid_choices:
            raise ValueError(
                f"Unknown action sequence bottle template {template_id!r}; "
                f"valid choices are {sorted(valid_choices)}."
            )
        return template_id
    return None


def apply_action_sequence_template_settings(args: SimpleNamespace) -> Optional[str]:
    settings = action_sequence_settings_for_args(args)
    template_id = action_sequence_template_id(settings)
    args.bottle_template = "surface"
    args.bottle_template_prompt_gui = False
    args.completion_bottle_template = "surface"
    return template_id

def load_sam_task_settings(config_path: Optional[Path]) -> Optional[dict[str, Any]]:
    if config_path is None:
        return None
    config_path = resolve_path(Path(config_path))
    if config_path is None or not config_path.exists():
        print(f"[real_pipeline] Warning: SAM task config not found: {config_path}")
        return None
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"SAM task config must be a JSON object: {config_path}")
    active_task = str(data.get("active_task", "manual_click")).strip()
    tasks = data.get("tasks", {})
    if isinstance(tasks, dict) and active_task in tasks:
        task = tasks[active_task]
    elif "mode" in data:
        task = data
        active_task = str(data.get("mode", active_task)).strip()
    else:
        raise KeyError(f"SAM task {active_task!r} was not found in {config_path}")
    if not isinstance(task, dict):
        raise ValueError(f"SAM task {active_task!r} must be a JSON object.")
    mode = str(task.get("mode", active_task)).strip().lower().replace("-", "_")
    raw_points = task.get("points", [])
    if raw_points is None:
        raw_points = []
    if not isinstance(raw_points, list):
        raise ValueError(f"SAM task {active_task!r} points must be a list.")
    points = [normalize_sam_task_point(point) for point in raw_points]
    if mode in {"manual", "manual_click"}:
        no_gui = False
        show_gui_with_points = False
        points = []
    elif mode in {"click_with_gui", "points_with_gui", "gui_points"}:
        if not points:
            raise ValueError(f"SAM task {active_task!r} requires at least one point.")
        no_gui = False
        show_gui_with_points = True
    elif mode in {"click_no_gui", "points_no_gui", "no_gui_points"}:
        if not points:
            raise ValueError(f"SAM task {active_task!r} requires at least one point.")
        no_gui = True
        show_gui_with_points = False
    else:
        raise ValueError(f"Unknown SAM task mode {mode!r}; use manual_click, click_with_gui, or click_no_gui.")
    return {
        "task": active_task,
        "mode": mode,
        "points": points,
        "no_gui": no_gui,
        "show_gui_with_points": show_gui_with_points,
        "config_path": config_path,
    }


def apply_sam_task_settings(args: SimpleNamespace, settings: dict[str, Any]) -> None:
    args.point = list(settings["points"])
    args.no_gui = bool(settings["no_gui"])
    args.show_gui_with_points = bool(settings["show_gui_with_points"])


def refresh_sam_task_settings(args: SimpleNamespace) -> Optional[dict[str, Any]]:
    template_id = apply_action_sequence_template_settings(args)
    settings = action_sequence_sam_task_settings(args)
    if settings is None:
        settings = load_sam_task_settings(getattr(args, "sam_task_config", None))
    if settings is None:
        if template_id is not None:
            print(
                f"[real_pipeline] Action sequence {current_action_sequence_key(args)}: "
                f"template_pointcloud_id={template_id} ignored; completion template=surface.",
                flush=True,
            )
        return None
    apply_sam_task_settings(args, settings)
    point_text = " ".join(settings["points"]) if settings["points"] else "manual clicks"
    gui_text = "no GUI" if settings["no_gui"] else ("GUI with points" if settings["show_gui_with_points"] else "manual GUI")
    template_text = "" if template_id is None else f"; template_pointcloud_id={template_id} ignored; completion template=surface"
    print(
        f"[real_pipeline] SAM task {settings['task']} ({settings['mode']}): "
        f"{gui_text}; points={point_text}; config={settings['config_path']}{template_text}",
        flush=True,
    )
    return settings


def append_path(cmd: list[str], flag: str, path: Optional[Path]) -> None:
    if path is not None:
        cmd.extend([flag, str(path)])


def append_value(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def run_command(label: str, cmd: list[str]) -> None:
    print(f"[real_pipeline] {label}:")
    print(f"  {subprocess.list2cmdline(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def resolve_object_output_dir(args: SimpleNamespace) -> Path:
    if args.object_output_dir is not None:
        return args.object_output_dir
    if args.capture_dir is not None:
        return args.capture_dir / "box_object_extraction"
    return Path.cwd() / "box_object_extraction"


def build_box_object_args(args: SimpleNamespace, output_dir: Path) -> SimpleNamespace:
    box_args = SimpleNamespace(
        capture_root=args.capture_root,
        capture_dir=args.capture_dir,
        ply=args.ply,
        prefer_world_ply=True,
        image=args.image,
        box_transform=args.box_transform,
        output_dir=output_dir,
        mask=args.mask,
        point=list(args.point),
        segment_box=args.segment_box,
        auto_segment_box=args.auto_segment_box,
        backend=args.backend,
        model=args.model,
        keep=args.keep,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_display=1400,
        no_gui=args.no_gui,
        show_gui_with_points=bool(getattr(args, "show_gui_with_points", False)),
        # YOLO 瓶子检测参数
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        no_yolo=args.no_yolo,
        inner_xy_margin=box_object_icp.BOX_OBJECT_INNER_XY_MARGIN,
        top_overhang_xy_margin=box_object_icp.BOX_OBJECT_TOP_OVERHANG_XY_MARGIN,
        above_top_margin=box_object_icp.BOX_OBJECT_ABOVE_TOP_MARGIN,
        remove_blue_box_points=box_object_icp.BOX_OBJECT_REMOVE_BLUE_BOX_POINTS,
        show_removed_context=box_object_icp.BOX_OBJECT_SHOW_REMOVED_CONTEXT,
        removed_context_xy_margin=box_object_icp.BOX_OBJECT_REMOVED_CONTEXT_XY_MARGIN,
        removed_context_z_margin=box_object_icp.BOX_OBJECT_REMOVED_CONTEXT_Z_MARGIN,
        pixel_color_tolerance=box_object_icp.BOX_OBJECT_PIXEL_COLOR_TOLERANCE,
        pixel_mapping_min_ratio=box_object_icp.BOX_OBJECT_PIXEL_MAPPING_MIN_RATIO,
        candidate_voxel=box_object_icp.BOX_OBJECT_CANDIDATE_VOXEL_DOWNSAMPLE,
        removed_voxel=box_object_icp.BOX_OBJECT_REMOVED_VOXEL_DOWNSAMPLE,
        selected_voxel=box_object_icp.BOX_OBJECT_SELECTED_VOXEL_DOWNSAMPLE,
        show_viewer=True,
        show_box_model=args.show_box_model,
        point_size=box_object_icp.BOX_OBJECT_POINT_SIZE,
        point_hint_model=getattr(args, "point_hint_model", None),
        point_hint_model_key=getattr(args, "point_hint_model_key", None),
        _yolo_detector=getattr(args, "_yolo_detector", None),
        _yolo_model_key=getattr(args, "_yolo_model_key", None),
    )
    bottle_config = box_object_icp.bottle_icp_config_from_runtime_options(
        args,
        {"enabled": False, "template": "surface", "template_prompt_gui": False},
    )
    for option_name, config_field in box_object_icp.BOTTLE_ICP_CONFIG_FIELDS.items():
        setattr(box_args, option_name, getattr(bottle_config, config_field))

    completion_option_names = (
        "completion_matching",
        "completion_template",
        "completion_template_ply",
        "completion_adapointr_script",
        "completion_adapointr_checkpoint",
        "completion_output_prefix",
        "completion_device",
        "completion_global_scale",
        "completion_num_points",
        "completion_num_query",
        "completion_voxel_size",
        "completion_template_voxel_size",
        "completion_ransac_n",
        "completion_ransac_attempts",
        "completion_icp_max_iteration",
        "completion_network_input_points",
        "completion_selected_outlier_nb_neighbors",
        "completion_selected_outlier_std_ratio",
        "completion_selected_outlier_min_keep_ratio",
        "completion_bottle_icp",
        "completion_bottle_template",
        "completion_bottle_template_ply",
        "completion_bottle_target_voxel_size",
        "completion_bottle_template_voxel_size",
    )
    for option_name in completion_option_names:
        setattr(box_args, option_name, getattr(args, option_name))
    return box_args

def prepare_interactive_pipeline_context(
    args: SimpleNamespace,
    capture: Optional[box_object_icp.CaptureData] = None,
) -> box_object_icp.PipelineContext:
    output_dir = resolve_object_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    box_args = build_box_object_args(args, output_dir)
    if capture is not None:
        return box_object_icp.prepare_pipeline_context_from_capture(box_args, capture)
    return box_object_icp.prepare_pipeline_context(box_args)

def make_sync_capture_args(args: SimpleNamespace, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        robot_ip=args.robot_ip,
        gp_port=args.gp_port,
        mock=args.mock,
        ply=None,
        ply_frame="auto",
        output_root=args.capture_root,
        output_dir=output_dir,
        ply_out=None,
        save_ply=bool(getattr(args, "save_capture_pointclouds", True)),
        depth_scale=args.depth_scale,
        depth_trunc=args.depth_trunc,
        detect_box=True,
        box_transform_out=output_dir / "detected_box_transform.txt",
        summary_out=output_dir / "robot_camera_box_summary.txt",
    )


def capture_synced_context(args: SimpleNamespace) -> tuple[box_object_icp.PipelineContext, dict]:
    output_dir = sync_scene.make_output_dir(args.capture_root, None)
    sync_args = make_sync_capture_args(args, output_dir)
    provider = sync_scene.make_robot_provider(sync_args)
    box_detection = None
    current_jnt_values = None
    current_tcp_pos = None
    current_jaw_width = None
    try:
        robot_status = conn_status.check_robot_provider(provider, mock=args.mock)
        conn_status.print_status(robot_status, prefix="[real_pipeline]")
        if not robot_status.ok:
            raise ConnectionError(robot_status.line())

        sync_scene.read_robot_snapshot(provider, "Initial robot")
        # Suspend RTDE before camera capture to prevent the C++ background
        # receiver thread from auto-reconnecting and crashing (0xC0000005)
        # when the connection drops during the long camera operation.
        provider.suspend()
        pcd, rgb_path, colored_ply_path, camera_status, pixel_indices = conn_status.capture_mech_eye_pointcloud_checked(
            output_dir=output_dir,
            ply_out=sync_args.ply_out,
            depth_scale=sync_args.depth_scale,
            depth_trunc=sync_args.depth_trunc,
            save_ply=sync_args.save_ply,
            return_pixel_indices=True,
        )
        conn_status.print_status(conn_status.LiveConnectionStatus([camera_status]), prefix="[real_pipeline]")

        points, colors = sync_scene.open3d_to_numpy(pcd)
        raw_count = len(points)
        points_world = sync_scene.transform_points(points, CAMERA_TO_WORLD)
        world_ply_path = output_dir / "world_colored_pointcloud.ply" if sync_args.save_ply else None
        if world_ply_path is not None:
            sync_scene.save_numpy_pointcloud(points_world, colors, world_ply_path)
        print(f"Saved RGB image to: {rgb_path}")
        if colored_ply_path is not None:
            print(f"Saved camera-frame colored point cloud to: {colored_ply_path}")
        if world_ply_path is not None:
            print(f"Saved world-frame colored point cloud to: {world_ply_path}")
        else:
            print("[real_pipeline] Skipping point-cloud PLY writes; using the in-memory capture.")
        sync_scene.print_camera_info(frame="camera", points_world=points_world, source_path=colored_ply_path)
        scene_data = sync_scene.CameraSceneData(
            points_world=points_world,
            colors=colors,
            raw_point_count=raw_count,
            frame="camera",
            rgb_path=rgb_path,
            colored_ply_path=colored_ply_path,
            world_ply_path=world_ply_path,
            output_dir=output_dir,
            pixel_indices=pixel_indices,
        )

        box_detection = sync_scene.detect_box(sync_args, scene_data)
        if box_detection is None:
            raise RuntimeError("C sync did not detect the box pose.")
        detected_box_transform = sync_args.box_transform_out
        if detected_box_transform is None or not Path(detected_box_transform).exists():
            raise FileNotFoundError(f"C sync did not write detected box transform: {detected_box_transform}")
        # Resume RTDE with fresh connections for the final robot state read.
        provider.resume()
        current_jnt_values, current_tcp_pos, current_jaw_width = sync_scene.read_robot_snapshot(provider, "Scene robot")
        sync_scene.write_summary(
            sync_args,
            scene_data,
            current_jnt_values,
            current_tcp_pos,
            current_jaw_width,
            box_detection,
        )
    finally:
        provider.close()

    args.capture_dir = output_dir
    args.image = scene_data.rgb_path if scene_data.rgb_path is not None else args.image
    args.ply = scene_data.world_ply_path if scene_data.world_ply_path is not None else scene_data.colored_ply_path
    detected_box_transform = output_dir / "detected_box_transform.txt"
    args.box_transform = detected_box_transform if detected_box_transform.exists() else None
    args.object_summary = None
    args.object_output_dir = output_dir / "box_object_extraction"
    if current_jnt_values is not None:
        args.start_conf_deg = np.degrees(np.asarray(current_jnt_values, dtype=float))

    if scene_data.rgb_path is None:
        raise RuntimeError("C sync did not save an RGB image for SAM segmentation.")
    capture = box_object_icp.CaptureData(
        pcd_world=None,
        points_world=scene_data.points_world,
        colors=scene_data.colors,
        target_ply=scene_data.world_ply_path,
        camera_ply=scene_data.colored_ply_path,
        rgb_path=scene_data.rgb_path,
        capture_dir=output_dir,
        frame="world" if scene_data.world_ply_path is not None else "world_memory",
        pixel_indices=scene_data.pixel_indices,
    )
    ctx = prepare_interactive_pipeline_context(args, capture=capture)
    metadata = {
        "output_dir": output_dir,
        "scene_data": scene_data,
        "box_detection": box_detection,
        "current_jnt_values": current_jnt_values,
        "current_tcp_pos": current_tcp_pos,
        "current_jaw_width": current_jaw_width,
        "robot_status": robot_status,
        "camera_status": camera_status,
    }
    return ctx, metadata


def find_close_index(mot_data) -> Optional[int]:
    values = [None if value is None else float(np.asarray(value).reshape(-1)[0]) for value in mot_data.ev_list]
    for index in range(1, len(values)):
        prev = values[index - 1]
        current = values[index]
        if prev is not None and current is not None and current < prev - 1e-5:
            return index
    return None


def infer_selected_grasp_index(robot, grasps, mot_data, pick_pose: tuple[np.ndarray, np.ndarray]) -> Optional[int]:
    if not grasps or len(mot_data.jv_list) == 0:
        return None
    close_index = find_close_index(mot_data)
    if close_index is None:
        close_index = min(range(len(mot_data.jv_list)), key=lambda idx: abs(idx - len(mot_data.jv_list) // 3))
    tcp_pos, _tcp_rotmat = robot.fk(jnt_values=np.asarray(mot_data.jv_list[close_index], dtype=float))
    tcp_pos = np.asarray(tcp_pos, dtype=float)
    distances = []
    for grasp in grasps:
        grasp_tcp_pos, _grasp_tcp_rotmat = sim_pick.tcp_pose_from_object_pose(pick_pose, grasp)
        distances.append(float(np.linalg.norm(np.asarray(grasp_tcp_pos, dtype=float) - tcp_pos)))
    selected_index = int(np.argmin(distances))
    sim_pick.debug_print(
        f"Inferred selected grasp from close frame {close_index}: "
        f"pickle_grasp_{selected_index}, tcp_error={distances[selected_index]:.6f} m"
    )
    return selected_index


def grasp_jaw_width(grasp) -> float:
    return float(np.asarray(grasp.ee_values, dtype=float).reshape(-1)[0])


def pick_approach_jaw_width_for_grasp(robot, grasp, override_jaw_width: Optional[float] = None) -> float:
    jaw_width = grasp_jaw_width(grasp) * 1.2 if override_jaw_width is None else float(override_jaw_width)
    return sim_pick.clamp_jaw_width(robot, jaw_width)


def decorate_grasp_for_rtde(robot, grasp, grasp_index: int, args: Optional[SimpleNamespace] = None):
    jaw_width = grasp_jaw_width(grasp)
    push_settings = action_sequence_push_settings(args)
    pick_approach_jaw_width_override = pick_approach_jaw_width_override_for_action_sequence(args)
    approach_distance_override = pick_approach_distance_for_action_sequence(args)
    approach_distance = float(sim_pick.APPROACH_DISTANCE if approach_distance_override is None else approach_distance_override)
    if push_settings is None:
        pick_approach_jaw_width = pick_approach_jaw_width_for_grasp(robot, grasp, pick_approach_jaw_width_override)
    else:
        pick_approach_jaw_width = sim_pick.clamp_jaw_width(robot, jaw_width)
    setattr(grasp, "name", f"pickle_grasp_{grasp_index}")
    setattr(grasp, "jaw_width", jaw_width)
    setattr(grasp, "pick_approach_jaw_width", pick_approach_jaw_width)
    setattr(grasp, "approach_distance", approach_distance)
    return grasp

def build_and_save_rtde_plan(args: SimpleNamespace, planning: PlanningResult, default_dir: Path) -> tuple[rtde_utils.RtdeExecutionPlan, Path]:
    robot = sim_pick.make_robot()
    grasps = sim_pick.load_grasps(robot, sim_pick.GRASP_PICKLE_PATH)
    grasp_index = planning.selected_grasp_index
    if grasp_index is None:
        grasp_index = infer_selected_grasp_index(robot, grasps, planning.mot_data, planning.pick_pose)
    if grasp_index is None or grasp_index < 0 or grasp_index >= len(grasps):
        raise RuntimeError("Could not infer selected grasp for RTDE execution plan.")
    planning.selected_grasp_index = grasp_index
    grasp = decorate_grasp_for_rtde(robot, grasps[grasp_index], grasp_index, args)
    path_type = getattr(planning.mot_data, "path_type", "pick_open")
    planner_name = "WRS pick-only planner + JSON push path" if path_type == "push" else "WRS pick-only planner + first-joint alignment"
    rtde_plan = rtde_utils.build_pick_place_rtde_plan(
        robot=robot,
        mot_data=planning.mot_data,
        pick_pose=RtdeObjectPose(pos=planning.pick_pose[0], rotmat=planning.pick_pose[1]),
        place_pose=RtdeObjectPose(pos=planning.place_pose[0], rotmat=planning.place_pose[1]),
        grasp=grasp,
        planner_name=planner_name,
        compliant_force=float(args.compliant_force),
        compliant_vel=float(args.compliant_vel),
        compliant_lateral_tolerance=float(args.compliant_lateral_tolerance),
        compliant_lateral_stop_tolerance=args.compliant_lateral_stop_tolerance,
        compliant_force_frame=args.compliant_force_frame,
        compliant_axes=args.compliant_axes,
        compliant_zero_ft_sensor=bool(args.compliant_zero_ft_sensor),
    )
    rtde_plan.metadata["path_type"] = path_type
    push_settings = getattr(planning.mot_data, "push_settings", None)
    if push_settings is not None:
        rtde_plan.metadata["push_settings"] = push_settings
    rtde_plan_path = args.rtde_plan_out
    if rtde_plan_path is None:
        rtde_plan_path = default_dir / "pick_place_rtde_plan.json"
    rtde_utils.save_rtde_execution_plan(rtde_plan, rtde_plan_path)
    sim_pick.debug_print(f"Saved RTDE execution plan: {rtde_plan_path}")
    return rtde_plan, rtde_plan_path

def bottle_pose_summary_from_summary(summary: dict) -> tuple[dict, str]:
    completion_summary = summary.get("completion_matching") or {}
    if isinstance(completion_summary, dict):
        completion_bottle_summary = completion_summary.get("completion_bottle_icp")
        if isinstance(completion_bottle_summary, dict) and completion_bottle_summary:
            return completion_bottle_summary, "completion_bottle_icp"

    bottle_summary = summary.get("bottle_icp")
    if isinstance(bottle_summary, dict) and bottle_summary:
        return bottle_summary, "bottle_icp"
    raise RuntimeError("Detection did not produce a bottle pose result; press D again after selecting an object.")


def _homomat_from_summary_value(value: object, label: str) -> np.ndarray:
    homomat = np.asarray(value, dtype=float)
    if homomat.shape != (4, 4):
        raise RuntimeError(f"{label} transform must be 4x4, got {homomat.shape}.")
    if not np.all(np.isfinite(homomat)):
        raise RuntimeError(f"{label} transform contains NaN or inf.")
    return homomat


def bottle_homomat_from_pose_summary(bottle_summary: dict, source_name: str) -> np.ndarray:
    if source_name != "completion_bottle_icp":
        return load_homomat(Path(bottle_summary["icp_transform_path"]), "bottle ICP")

    world_transform = bottle_summary.get("icp_transform_world")
    if world_transform is not None:
        return _homomat_from_summary_value(world_transform, "completion bottle world ICP")

    local_transform = bottle_summary.get("icp_transform_local") or bottle_summary.get("icp_transform")
    if local_transform is None:
        local_path = bottle_summary.get("icp_transform_local_path") or bottle_summary.get("icp_transform_path")
        if local_path is not None and Path(local_path).exists():
            local_transform = np.loadtxt(local_path)

    if local_transform is not None and bottle_summary.get("registration_frame") == "world":
        return _homomat_from_summary_value(local_transform, "completion bottle world ICP")

    local_to_world = bottle_summary.get("local_to_world_transform")
    if local_transform is not None and local_to_world is not None:
        return _homomat_from_summary_value(
            local_to_world,
            "completion bottle local_to_world",
        ) @ _homomat_from_summary_value(local_transform, "completion bottle local ICP")

    world_path = bottle_summary.get("icp_transform_world_path")
    if world_path is not None and Path(world_path).exists():
        return load_homomat(Path(world_path), "completion bottle world ICP")

    if local_transform is not None:
        return _homomat_from_summary_value(local_transform, "completion bottle ICP (fallback)")

    raise RuntimeError(
        "Completion bottle ICP did not provide a usable world-frame pose. "
        "Run D again with the latest box_object completion ICP code."
    )


def bottle_transform_path_from_pose_summary(bottle_summary: dict, source_name: str) -> Optional[Path]:
    if source_name == "completion_bottle_icp":
        path = bottle_summary.get("icp_transform_world_path")
        return None if path is None else Path(path)
    return Path(bottle_summary["icp_transform_path"])

def global_registered_path_from_pose_summary(bottle_summary: dict, source_name: str) -> Optional[Path]:
    if source_name == "completion_bottle_icp":
        path = (
            bottle_summary.get("global_registered_world_path")
            or bottle_summary.get("icp_registered_world_path")
            or bottle_summary.get("global_registered_path")
        )
    else:
        path = bottle_summary.get("global_registered_path")
    return None if path is None else Path(path)


def object_icp_result_from_summary(summary: dict) -> ObjectIcpResult:
    summary_path = Path(summary["summary_path"])
    bottle_summary, source_name = bottle_pose_summary_from_summary(summary)
    bottle_model_path = Path(bottle_summary.get("bottle_stl") or DEFAULT_OBJECT_MODEL_PATH)
    bottle_homomat = bottle_homomat_from_pose_summary(bottle_summary, source_name)
    result = ObjectIcpResult(
        output_dir=summary_path.parent,
        summary_path=summary_path,
        bottle_transform_path=bottle_transform_path_from_pose_summary(bottle_summary, source_name),
        box_transform_path=Path(summary["box_transform_used"]),
        bottle_model_path=bottle_model_path,
        global_registered_path=global_registered_path_from_pose_summary(bottle_summary, source_name),
        bottle_homomat=bottle_homomat,
    )
    for label, path in (
        ("summary", result.summary_path),
        ("box transform", result.box_transform_path),
        ("bottle model", result.bottle_model_path),
        *( () if result.global_registered_path is None else (("global registered template points", result.global_registered_path),) ),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    return result

def read_object_summary(summary_path: Path) -> ObjectIcpResult:
    if not summary_path.exists():
        raise FileNotFoundError(f"Object summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("summary_path", str(summary_path))
    return object_icp_result_from_summary(summary)

def build_extraction_command(args: SimpleNamespace, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(BOX_OBJECT_SCRIPT),
        "--capture-root",
        str(args.capture_root),
        "--output-dir",
        str(output_dir),
        "--backend",
        args.backend,
        "--keep",
        args.keep,
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
    ]
    append_path(cmd, "--capture-dir", args.capture_dir)
    append_path(cmd, "--ply", args.ply)
    append_path(cmd, "--image", args.image)
    append_path(cmd, "--box-transform", args.box_transform)
    append_path(cmd, "--mask", args.mask)
    append_value(cmd, "--segment-box", args.segment_box)
    append_value(cmd, "--model", args.model)
    append_value(cmd, "--device", args.device)
    bottle_args = copy.copy(args)
    bottle_args.bottle_icp = False
    bottle_args.bottle_template = "surface"
    bottle_args.bottle_template_prompt_gui = False
    box_object_icp.append_bottle_icp_cli_args(cmd, bottle_args)
    cmd.append("--completion-matching")
    cmd.extend(["--completion-template", "surface"])
    append_path(cmd, "--completion-adapointr-script", args.completion_adapointr_script)
    append_path(cmd, "--completion-adapointr-checkpoint", args.completion_adapointr_checkpoint)
    append_value(cmd, "--completion-output-prefix", args.completion_output_prefix)
    append_value(cmd, "--completion-device", args.completion_device)
    append_value(cmd, "--completion-global-scale", args.completion_global_scale)
    append_value(cmd, "--completion-num-points", args.completion_num_points)
    append_value(cmd, "--completion-num-query", args.completion_num_query)
    append_value(cmd, "--completion-voxel-size", args.completion_voxel_size)
    append_value(cmd, "--completion-template-voxel-size", args.completion_template_voxel_size)
    append_value(cmd, "--completion-ransac-n", args.completion_ransac_n)
    append_value(cmd, "--completion-ransac-attempts", args.completion_ransac_attempts)
    append_value(cmd, "--completion-icp-max-iteration", args.completion_icp_max_iteration)
    append_value(cmd, "--completion-network-input-points", args.completion_network_input_points)
    append_value(cmd, "--completion-selected-outlier-nb-neighbors", args.completion_selected_outlier_nb_neighbors)
    append_value(cmd, "--completion-selected-outlier-std-ratio", args.completion_selected_outlier_std_ratio)
    append_value(cmd, "--completion-selected-outlier-min-keep-ratio", args.completion_selected_outlier_min_keep_ratio)
    cmd.append("--completion-bottle-icp")
    cmd.extend(["--completion-bottle-template", "surface"])
    append_value(cmd, "--completion-bottle-target-voxel-size", args.completion_bottle_target_voxel_size)
    append_value(cmd, "--completion-bottle-template-voxel-size", args.completion_bottle_template_voxel_size)
    for point in args.point:
        cmd.extend(["--point", point])
    cmd.append("--auto-segment-box" if args.auto_segment_box else "--no-auto-segment-box")
    cmd.append("--show-viewer" if args.show_object_viewer else "--no-show-viewer")
    cmd.append("--show-box-model" if args.show_box_model else "--no-show-box-model")
    if args.no_gui:
        cmd.append("--no-gui")
    if getattr(args, "show_gui_with_points", False):
        cmd.append("--show-gui-with-points")
    # YOLO 瓶子检测参数
    if getattr(args, "no_yolo", False):
        cmd.append("--no-yolo")
    else:
        append_path(cmd, "--yolo-model", args.yolo_model)
        append_value(cmd, "--yolo-conf", args.yolo_conf)
        append_value(cmd, "--yolo-iou", args.yolo_iou)
    return cmd


def run_or_reuse_object_icp(args: SimpleNamespace) -> ObjectIcpResult:
    if args.object_summary is not None:
        return read_object_summary(args.object_summary)

    output_dir = resolve_object_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_extraction_command(args, output_dir)
    run_command("run box/object extraction and completion bottle ICP", cmd)
    return read_object_summary(output_dir / "box_object_extraction_summary.json")


def load_homomat(path: Path, label: str) -> np.ndarray:
    homomat = np.asarray(np.loadtxt(path), dtype=float)
    if homomat.shape != (4, 4):
        raise ValueError(f"{label} transform must be 4x4, got {homomat.shape}: {path}")
    if not np.all(np.isfinite(homomat)):
        raise ValueError(f"{label} transform contains NaN or inf: {path}")
    return homomat


def bottle_homomat_for_icp(icp: ObjectIcpResult) -> np.ndarray:
    if icp.bottle_homomat is not None:
        return np.asarray(icp.bottle_homomat, dtype=float)
    if icp.bottle_transform_path is None:
        raise RuntimeError("Bottle pose is missing; press D to estimate the bottle pose again.")
    return load_homomat(icp.bottle_transform_path, "bottle ICP")


def homomat_to_pose(homomat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return homomat[:3, 3].copy(), homomat[:3, :3].copy()


def resolve_place_pose(
    args: SimpleNamespace,
    pick_pose: tuple[np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray], str, str]:
    if args.place_pos is None:
        place_pos = np.asarray(BOTTLE_ROBOT_SIDE_PLACE_POS, dtype=float).copy()
        place_pos_source = f"constants.BOTTLE_ROBOT_SIDE_PLACE_POS {sim_pick.format_vec(BOTTLE_ROBOT_SIDE_PLACE_POS, digits=3)}"
    else:
        place_pos = np.asarray(args.place_pos, dtype=float)
        place_pos_source = "--place-pos"

    if args.place_rpy_deg is None:
        place_rotmat = pick_pose[1].copy()
        place_rot_source = "keep ICP pick orientation"
    else:
        _place_pos, place_rotmat = sim_pick.pose_from_pos_rpy(place_pos, args.place_rpy_deg)
        place_rot_source = f"--place-rpy-deg {sim_pick.format_vec(args.place_rpy_deg, digits=3)}"
    return (place_pos, place_rotmat), place_pos_source, place_rot_source

ROBOT_SIDE_PLACE_BOX_XYZ_LENGTHS = np.array([0.20, 0.2, 0.1], dtype=float)
ROBOT_SIDE_PLACE_BOX_EX_RADIUS = 0.005
ROBOT_SIDE_PLACE_BOX_RGB = np.array([1.0, 0.62, 0.2], dtype=float)
ROBOT_SIDE_PLACE_BOX_ALPHA = 0.28
def make_robot_side_place_box_collision_obstacle(show_cdprim: bool = False):
    from wrs import mcm, mgm

    place_pos = np.asarray(BOTTLE_ROBOT_SIDE_PLACE_POS, dtype=float).reshape(3)
    box_sgm = mgm.gen_box(
        xyz_lengths=ROBOT_SIDE_PLACE_BOX_XYZ_LENGTHS,
        rgb=ROBOT_SIDE_PLACE_BOX_RGB,
        alpha=ROBOT_SIDE_PLACE_BOX_ALPHA,
    )
    place_box = mcm.CollisionModel(
        box_sgm,
        name="robot_side_place_box_collision",
        cdprim_type=mcm.const.CDPrimType.AABB,
        ex_radius=ROBOT_SIDE_PLACE_BOX_EX_RADIUS,
        rgb=ROBOT_SIDE_PLACE_BOX_RGB,
        alpha=ROBOT_SIDE_PLACE_BOX_ALPHA,
    )
    place_box._name = "robot_side_place_box_collision"
    place_box.pose = (place_pos, np.eye(3))
    place_box.show_cdprim()
    return place_box

def build_obstacle_lists(
    args: SimpleNamespace,
    box_homomat: np.ndarray,
    include_display: bool = True,
) -> tuple[list[object], list[object]]:
    planning_obstacles: list[object] = []
    display_obstacles: list[object] = []
    if not args.no_env:
        table = sim_pick.make_table_obstacle()
        planning_obstacles.append(table)
        if include_display:
            display_obstacles.append(table)
    # 设置箱子碰撞体
    detected_box_panels = make_concave_box_collision_obstacles(box_homomat, show_cdprim=include_display)
    if include_display:
        display_obstacles.append(make_detected_box_visual_model(box_homomat))
        display_obstacles.extend(detected_box_panels)
    if args.box_obstacle:
        planning_obstacles.extend(detected_box_panels)
    robot_side_place_box = make_robot_side_place_box_collision_obstacle(show_cdprim=include_display)
    # robot_side_place_box.attach_to(base)
    planning_obstacles.append(robot_side_place_box)
    if include_display:
        display_obstacles.append(robot_side_place_box)
    return planning_obstacles, display_obstacles


def precheck_start_collision(robot, obstacle_list: list[object]) -> None:
    if not obstacle_list:
        return
    robot.backup_state()
    try:
        collided = bool(robot.is_collided(obstacle_list=obstacle_list, toggle_contacts=False))
    finally:
        robot.restore_state()
    if collided:
        names = [getattr(obstacle, "name", type(obstacle).__name__) for obstacle in obstacle_list]
        sim_pick.debug_print(
            "WARNING: start configuration is already in collision with planning obstacles "
            f"{names}. If this offline default is not the real robot start, pass --start-conf-deg."
        )


def configure_pick_place(args: SimpleNamespace, object_model_path: Path) -> None:
    sim_pick.IK_SOLVER = "ikfast"
    sim_pick.OBJECT_MODEL_PATH = object_model_path
    sim_pick.GRASP_PICKLE_PATH = action_sequence_grasp_pickle_path(args)
    sim_pick.USE_RRT = bool(args.use_rrt)
    sim_pick.VISUALIZE_RESULT = not bool(args.dry_run)
    sim_pick.VISUALIZE_FAILURE = bool(args.visualize_failure)
    sim_pick.USE_REASONED_COMMON_GRASPS = True
    sim_pick.PRINT_WRS_SUMMARY = True
    sim_pick.DEBUG_IKFAST_FRAME = False
    if args.open_jaw is not None:
        sim_pick.OPEN_JAW_WIDTH = float(args.open_jaw)
    if args.approach_distance is not None:
        sim_pick.APPROACH_DISTANCE = float(args.approach_distance)
    if args.start_conf_deg is not None:
        sim_pick.DEFAULT_HOME_CONF = np.radians(np.asarray(args.start_conf_deg, dtype=float))

def expand_grasp_index_ranges(range_specs: Any) -> set[int]:
    blocked_indices: set[int] = set()
    if range_specs is None:
        return blocked_indices
    if isinstance(range_specs, (range, int, np.integer)):
        range_specs = [range_specs]
    elif isinstance(range_specs, (list, tuple)) and len(range_specs) == 2:
        nested_range_types = (list, tuple, range, set)
        if not any(isinstance(item, nested_range_types) for item in range_specs):
            range_specs = [range_specs]
    for spec in range_specs:
        if isinstance(spec, range):
            blocked_indices.update(int(index) for index in spec)
            continue
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            start, end = (int(spec[0]), int(spec[1]))
            step = 1 if start <= end else -1
            blocked_indices.update(range(start, end + step, step))
            continue
        blocked_indices.add(int(spec))
    return blocked_indices


def blocked_grasp_specs_for_action_sequence(args: SimpleNamespace) -> Any:
    settings = action_sequence_settings_for_args(args)
    for key in ("blocked_grasp_ranges", "blocked_grasps", "blocked_grasp_indices"):
        if key in settings:
            return settings[key]
    action_sequence = getattr(args, "action_sequence", None)
    blocked_by_sequence = getattr(args, "blocked_grasp_ranges_by_action_sequence", {}) or {}
    if action_sequence is None:
        return []
    return blocked_by_sequence.get(action_sequence, blocked_by_sequence.get(str(action_sequence), []))


def blocked_grasp_indices_for_action_sequence(args: SimpleNamespace, grasp_count: Optional[int] = None) -> list[int]:
    action_sequence = getattr(args, "action_sequence", None)
    if action_sequence is None:
        return []
    blocked_indices = expand_grasp_index_ranges(blocked_grasp_specs_for_action_sequence(args))
    valid_indices = [index for index in blocked_indices if index >= 0]
    if grasp_count is not None:
        valid_indices = [index for index in valid_indices if index < grasp_count]
    return sorted(valid_indices)


def parse_optional_grasp_index(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def selected_grasp_index_for_action_sequence(args: SimpleNamespace) -> Optional[int]:
    settings = action_sequence_settings_for_args(args)
    configured_indices: list[tuple[str, int]] = []
    for key in ("specified_grasp_index", "selected_grasp_index", "selected_grasp", "grasp_index"):
        if key not in settings:
            continue
        index = parse_optional_grasp_index(settings[key])
        if index is not None:
            configured_indices.append((key, index))
    if not configured_indices:
        return None
    primary_key, primary_index = configured_indices[0]
    for key, index in configured_indices[1:]:
        if index != primary_index:
            raise ValueError(
                f"Action sequence {current_action_sequence_key(args)} has conflicting grasp fields: "
                f"{primary_key}={primary_index}, {key}={index}."
            )
    return primary_index


def pick_approach_jaw_width_override_for_action_sequence(args: Optional[SimpleNamespace]) -> Optional[float]:
    if args is None:
        return None
    settings = action_sequence_settings_for_args(args)
    for key in ("pick_approach_jaw_width", "pick_approach_jawwidth", "approach_jaw_width"):
        if key not in settings:
            continue
        value = settings[key]
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        jaw_width = float(value)
        if jaw_width < 0:
            raise ValueError(
                f"Action sequence {current_action_sequence_key(args)} has negative {key}: {jaw_width}."
            )
        return jaw_width
    return None


def pick_approach_distance_for_action_sequence(args: Optional[SimpleNamespace]) -> Optional[float]:
    if args is None:
        return None
    settings = action_sequence_settings_for_args(args)
    for key in ("pick_approach_distance", "approach_distance"):
        if key not in settings:
            continue
        value = settings[key]
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        distance = float(value)
        if distance < 0:
            raise ValueError(
                f"Action sequence {current_action_sequence_key(args)} has negative {key}: {distance}."
            )
        return distance
    return None


def filter_candidate_grasps_for_action_sequence(
    args: SimpleNamespace,
    candidate_indices: list[int],
    grasp_count: int,
) -> list[int]:
    action_sequence = getattr(args, "action_sequence", None)
    if action_sequence is None:
        return candidate_indices

    blocked_indices = blocked_grasp_indices_for_action_sequence(args, grasp_count)
    filtered_indices = list(candidate_indices)
    if blocked_indices:
        blocked_set = set(blocked_indices)
        filtered_indices = [index for index in candidate_indices if index not in blocked_set]
        removed_count = len(candidate_indices) - len(filtered_indices)
        preview = blocked_indices[:24]
        suffix = "..." if len(blocked_indices) > len(preview) else ""
        sim_pick.debug_print(
            f"  action sequence {action_sequence}: blocked grasp indices {preview}{suffix}; "
            f"removed {removed_count}/{len(candidate_indices)} candidate(s)"
        )
    else:
        sim_pick.debug_print(f"  action sequence {action_sequence}: no blocked grasp indices")

    selected_index = selected_grasp_index_for_action_sequence(args)
    if selected_index is not None:
        if selected_index < 0 or selected_index >= grasp_count:
            raise RuntimeError(
                f"Pick planner failed: action sequence {action_sequence} specified grasp {selected_index}, "
                f"but valid grasp indices are 0..{grasp_count - 1}."
            )
        if selected_index not in candidate_indices:
            raise RuntimeError(
                f"Pick planner failed: action sequence {action_sequence} specified grasp {selected_index}, "
                "but it is not pick-feasible after planner reasoning."
            )
        if selected_index not in filtered_indices:
            raise RuntimeError(
                f"Pick planner failed: action sequence {action_sequence} specified grasp {selected_index}, "
                "but that grasp is also blocked by the same action sequence config."
            )
        sim_pick.debug_print(f"  action sequence {action_sequence}: specified grasp index {selected_index} (grasps[{selected_index}])")
        return [selected_index]

    if filtered_indices:
        candidate_preview = filtered_indices[:20]
        candidate_suffix = "..." if len(filtered_indices) > len(candidate_preview) else ""
        sim_pick.debug_print(f"  candidate grasp indices after action mask: {candidate_preview}{candidate_suffix}")
        return filtered_indices
    raise RuntimeError(
        f"Pick planner failed: action sequence {action_sequence} blocked all pick-feasible grasps."
    )


def xy_angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    vector_a = np.asarray(vector_a, dtype=float).reshape(2)
    vector_b = np.asarray(vector_b, dtype=float).reshape(2)
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return float("inf")
    cosine = float(np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def robot_arm_base_pos(robot) -> np.ndarray:
    arm = getattr(robot, "arm", getattr(robot, "manipulator", None))
    if arm is None:
        return np.zeros(3)
    return np.asarray(getattr(arm, "pos", np.zeros(3)), dtype=float).reshape(3)


def first_joint_range(robot) -> tuple[float, float]:
    ranges = getattr(robot, "jnt_ranges", None)
    ranges = ranges() if callable(ranges) else ranges
    if ranges is None:
        arm = getattr(robot, "arm", getattr(robot, "manipulator", None))
        ranges = getattr(arm, "jnt_ranges", None) if arm is not None else None
    ranges = np.asarray(ranges, dtype=float)
    if ranges.ndim != 2 or ranges.shape[0] < 1 or ranges.shape[1] < 2:
        raise RuntimeError("Could not read first joint range for post-pick alignment.")
    lower, upper = float(ranges[0, 0]), float(ranges[0, 1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise RuntimeError(f"Invalid first joint range for post-pick alignment: {ranges[0].tolist()}")
    return lower, upper


def tcp_base_xy_angle_for_conf(robot, conf: np.ndarray, base_pos: np.ndarray, target_xy: np.ndarray) -> float:
    tcp_pos, _tcp_rotmat = robot.fk(jnt_values=np.asarray(conf, dtype=float))
    tcp_vector_xy = (np.asarray(tcp_pos, dtype=float).reshape(3) - base_pos)[:2]
    return xy_angle_deg(tcp_vector_xy, target_xy)


def make_world_z_lowering_path(robot, start_conf: np.ndarray) -> tuple[list[np.ndarray], float]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    lowering_distance = float(sim_pick.PICK_APPROACH_DEPART_DISTANCE) * 0.75
    step_count = max(1, int(np.ceil(lowering_distance / max(float(sim_pick.LINEAR_GRANULARITY), 1e-4))))
    start_tcp_pos, start_tcp_rotmat = robot.fk(jnt_values=start_conf)
    start_tcp_pos = np.asarray(start_tcp_pos, dtype=float).reshape(3)
    start_tcp_rotmat = np.asarray(start_tcp_rotmat, dtype=float).reshape(3, 3)
    lowering_path: list[np.ndarray] = []
    seed_conf = start_conf
    for step_id in range(1, step_count + 1):
        ratio = float(step_id) / float(step_count)
        target_pos = start_tcp_pos - sim_pick.rm.const.z_ax * lowering_distance * ratio
        jnt_values = robot.ik(
            tgt_pos=target_pos,
            tgt_rotmat=start_tcp_rotmat,
            seed_jnt_values=seed_conf,
        )
        if jnt_values is None:
            return [], lowering_distance
        jnt_values = np.asarray(jnt_values, dtype=float).reshape(-1)
        lowering_path.append(jnt_values)
        seed_conf = jnt_values
    return lowering_path, lowering_distance


def make_world_linear_tcp_path(
    robot,
    start_conf: np.ndarray,
    direction: np.ndarray,
    distance: float,
    linear_granularity: Optional[float] = None,
) -> tuple[list[np.ndarray], float]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-9:
        raise ValueError("World linear TCP direction cannot be zero.")
    direction = direction / direction_norm
    distance = float(distance)
    if distance <= 0:
        raise ValueError(f"World linear TCP distance must be positive, got {distance}.")
    granularity = float(sim_pick.LINEAR_GRANULARITY if linear_granularity is None else linear_granularity)
    step_count = max(1, int(np.ceil(distance / max(granularity, 1e-4))))
    start_tcp_pos, start_tcp_rotmat = robot.fk(jnt_values=start_conf)
    start_tcp_pos = np.asarray(start_tcp_pos, dtype=float).reshape(3)
    start_tcp_rotmat = np.asarray(start_tcp_rotmat, dtype=float).reshape(3, 3)
    path: list[np.ndarray] = []
    seed_conf = start_conf
    for step_id in range(1, step_count + 1):
        ratio = float(step_id) / float(step_count)
        target_pos = start_tcp_pos + direction * distance * ratio
        jnt_values = robot.ik(
            tgt_pos=target_pos,
            tgt_rotmat=start_tcp_rotmat,
            seed_jnt_values=seed_conf,
        )
        if jnt_values is None:
            return [], distance
        jnt_values = np.asarray(jnt_values, dtype=float).reshape(-1)
        path.append(jnt_values)
        seed_conf = jnt_values
    return path, distance


def make_world_z_lift_path(robot, start_conf: np.ndarray, distance: float) -> tuple[list[np.ndarray], float]:
    return make_world_linear_tcp_path(
        robot,
        start_conf=start_conf,
        direction=sim_pick.rm.const.z_ax,
        distance=distance,
    )


def make_world_tcp_position_path(
    robot,
    start_conf: np.ndarray,
    target_pos: np.ndarray,
    linear_granularity: Optional[float] = None,
) -> tuple[list[np.ndarray], float]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    start_tcp_pos, _start_tcp_rotmat = robot.fk(jnt_values=start_conf)
    start_tcp_pos = np.asarray(start_tcp_pos, dtype=float).reshape(3)
    delta = target_pos - start_tcp_pos
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        return [], 0.0
    return make_world_linear_tcp_path(
        robot,
        start_conf=start_conf,
        direction=delta,
        distance=distance,
        linear_granularity=linear_granularity,
    )


def make_joint_interpolation_path(
    start_conf: np.ndarray,
    goal_conf: np.ndarray,
    path_step_deg: float = 2.0,
) -> list[np.ndarray]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    goal_conf = np.asarray(goal_conf, dtype=float).reshape(-1)
    if start_conf.shape != goal_conf.shape:
        raise ValueError(f"Joint interpolation shape mismatch: {start_conf.shape} vs {goal_conf.shape}.")
    max_delta = float(np.max(np.abs(goal_conf - start_conf))) if start_conf.size else 0.0
    if max_delta < 1e-9:
        return []
    step = max(float(np.radians(path_step_deg)), 1e-4)
    step_count = max(1, int(np.ceil(max_delta / step)))
    return [np.asarray(conf, dtype=float).reshape(-1) for conf in np.linspace(start_conf, goal_conf, step_count + 1)[1:]]


def make_j2_retreat_path(
    start_conf: np.ndarray,
    target_deg: Optional[float] = None,
    delta_deg: Optional[float] = None,
    path_step_deg: float = 2.0,
) -> tuple[list[np.ndarray], float]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    if start_conf.size <= PUSH_J2_JOINT_INDEX:
        raise ValueError(f"Joint path has no J2 at index {PUSH_J2_JOINT_INDEX}.")
    goal_conf = start_conf.copy()
    if delta_deg is not None:
        goal_conf[PUSH_J2_JOINT_INDEX] = start_conf[PUSH_J2_JOINT_INDEX] + np.radians(float(delta_deg))
    else:
        target_rad = np.radians(DEFAULT_PUSH_J2_RETREAT_TARGET_DEG if target_deg is None else float(target_deg))
        goal_conf[PUSH_J2_JOINT_INDEX] = min(start_conf[PUSH_J2_JOINT_INDEX], target_rad)
    actual_delta_deg = float(np.degrees(goal_conf[PUSH_J2_JOINT_INDEX] - start_conf[PUSH_J2_JOINT_INDEX]))
    return make_joint_interpolation_path(start_conf, goal_conf, path_step_deg=path_step_deg), actual_delta_deg

def make_first_joint_alignment_path(
    robot,
    start_conf: np.ndarray,
    ee_value,
    obstacle_list: list[object],
    max_angle_deg: float = 5.0,
    sample_step_deg: float = 1.0,
    path_step_deg: float = 2.0,
) -> tuple[list[np.ndarray], float]:
    start_conf = np.asarray(start_conf, dtype=float).reshape(-1)
    base_pos = robot_arm_base_pos(robot)
    target_xy = (np.asarray(BOTTLE_ROBOT_SIDE_PLACE_POS, dtype=float).reshape(3) - base_pos)[:2]
    if np.linalg.norm(target_xy) < 1e-9:
        raise RuntimeError("robot-base-to-BOTTLE_ROBOT_SIDE_PLACE_POS xy vector is too small for post-pick alignment.")
    lower, upper = first_joint_range(robot)
    sample_step = max(float(np.radians(sample_step_deg)), 1e-4)
    sample_count = int(np.ceil((upper - lower) / sample_step)) + 1
    sample_count = max(2, min(sample_count, 2000))
    sampled_q0 = np.linspace(lower, upper, sample_count)
    sampled_q0 = np.unique(np.concatenate(([start_conf[0]], sampled_q0)))

    candidates: list[tuple[float, float, float]] = []
    robot.backup_state()
    try:
        for q0 in sampled_q0:
            counterclockwise_delta = float(q0) - float(start_conf[0])
            if counterclockwise_delta < -1e-9:
                continue
            conf = start_conf.copy()
            conf[0] = float(q0)
            angle_deg = tcp_base_xy_angle_for_conf(robot, conf, base_pos, target_xy)
            if angle_deg <= max_angle_deg:
                candidates.append((counterclockwise_delta, angle_deg, float(q0)))
    finally:
        robot.restore_state()

    if not candidates:
        raise RuntimeError(
            f"No counterclockwise first-joint angle makes TCP-base XY direction within {max_angle_deg:.1f} deg "
            f"of robot-base-to-BOTTLE_ROBOT_SIDE_PLACE_POS xy={target_xy.tolist()}."
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    path_step = max(float(np.radians(path_step_deg)), 1e-4)
    for _delta, angle_deg, goal_q0 in candidates:
        delta = float(goal_q0) - float(start_conf[0])
        if abs(delta) < 1e-6:
            return [], angle_deg
        step_count = max(1, int(np.ceil(abs(delta) / path_step)))
        q0_values = np.linspace(float(start_conf[0]), goal_q0, step_count + 1)[1:]
        path = []
        for q0 in q0_values:
            conf = start_conf.copy()
            conf[0] = float(q0)
            path.append(conf)
        return path, angle_deg


def push_clearance_text(push_settings: dict[str, Any]) -> str:
    clearance_mode = str(push_settings.get("clearance_mode", "joint"))
    if clearance_mode == "joint":
        values = push_settings.get("clearance_jnt_values")
        if values is None:
            values_deg = push_settings.get("clearance_jnt_values_deg")
            values = None if values_deg is None else np.radians(np.asarray(values_deg, dtype=float))
        return f"clearance_jnt_values_deg={sim_pick.format_jnts_deg(values, digits=3)}"
    if clearance_mode == "tcp":
        return f"clearance_tcp_pos={sim_pick.format_vec(push_settings['clearance_tcp_pos'], digits=4)}"
    if clearance_mode == "j2_retreat":
        delta_deg = push_settings.get("j2_retreat_delta_deg")
        if delta_deg is not None:
            return f"j2_retreat_delta_deg={float(delta_deg):.3f}"
        target_deg = push_settings.get("j2_retreat_target_deg", DEFAULT_PUSH_J2_RETREAT_TARGET_DEG)
        return f"j2_retreat_target_deg={float(target_deg):.3f}"
    return f"clearance_mode={clearance_mode}"


def serializable_push_settings(push_settings: dict[str, Any]) -> dict[str, Any]:
    clearance_jnt_values_deg = push_settings.get("clearance_jnt_values_deg")
    clearance_tcp_pos = push_settings.get("clearance_tcp_pos")
    j2_retreat_target_deg = push_settings.get("j2_retreat_target_deg")
    j2_retreat_delta_deg = push_settings.get("j2_retreat_delta_deg")
    return {
        "direction": np.asarray(push_settings["direction"], dtype=float).reshape(3).tolist(),
        "distance": float(push_settings["distance"]),
        "lift_distance": float(push_settings["lift_distance"]),
        "clearance_mode": str(push_settings.get("clearance_mode", "joint")),
        "clearance_jnt_values_deg": None
        if clearance_jnt_values_deg is None
        else np.asarray(clearance_jnt_values_deg, dtype=float).reshape(-1).tolist(),
        "clearance_tcp_pos": None
        if clearance_tcp_pos is None
        else np.asarray(clearance_tcp_pos, dtype=float).reshape(3).tolist(),
        "j2_retreat_joint_index": int(push_settings.get("j2_retreat_joint_index", PUSH_J2_JOINT_INDEX)),
        "j2_retreat_target_deg": None if j2_retreat_target_deg is None else float(j2_retreat_target_deg),
        "j2_retreat_delta_deg": None if j2_retreat_delta_deg is None else float(j2_retreat_delta_deg),
        "open_after_push": True,
    }


def make_fresh_pick_object_model(object_model_path: Path, pick_pose: tuple[np.ndarray, np.ndarray], name: str):
    return sim_pick.make_object_model(object_model_path, pick_pose, name=name)


def gen_pick_push_path_for_grasp(
    planner,
    robot,
    object_model_path: Path,
    grasp,
    pick_pose: tuple[np.ndarray, np.ndarray],
    pick_approach_jaw_width: float,
    pick_approach_distance: Optional[float],
    push_settings: dict[str, Any],
    obstacle_list: list[object],
):
    approach_obj_cmodel = make_fresh_pick_object_model(
        object_model_path,
        pick_pose,
        name="icp_bottle_pick_push_approach",
    )
    pick_tcp_pos, pick_tcp_rotmat = sim_pick.tcp_pose_from_object_pose(pick_pose, grasp)
    grasp_jaw_width = sim_pick.clamp_jaw_width(robot, grasp.ee_values)
    mot_data = planner.gen_approach(
        goal_tcp_pos=pick_tcp_pos,
        goal_tcp_rotmat=pick_tcp_rotmat,
        start_jnt_values=sim_pick.DEFAULT_HOME_CONF,
        linear_direction=None,
        linear_distance=float(sim_pick.APPROACH_DISTANCE if pick_approach_distance is None else pick_approach_distance),
        linear_granularity=sim_pick.LINEAR_GRANULARITY,
        ee_values=grasp_jaw_width,
        obstacle_list=obstacle_list,
        object_list=[approach_obj_cmodel],
        use_rrt=sim_pick.USE_RRT,
        toggle_dbg=False,
    )
    if mot_data is None:
        return None, "pick approach failed"

    obj_cmodel_copy = make_fresh_pick_object_model(
        object_model_path,
        pick_pose,
        name="icp_bottle_pick_push_held",
    )
    for robot_mesh in mot_data.mesh_list:
        if robot_mesh is not None:
            obj_cmodel_copy.attach_to(robot_mesh)

    close_index = 0
    approach_end_index = len(mot_data.jv_list) - 1
    push_start_index = approach_end_index
    push_end_index = approach_end_index
    forced_open_index = None
    lift_end_index = approach_end_index
    clearance_mode = str(push_settings.get("clearance_mode", "joint"))
    clearance_distance = 0.0

    robot.backup_state()
    try:
        robot.goto_given_conf(mot_data.jv_list[-1], ee_values=grasp_jaw_width)
        robot.hold(obj_cmodel=obj_cmodel_copy, jaw_width=grasp_jaw_width)

        push_path, push_distance = make_world_linear_tcp_path(
            robot,
            np.asarray(mot_data.jv_list[-1], dtype=float),
            direction=push_settings["direction"],
            distance=push_settings["distance"],
        )
        if not push_path:
            return None, "world push linear IK failed"
        mot_data.extend(
            push_path,
            ev_list=[grasp_jaw_width] * len(push_path),
            mesh_list=[],
        )
        push_end_index = len(mot_data.jv_list) - 1

        active_jaw_width = sim_pick.clamp_jaw_width(robot, sim_pick.OPEN_JAW_WIDTH)
        mot_data.extend(
            [np.asarray(mot_data.jv_list[-1], dtype=float).copy()],
            ev_list=[active_jaw_width],
            mesh_list=[],
        )
        forced_open_index = len(mot_data.jv_list) - 1
        setattr(mot_data, "force_final_open_gripper", True)
        setattr(mot_data, "forced_open_index", forced_open_index)

        lift_path, lift_distance = make_world_z_lift_path(
            robot,
            np.asarray(mot_data.jv_list[-1], dtype=float),
            distance=push_settings["lift_distance"],
        )
        if not lift_path:
            return None, "world +Z lift after push IK failed"
        mot_data.extend(
            lift_path,
            ev_list=[active_jaw_width] * len(lift_path),
            mesh_list=[],
        )
        lift_end_index = len(mot_data.jv_list) - 1

        if clearance_mode == "joint":
            clearance_path = make_joint_interpolation_path(
                np.asarray(mot_data.jv_list[-1], dtype=float),
                np.asarray(push_settings["clearance_jnt_values"], dtype=float),
            )
        elif clearance_mode == "tcp":
            clearance_path, clearance_distance = make_world_tcp_position_path(
                robot,
                np.asarray(mot_data.jv_list[-1], dtype=float),
                np.asarray(push_settings["clearance_tcp_pos"], dtype=float),
            )
        elif clearance_mode == "j2_retreat":
            clearance_path, clearance_distance = make_j2_retreat_path(
                np.asarray(mot_data.jv_list[-1], dtype=float),
                target_deg=push_settings.get("j2_retreat_target_deg"),
                delta_deg=push_settings.get("j2_retreat_delta_deg"),
            )
        else:
            raise ValueError(f"Unknown push clearance mode: {clearance_mode!r}")
        if clearance_path:
            mot_data.extend(
                clearance_path,
                ev_list=[active_jaw_width] * len(clearance_path),
                mesh_list=[],
            )
    finally:
        robot.restore_state()

    setattr(mot_data, "path_type", "push")
    setattr(mot_data, "push_settings", serializable_push_settings(push_settings))
    setattr(mot_data, "close_index", close_index)
    setattr(mot_data, "approach_end_index", approach_end_index)
    setattr(mot_data, "push_start_index", push_start_index)
    setattr(mot_data, "push_end_index", push_end_index)
    setattr(mot_data, "push_lift_end_index", lift_end_index)
    setattr(mot_data, "camera_clearance_end_index", len(mot_data.jv_list) - 1)
    setattr(mot_data, "camera_clearance_mode", clearance_mode)
    setattr(mot_data, "camera_clearance_distance", clearance_distance)
    return mot_data, None

def gen_pick_only_path(robot, object_model_path: Path, grasps, pick_pose, obstacle_list, args: SimpleNamespace) -> tuple[Optional[int], Any]:
    from wrs import ppp

    sim_pick.debug_print("Starting pick-only planner...")
    sim_pick.debug_print(f"  grasps={len(grasps)}, use_rrt={sim_pick.USE_RRT}")
    sim_pick.debug_print(f"  reason_pick_grasps={sim_pick.USE_REASONED_COMMON_GRASPS}")
    pick_approach_distance_override = pick_approach_distance_for_action_sequence(args)
    pick_approach_distance = float(sim_pick.APPROACH_DISTANCE if pick_approach_distance_override is None else pick_approach_distance_override)
    pick_approach_distance_source = "default" if pick_approach_distance_override is None else "action sequence JSON"
    sim_pick.debug_print(
        f"  pick approach distance={pick_approach_distance:.3f} m ({pick_approach_distance_source}), "
        f"depart distance={sim_pick.PICK_APPROACH_DEPART_DISTANCE:.3f} m, "
        f"linear_granularity={sim_pick.LINEAR_GRANULARITY:.3f} m"
    )
    sim_pick.debug_print("  directions: pick approach=grasp TCP +Z, pick depart=+Z")

    planner = ppp.PickPlacePlanner(robot)
    candidate_indices = list(range(len(grasps)))
    if sim_pick.USE_REASONED_COMMON_GRASPS:
        grasp_collection = sim_pick.make_grasp_collection(robot, grasps)
        candidate_indices = list(
            planner.reason_common_gids(
                grasp_collection=grasp_collection,
                goal_pose_list=[pick_pose],
                obstacle_list=obstacle_list,
                toggle_dbg=False,
            )
        )
        sim_pick.debug_print(f"  reasoned pick grasps: {len(candidate_indices)}/{len(grasps)}")
        if candidate_indices:
            preview = candidate_indices[:20]
            suffix = "..." if len(candidate_indices) > len(preview) else ""
            sim_pick.debug_print(f"  pick grasp indices: {preview}{suffix}")
        else:
            raise RuntimeError("Pick planner failed: no pick-feasible grasps.")

    candidate_indices = filter_candidate_grasps_for_action_sequence(args, candidate_indices, len(grasps))

    pick_approach_jaw_width_override = pick_approach_jaw_width_override_for_action_sequence(args)
    pick_approach_jaw_width_source = "default 1.2x grasp jaw" if pick_approach_jaw_width_override is None else "action sequence JSON"
    push_settings = action_sequence_push_settings(args)
    if push_settings is not None:
        clearance_text = push_clearance_text(push_settings)
        sim_pick.debug_print(
            "  push path enabled by action sequence JSON: "
            f"direction={sim_pick.format_vec(push_settings['direction'], digits=4)}, "
            f"distance={push_settings['distance']:.4f} m, "
            f"lift_distance={push_settings['lift_distance']:.4f} m, "
            f"{clearance_text}"
        )

    for grasp_index in candidate_indices:
        grasp = grasps[grasp_index]
        grasp_width = grasp_jaw_width(grasp)
        if push_settings is None:
            pick_approach_jaw_width = pick_approach_jaw_width_for_grasp(robot, grasp, pick_approach_jaw_width_override)
            jaw_text = f"pick approach jaw={pick_approach_jaw_width:.5f} m ({pick_approach_jaw_width_source})"
        else:
            pick_approach_jaw_width = sim_pick.clamp_jaw_width(robot, grasp_width)
            jaw_text = f"push pre-approach close jaw={pick_approach_jaw_width:.5f} m (grasp jaw)"
        robot.goto_given_conf(sim_pick.DEFAULT_HOME_CONF, ee_values=pick_approach_jaw_width)
        sim_pick.debug_print(
            f"  pickle_grasp_{grasp_index}: grasp jaw={grasp_width:.5f} m, "
            f"{jaw_text}"
        )
        if push_settings is not None:
            mot_data, push_error = gen_pick_push_path_for_grasp(
                planner=planner,
                robot=robot,
                object_model_path=object_model_path,
                grasp=grasp,
                pick_pose=pick_pose,
                pick_approach_jaw_width=pick_approach_jaw_width,
                pick_approach_distance=pick_approach_distance,
                push_settings=push_settings,
                obstacle_list=obstacle_list,
            )
            if mot_data is None:
                sim_pick.debug_print(f"  pickle_grasp_{grasp_index}: push path failed: {push_error}")
                continue
            frame_count = len(mot_data.jv_list) if hasattr(mot_data, "jv_list") else len(mot_data)
            approach_end_index = int(getattr(mot_data, "approach_end_index", 0))
            push_start_index = int(getattr(mot_data, "push_start_index", approach_end_index))
            push_end_index = int(getattr(mot_data, "push_end_index", push_start_index))
            forced_open_index = int(getattr(mot_data, "forced_open_index", push_end_index))
            push_lift_end_index = int(getattr(mot_data, "push_lift_end_index", forced_open_index))
            clearance_end_index = int(getattr(mot_data, "camera_clearance_end_index", frame_count - 1))
            sim_pick.debug_print(
                f"Pick push path produced {frame_count} frame(s): "
                f"close_gripper_to before approach, approach_end={approach_end_index}, "
                f"push={max(0, push_end_index - push_start_index)} frame(s), "
                f"forced open at frame {forced_open_index}, "
                f"world +Z lift={max(0, push_lift_end_index - forced_open_index)} frame(s), "
                f"J2/camera-clearance move={max(0, clearance_end_index - push_lift_end_index)} frame(s)."
            )
            sim_pick.debug_print("  Push path uses no collision check for push, world +Z lift, or camera-clearance move.")
            sim_pick.debug_print("Pick planner succeeded.")
            sim_pick.debug_print(f"  selected grasp: pickle_grasp_{grasp_index}")
            return grasp_index, mot_data

        mot_data = planner.gen_pick_and_moveto(
            obj_cmodel=make_fresh_pick_object_model(
                object_model_path,
                pick_pose,
                name=f"icp_bottle_pick_place_grasp_{grasp_index}",
            ),
            grasp=grasp,
            moveto_pose_list=[],
            moveto_approach_direction_list=[],
            moveto_approach_distance_list=[],
            moveto_depart_direction_list=[],
            moveto_depart_distance_list=[],
            start_jnt_values=sim_pick.DEFAULT_HOME_CONF,
            pick_approach_jaw_width=pick_approach_jaw_width,
            pick_approach_direction=None,
            pick_approach_distance=pick_approach_distance,
            pick_depart_direction=sim_pick.rm.const.z_ax,
            pick_depart_distance=sim_pick.PICK_APPROACH_DEPART_DISTANCE,
            linear_granularity=sim_pick.LINEAR_GRANULARITY,
            obstacle_list=obstacle_list,
            use_rrt=sim_pick.USE_RRT,
            toggle_dbg=False,
        )
        if mot_data is None:
            sim_pick.debug_print(f"  pickle_grasp_{grasp_index}: pick approach/depart failed")
            continue
        pick_frame_count = len(mot_data.jv_list) if hasattr(mot_data, "jv_list") else len(mot_data)
        sim_pick.debug_print(f"Pick approach/depart produced {pick_frame_count} frame(s).")

        try:
            align_path, final_angle_deg = make_first_joint_alignment_path(
                robot,
                np.asarray(mot_data.jv_list[-1], dtype=float),
                mot_data.ev_list[-1],
                obstacle_list,
            )
        except Exception as exc:
            sim_pick.debug_print(f"  pickle_grasp_{grasp_index}: post-pick first-joint alignment failed: {exc}")
            continue
        if align_path:
            mot_data.extend(align_path, toggle_mesh=False)
            sim_pick.debug_print(
                f"  Post-pick first-joint alignment: {len(align_path)} frame(s), "
                f"final angle={final_angle_deg:.2f} deg"
            )
        else:
            sim_pick.debug_print(
                f"  Post-pick first-joint alignment already within threshold: "
                f"angle={final_angle_deg:.2f} deg"
            )

        pre_lowering_conf = np.asarray(mot_data.jv_list[-1], dtype=float).copy()
        lowering_path, lowering_distance = make_world_z_lowering_path(
            robot,
            pre_lowering_conf,
        )
        if not lowering_path:
            sim_pick.debug_print(
                f"  pickle_grasp_{grasp_index}: world -Z lowering before open failed; IK not solvable"
            )
            continue
        mot_data.extend(
            lowering_path,
            ev_list=[mot_data.ev_list[-1]] * len(lowering_path),
            mesh_list=[],
        )
        sim_pick.debug_print(
            f"  World -Z lowering before open without collision check: {len(lowering_path)} frame(s), "
            f"distance={lowering_distance:.4f} m"
        )

        final_open_width = sim_pick.clamp_jaw_width(robot, sim_pick.OPEN_JAW_WIDTH)
        mot_data.extend(
            [np.asarray(mot_data.jv_list[-1], dtype=float).copy()],
            ev_list=[final_open_width],
            mesh_list=[],
        )
        setattr(mot_data, "force_final_open_gripper", True)
        setattr(mot_data, "forced_open_index", len(mot_data.jv_list) - 1)
        sim_pick.debug_print(f"  Forced open gripper frame appended before lift: jaw={final_open_width:.5f} m")

        lift_path = [np.asarray(conf, dtype=float).copy() for conf in reversed(lowering_path[:-1])]
        lift_path.append(pre_lowering_conf)
        mot_data.extend(
            lift_path,
            ev_list=[final_open_width] * len(lift_path),
            mesh_list=[],
        )
        sim_pick.debug_print(
            f"  World +Z lift after open without collision check: {len(lift_path)} frame(s), "
            f"distance={lowering_distance:.4f} m; returned to pre-lowering pose"
        )

        frame_count = len(mot_data.jv_list) if hasattr(mot_data, "jv_list") else len(mot_data)
        sim_pick.debug_print(
            f"Pick planner produced {frame_count} frame(s), including post-pick first-joint alignment, "
            "world -Z lowering, forced open, and world +Z lift."
        )
        sim_pick.debug_print("Pick planner succeeded.")
        sim_pick.debug_print(f"  selected grasp: pickle_grasp_{grasp_index}")
        return grasp_index, mot_data

    if push_settings is not None:
        raise RuntimeError(
            "Pick push planner failed: no grasp produced a valid approach/close/push/lift/camera-clearance path."
        )
    raise RuntimeError("Pick planner failed: no grasp produced a valid pick approach/depart path.")


def print_pick_only_summary(mot_data, grasp_count: int, selected_grasp_index: int, pick_pose, obstacle_names: list[str]) -> None:
    sim_pick.debug_print("Pick-only planner path ready.")
    sim_pick.debug_print(f"Loaded grasps: {grasp_count}")
    if selected_grasp_index is None:
        sim_pick.debug_print("Selected grasp: unknown")
    else:
        sim_pick.debug_print(f"Selected grasp: pickle_grasp_{selected_grasp_index}")
    sim_pick.debug_print(f"Frames: {len(mot_data.jv_list)}")
    path_type = getattr(mot_data, "path_type", "pick_open")
    sim_pick.debug_print(f"Path type: {path_type}")
    push_settings = getattr(mot_data, "push_settings", None)
    if push_settings is not None:
        clearance_text = push_clearance_text(push_settings)
        sim_pick.debug_print(
            "Push settings: "
            f"direction={sim_pick.format_vec(push_settings['direction'], digits=4)}, "
            f"distance={push_settings['distance']:.4f} m, "
            f"lift_distance={push_settings['lift_distance']:.4f} m, "
            f"{clearance_text}"
        )
    if len(mot_data.jv_list) > 1:
        max_step = max(
            float(np.max(np.abs(np.asarray(mot_data.jv_list[i + 1]) - np.asarray(mot_data.jv_list[i]))))
            for i in range(len(mot_data.jv_list) - 1)
        )
        sim_pick.debug_print(f"Max adjacent joint step(deg): {np.degrees(max_step):.3f}")
    sim_pick.debug_print(f"Collision obstacles: {obstacle_names}")
    sim_pick.debug_print(f"Pick object pos(m): {sim_pick.format_vec(pick_pose[0], digits=6)}")
    sim_pick.debug_print(f"Start conf(deg): {sim_pick.format_jnts_deg(mot_data.jv_list[0], digits=3)}")
    sim_pick.debug_print(f"End conf(deg): {sim_pick.format_jnts_deg(mot_data.jv_list[-1], digits=3)}")

def run_or_skip_plan(args: SimpleNamespace, icp: ObjectIcpResult) -> Optional[PlanningResult]:
    if args.skip_plan:
        return None

    object_model_path = args.object_model if args.object_model is not None else icp.bottle_model_path
    object_model_path = object_model_path.resolve()
    if not object_model_path.exists():
        raise FileNotFoundError(f"Object model not found: {object_model_path}")

    bottle_homomat = bottle_homomat_for_icp(icp)
    box_homomat = load_homomat(icp.box_transform_path, "box")
    pick_pose = homomat_to_pose(bottle_homomat)
    place_pose, place_pos_source, place_rot_source = resolve_place_pose(args, pick_pose)

    configure_pick_place(args, object_model_path)
    if not sim_pick.GRASP_PICKLE_PATH.exists():
        raise FileNotFoundError(f"Grasp pickle not found: {sim_pick.GRASP_PICKLE_PATH}")
    action_sequence = getattr(args, "action_sequence", None)
    action_sequence = None if action_sequence is None else int(action_sequence)

    sim_pick.debug_print("=== real_bottle_pick_place_interactive: pick-only planner run ===")
    if action_sequence is None:
        sim_pick.debug_print("Action sequence: disabled")
    else:
        sim_pick.debug_print(f"Action sequence: {action_sequence}")
    sim_pick.debug_print(f"Object model: {object_model_path}")
    sim_pick.debug_print(f"Grasp pickle: {sim_pick.GRASP_PICKLE_PATH}")
    sim_pick.debug_print("Robot: UR7EDH76(enable_cc=True, ik_solver='ikfast')")
    sim_pick.debug_print(f"Pick pose from ICP: pos={sim_pick.format_vec(pick_pose[0], digits=6)}")
    sim_pick.debug_print(
        f"Place pose preview only: pos={sim_pick.format_vec(place_pose[0], digits=6)} "
        f"({place_pos_source}), rot={place_rot_source}; not used by pick-only planner"
    )

    robot = sim_pick.make_robot()
    sim_pick.debug_ikfast_frame_conversion(robot)
    grasps = sim_pick.load_grasps(robot, sim_pick.GRASP_PICKLE_PATH)
    sim_pick.debug_print(f"Loaded grasps: {len(grasps)}")

    obstacle_list, display_obstacle_list = build_obstacle_lists(args, box_homomat, include_display=True)
    obstacle_names = [getattr(obstacle, "name", type(obstacle).__name__) for obstacle in obstacle_list]
    display_obstacle_names = [getattr(obstacle, "name", type(obstacle).__name__) for obstacle in display_obstacle_list]
    sim_pick.debug_print(f"Collision obstacles: {obstacle_names}")
    sim_pick.debug_print(f"Visualization context: {display_obstacle_names}")
    precheck_start_collision(robot, obstacle_list)

    selected_grasp_index, mot_data = gen_pick_only_path(
        robot,
        object_model_path,
        grasps,
        pick_pose,
        obstacle_list,
        args,
    )
    if selected_grasp_index is None:
        selected_grasp_index = infer_selected_grasp_index(robot, grasps, mot_data, pick_pose)
    print_pick_only_summary(mot_data, len(grasps), selected_grasp_index, pick_pose, obstacle_names)

    planning = PlanningResult(
        selected_grasp_index=selected_grasp_index,
        action_sequence=action_sequence,
        mot_data=mot_data,
        pick_pose=pick_pose,
        place_pose=place_pose,
        object_model_path=object_model_path,
        obstacle_names=obstacle_names,
        place_pos_source=place_pos_source,
        place_rot_source=place_rot_source,
    )
    planning.rtde_plan, planning.rtde_plan_path = build_and_save_rtde_plan(args, planning, icp.output_dir)

    if sim_pick.VISUALIZE_RESULT:
        sim_pick.visualize_result(mot_data, pick_pose, place_pose, display_obstacle_list)

    return planning


def write_summary(args: SimpleNamespace, icp: ObjectIcpResult, planning: Optional[PlanningResult]) -> Path:
    summary_path = args.summary_out if args.summary_out is not None else icp.output_dir / "real_bottle_pick_place_summary.json"
    summary = {
        "object_summary_path": str(icp.summary_path),
        "bottle_transform_path": None if icp.bottle_transform_path is None else str(icp.bottle_transform_path),
        "box_transform_path": str(icp.box_transform_path),
        "bottle_model_path": str(icp.bottle_model_path),
        "global_registered_path": None if icp.global_registered_path is None else str(icp.global_registered_path),
        "bottle_homomat": None if icp.bottle_homomat is None else np.asarray(icp.bottle_homomat, dtype=float).tolist(),
        "box_collision_obstacle": bool(args.box_obstacle),
        "table_obstacle": not bool(args.no_env),
        "ik_solver": "ikfast",
        "planned": planning is not None,
    }
    if planning is not None:
        summary.update(
            {
                "object_model_path": str(planning.object_model_path),
                "selected_grasp_index": planning.selected_grasp_index,
                "grasp_pickle_path": str(sim_pick.GRASP_PICKLE_PATH),
                "action_sequence": planning.action_sequence,
                "path_type": getattr(planning.mot_data, "path_type", "pick_open"),
                "push_settings": getattr(planning.mot_data, "push_settings", None),
                "blocked_grasp_indices": blocked_grasp_indices_for_action_sequence(args),
                "frame_count": len(planning.mot_data.jv_list),
                "collision_obstacle_names": planning.obstacle_names,
                "pick_pos": np.asarray(planning.pick_pose[0], dtype=float).tolist(),
                "pick_rotmat": np.asarray(planning.pick_pose[1], dtype=float).tolist(),
                "place_pos": np.asarray(planning.place_pose[0], dtype=float).tolist(),
                "place_rotmat": np.asarray(planning.place_pose[1], dtype=float).tolist(),
                "place_pos_source": planning.place_pos_source,
                "place_rot_source": planning.place_rot_source,
                "rtde_plan_path": None if planning.rtde_plan_path is None else str(planning.rtde_plan_path),
                "rtde_segment_count": None if planning.rtde_plan is None else len(planning.rtde_plan.segments),
                "use_move_l_compliant": bool(args.use_move_l_compliant),
                "visualized_result": not bool(args.dry_run),
            }
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def cuda_device_from_ultralytics_device(device: Any) -> Optional[str]:
    if device is None:
        return None
    raw = str(device).strip().lower()
    if raw in {"", "none", "cpu"}:
        return None
    if raw == "cuda":
        return "cuda:0"
    if raw.startswith("cuda:"):
        return raw
    if raw.isdigit():
        return f"cuda:{raw}"
    return raw


def preload_point_hint_model(args: SimpleNamespace) -> None:
    from yanjiuyuan import point_hint_segment as phs

    args.model = args.model or phs.default_model(args.backend)
    model_key = (args.backend, str(args.model))
    if getattr(args, "point_hint_model", None) is not None and getattr(args, "point_hint_model_key", None) == model_key:
        return

    print(
        f"[real_pipeline] Loading point_hint model at startup: "
        f"backend={args.backend}, model={args.model}, device={args.device}"
    )
    model = phs.load_model(args.backend, args.model)
    cuda_device = cuda_device_from_ultralytics_device(args.device)
    if cuda_device is not None:
        to_device = getattr(model, "to", None)
        if not callable(to_device):
            raise RuntimeError(f"Loaded point_hint model does not support moving to {cuda_device}.")
        to_device(cuda_device)
        print(f"[real_pipeline] point_hint model moved to {cuda_device}")
    args.point_hint_model = model
    args.point_hint_model_key = model_key


def preload_yolo_model(args: SimpleNamespace) -> None:
    """启动时预加载 YOLO 瓶子检测模型，避免首次检测时延迟。"""
    if getattr(args, "no_yolo", False):
        return
    from yanjiuyuan.yolo_detect import BottleDetector

    yolo_model_key = str(args.yolo_model)
    if getattr(args, "_yolo_detector", None) is not None and getattr(args, "_yolo_model_key", None) == yolo_model_key:
        return
    print(f"[real_pipeline] Loading YOLO model at startup: {args.yolo_model}")
    yolo_detector = BottleDetector(
        model_path=str(args.yolo_model),
        conf_threshold=args.yolo_conf,
        iou_threshold=args.yolo_iou,
    )
    args._yolo_detector = yolo_detector
    args._yolo_model_key = yolo_model_key
    print("[real_pipeline] YOLO model loaded.")

class InteractiveBottlePickPlaceApp:
    def __init__(
        self,
        args: SimpleNamespace,
        ctx: Optional[box_object_icp.PipelineContext] = None,
        initial_icp: Optional[ObjectIcpResult] = None,
    ):
        import wrs.modeling.geometric_model as mgm
        import wrs.visualization.panda.world as wd
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode, Notify

        # Suppress Panda3D "Ignoring recursive poll() within another task." spam.
        # This warning fires repeatedly when a key callback (C/D/P/O) runs a long
        # synchronous operation (planning, capture, etc.) inside the task loop.
        # In Panda3D 1.10.x, severity 5 = error level (filters out warnings).
        _task_cat = Notify.ptr().getCategory("task")
        if _task_cat is not None:
            _task_cat.setSeverity(5)

        self.args = args
        self.ctx = ctx
        self.icp_result = initial_icp
        self.planning_result: Optional[PlanningResult] = None
        self.environment_stale = False
        self.running = False
        self.detect_attempt_count = 0
        self.static_scene_models: list[object] = []
        self.scene_obstacle_models: list[object] = []
        self.robot_sync_models: list[object] = []
        self.detection_models: list[object] = []
        self.plan_models: list[object] = []
        self.animation_data = None
        self.animation_task_name = "real_bottle_pick_place_interactive_animation"
        self.mgm = mgm

        scene_points = self.compute_scene_points()
        cam_pos, lookat_pos, extent = box_object_icp.compute_camera_from_points(scene_points)
        self.base = wd.World(cam_pos=cam_pos, lookat_pos=lookat_pos, w=1280, h=720)

        frame_length = max(extent * 0.25, 0.03)
        frame_radius = max(frame_length * 0.015, 0.0005)
        mgm.gen_frame(ax_length=frame_length, ax_radius=frame_radius).attach_to(self.base)
        self.attach_static_pointclouds()
        self.attach_scene_obstacles()

        initial_text = "Step 1/4: Press C to sync robot, capture point cloud, and detect the box."
        self.status_text = OnscreenText(
            text=initial_text,
            pos=(-1.28, 0.92),
            align=TextNode.ALeft,
            scale=0.044,
            fg=(0.02, 0.02, 0.02, 1.0),
            mayChange=True,
        )
        self.connection_status_text = OnscreenText(
            text="Connections: unchecked",
            pos=(-1.28, 0.86),
            align=TextNode.ALeft,
            scale=0.036,
            fg=(0.02, 0.02, 0.02, 1.0),
            mayChange=True,
        )
        self.base.accept("c", self.run_sync_capture)
        self.base.accept("d", self.run_detection)
        self.base.accept("p", self.run_plan)
        self.base.accept("o", self.run_execute)

        if self.icp_result is not None:
            self.clear_detection_models()
            self.attach_global_registered_template_points(self.icp_result)
            self.attach_start_and_place_models(self.icp_result)
        print("Viewer colors: green=kept candidate points, gray=removed/context points, red=selected object, purple=global registered template points.")
        print("WRS viewer is ready. Press C to sync/capture, D for segmentation/completion ICP, P to plan, O to execute.")

    def current_action_sequence(self) -> Optional[int]:
        action_sequence = getattr(self.args, "action_sequence", None)
        return None if action_sequence is None else int(action_sequence)

    def advance_action_sequence(self) -> Optional[int]:
        action_sequence = self.current_action_sequence()
        if action_sequence is None:
            return None
        next_action_sequence = action_sequence + 1
        self.args.action_sequence = next_action_sequence
        return next_action_sequence

    def compute_scene_points(self) -> np.ndarray:
        if self.ctx is not None:
            static_mask = self.ctx.candidate_mask | self.ctx.removed_mask
            scene_points = self.ctx.capture.points_world[static_mask]
            if len(scene_points) > 0:
                return scene_points
            return self.ctx.capture.points_world
        if self.icp_result is not None:
            try:
                pick_pose = homomat_to_pose(bottle_homomat_for_icp(self.icp_result))
                place_pose, _place_pos_source, _place_rot_source = resolve_place_pose(self.args, pick_pose)
                return np.vstack([pick_pose[0], place_pose[0]])
            except Exception:
                pass
        return np.array([[0.58, -0.12, 0.28]], dtype=np.float64)

    @staticmethod
    def detach_models(models: list[object]) -> None:
        for model in models:
            for method_name in ("detach", "remove"):
                method = getattr(model, method_name, None)
                if method is None:
                    continue
                try:
                    method()
                    break
                except Exception:
                    continue
        models.clear()

    def clear_detection_models(self) -> None:
        self.detach_models(self.detection_models)

    def clear_plan_models(self) -> None:
        try:
            self.base.taskMgr.remove(self.animation_task_name)
        except Exception:
            pass
        if self.animation_data is not None:
            for attr in ("mesh_model", "obj_model"):
                model = getattr(self.animation_data, attr, None)
                if model is not None:
                    self.detach_models([model])
            self.animation_data = None
        self.detach_models(self.plan_models)

    def clear_static_scene_models(self) -> None:
        self.detach_models(self.static_scene_models)

    def clear_scene_obstacle_models(self) -> None:
        self.detach_models(self.scene_obstacle_models)

    def clear_robot_sync_models(self) -> None:
        self.detach_models(self.robot_sync_models)

    def clear_synced_scene_models(self) -> None:
        self.clear_plan_models()
        self.clear_detection_models()
        self.clear_static_scene_models()
        self.clear_scene_obstacle_models()
        self.clear_robot_sync_models()
    def attach_static_pointclouds(self) -> None:
        if self.ctx is None:
            return
        args = self.ctx.args
        candidate_points = self.ctx.capture.points_world[self.ctx.candidate_mask]
        removed_points = self.ctx.capture.points_world[self.ctx.removed_mask]
        candidate_points, _ = box_object_icp.voxel_downsample_arrays(candidate_points, None, args.candidate_voxel)
        removed_points, _ = box_object_icp.voxel_downsample_arrays(removed_points, None, args.removed_voxel)
        max_points = int(max(0, getattr(self.args, "scene_max_points", 0)))
        total_points = len(candidate_points) + len(removed_points)
        if max_points > 0 and total_points > max_points:
            candidate_limit = max(1, int(max_points * len(candidate_points) / max(1, total_points)))
            removed_limit = max(1, max_points - candidate_limit)
            candidate_points, _ = sync_scene.downsample_points(candidate_points, None, candidate_limit)
            removed_points, _ = sync_scene.downsample_points(removed_points, None, removed_limit)
        point_size = float(getattr(self.args, "scene_point_size", args.point_size))
        if len(removed_points) > 0:
            removed_model = self.mgm.gen_pointcloud(
                removed_points,
                rgba=np.array([0.55, 0.55, 0.55, 0.7]),
                point_size=point_size,
            )
            removed_model.attach_to(self.base)
            self.static_scene_models.append(removed_model)
        if len(candidate_points) > 0:
            candidate_model = self.mgm.gen_pointcloud(
                candidate_points,
                rgba=np.array([0.0, 0.85, 0.15, 0.78]),
                point_size=point_size,
            )
            candidate_model.attach_to(self.base)
            self.static_scene_models.append(candidate_model)

    def attach_scene_obstacles(self) -> None:
        try:
            if self.ctx is not None:
                box_homomat = self.ctx.box_transform
            elif self.icp_result is not None:
                box_homomat = load_homomat(self.icp_result.box_transform_path, "box")
            else:
                return
            _planning_obstacles, display_obstacles = build_obstacle_lists(self.args, box_homomat, include_display=True)
            for obstacle in display_obstacles:
                obstacle.attach_to(self.base)
                self.scene_obstacle_models.append(obstacle)
        except Exception as exc:
            print(f"[real_pipeline] Warning: could not attach scene obstacles: {exc}")

    def attach_synced_robot(self, jnt_values: Optional[np.ndarray], jaw_width: Optional[float]) -> None:
        if jnt_values is None:
            return
        from wrs.robot_sim.robots.ur7e.ur7e_withouttable_dh76 import UR7EDH76

        self.clear_robot_sync_models()
        robot = UR7EDH76(enable_cc=True, ik_solver="ikfast")
        robot.goto_given_conf(jnt_values=np.asarray(jnt_values, dtype=float))
        if jaw_width is not None:
            try:
                robot.jaw_to(jawwidth=sim_pick.clamp_jaw_width(robot, float(jaw_width)))
            except Exception as exc:
                print(f"[real_pipeline] Warning: could not set synced jaw width: {exc}")
        robot_model = robot.gen_meshmodel(
            alpha=0.78,
            toggle_tcp_frame=True,
            toggle_flange_frame=False,
            toggle_jnt_frames=False,
        )
        robot_model.attach_to(self.base)
        self.robot_sync_models.append(robot_model)
        print(f"[real_pipeline] synced robot display joints(deg): {sim_pick.format_jnts_deg(jnt_values, digits=2)}")

    def refresh_synced_scene_display(self, sync_metadata: dict) -> None:
        self.clear_synced_scene_models()
        self.attach_static_pointclouds()
        self.attach_scene_obstacles()
        self.attach_synced_robot(
            sync_metadata.get("current_jnt_values"),
            sync_metadata.get("current_jaw_width"),
        )

    def resolve_object_model_path(self, icp: ObjectIcpResult) -> Path:
        object_model_path = self.args.object_model if self.args.object_model is not None else icp.bottle_model_path
        object_model_path = object_model_path.resolve()
        if not object_model_path.exists():
            raise FileNotFoundError(f"Object model not found: {object_model_path}")
        return object_model_path

    def load_pick_and_place_poses(
        self,
        icp: ObjectIcpResult,
    ) -> tuple[Path, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], str, str]:
        object_model_path = self.resolve_object_model_path(icp)
        bottle_homomat = bottle_homomat_for_icp(icp)
        pick_pose = homomat_to_pose(bottle_homomat)
        place_pose, place_pos_source, place_rot_source = resolve_place_pose(self.args, pick_pose)
        return object_model_path, pick_pose, place_pose, place_pos_source, place_rot_source

    def attach_start_and_place_models(self, icp: ObjectIcpResult) -> None:
        object_model_path, pick_pose, place_pose, place_pos_source, place_rot_source = self.load_pick_and_place_poses(icp)
        start_model = sim_pick.make_object_model(
            object_model_path,
            pick_pose,
            name="estimated_pick_start_bottle",
            alpha=0.55,
            rgb=np.array([1.0, 0.76, 0.18]),
        )
        place_model = sim_pick.make_object_model(
            object_model_path,
            place_pose,
            name="planned_place_goal_bottle",
            alpha=0.32,
            rgb=np.array([0.2, 0.9, 0.45]),
        )
        start_model.attach_to(self.base)
        start_model.show_cdprim()
        place_model.attach_to(self.base)
        place_model.show_cdprim()
        self.detection_models.extend([start_model, place_model])

        frame_len = 0.085
        for pose, color in ((pick_pose, np.array([1.0, 0.76, 0.18])), (place_pose, np.array([0.2, 0.9, 0.45]))):
            frame = self.mgm.gen_frame(pos=pose[0], rotmat=pose[1], ax_length=frame_len, ax_radius=0.002)
            frame.attach_to(self.base)
            self.detection_models.append(frame)
            marker = self.mgm.gen_sphere(pos=pose[0], radius=0.01, rgb=color, alpha=0.85)
            marker.attach_to(self.base)
            self.detection_models.append(marker)
        if np.linalg.norm(place_pose[0] - pick_pose[0]) > 1e-8:
            arrow = self.mgm.gen_arrow(
                spos=pick_pose[0],
                epos=place_pose[0],
                rgb=np.array([0.2, 0.45, 1.0]),
                alpha=0.72,
                stick_radius=0.004,
            )
            arrow.attach_to(self.base)
            self.detection_models.append(arrow)
        print(
            "[real_pipeline] start/place preview ready: "
            f"pick={sim_pick.format_vec(pick_pose[0], digits=6)}, "
            f"place={sim_pick.format_vec(place_pose[0], digits=6)} "
            f"({place_pos_source}, rot={place_rot_source})"
        )

    def attach_global_registered_template_points(self, icp: ObjectIcpResult) -> None:
        if icp.global_registered_path is None:
            return
        try:
            import open3d as o3d

            registered_pcd = o3d.io.read_point_cloud(str(icp.global_registered_path))
            registered_points = np.asarray(registered_pcd.points, dtype=np.float64)
            max_points = int(max(0, getattr(self.args, "scene_max_points", 0)))
            if max_points > 0 and len(registered_points) > max_points:
                registered_points, _ = sync_scene.downsample_points(registered_points, None, max_points)
            if len(registered_points) == 0:
                return
            point_size = float(max(getattr(self.args, "scene_point_size", 0.002), 0.0035))
            registered_model = self.mgm.gen_pointcloud(
                registered_points,
                rgba=np.array([*box_object_icp.GLOBAL_REGISTERED_POINTS_RGB, 0.95]),
                point_size=point_size,
            )
            registered_model.attach_to(self.base)
            self.detection_models.append(registered_model)
        except Exception as exc:
            print(f"[real_pipeline] Warning: failed to draw global registered template points: {exc}")

    def attach_detection_result(self, summary: dict, selected_mask: np.ndarray) -> None:
        self.clear_detection_models()
        if self.ctx is not None:
            selected_points = self.ctx.capture.points_world[selected_mask]
            selected_points, _ = box_object_icp.voxel_downsample_arrays(
                selected_points,
                None,
                self.ctx.args.selected_voxel,
            )
            if len(selected_points) > 0:
                selected_model = self.mgm.gen_pointcloud(
                    selected_points,
                    rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                    point_size=max(self.ctx.args.point_size, 0.0025),
                )
                selected_model.attach_to(self.base)
                self.detection_models.append(selected_model)
        self.icp_result = object_icp_result_from_summary(summary)
        self.attach_global_registered_template_points(self.icp_result)
        self.attach_start_and_place_models(self.icp_result)
        summary_path = write_summary(self.args, self.icp_result, None)
        print(f"[real_pipeline] summary after ICP: {summary_path}")

    def update_connection_status_display(self, metadata: dict) -> None:
        items = []
        robot_status = metadata.get("robot_status")
        if robot_status is not None:
            for check in robot_status.checks:
                if check.name == "robot RTDE control":
                    items.append(f"RTDE-C:{'OK' if check.ok else 'FAIL'}")
                elif check.name == "robot RTDE receive":
                    items.append(f"RTDE-R:{'OK' if check.ok else 'FAIL'}")
                elif check.name == "DH76 gripper":
                    items.append(f"Gripper:{'OK' if check.ok else 'FAIL'}")
        camera_status = metadata.get("camera_status")
        if camera_status is not None:
            items.append(f"Camera:{'OK' if camera_status.ok else 'FAIL'}")
        if not items:
            items.append("unchecked")
        self.connection_status_text.setText("Connections: " + " | ".join(items))

    def require_fresh_scene(self, action: str) -> bool:
        if self.environment_stale:
            self.status_text.setText(f"Environment changed after O. Press C before {action}.")
            print(f"[real_pipeline] {action} blocked: press C to refresh the scene after execution.")
            return False
        if self.ctx is None:
            self.status_text.setText(f"No current scene. Press C before {action}.")
            return False
        return True

    def run_sync_capture(self) -> None:
        if self.running:
            print("[real_pipeline] Another operation is already running; ignoring C key.")
            return
        self.running = True
        self.status_text.setText("C sync: checking connections, reading robot state, capturing Mech-Eye point cloud...")
        self.connection_status_text.setText("Connections: checking...")
        try:
            ctx, metadata = capture_synced_context(self.args)
            self.ctx = ctx
            self.icp_result = None
            self.planning_result = None
            self.environment_stale = False
            self.detect_attempt_count = 0
            self.update_connection_status_display(metadata)
            self.refresh_synced_scene_display(metadata)
            output_dir = metadata["output_dir"]
            action_sequence = self.current_action_sequence()
            sequence_text = "" if action_sequence is None else f" Action sequence {action_sequence}."
            self.status_text.setText(f"C sync done: {output_dir.name}.{sequence_text} Press D to estimate pose.")
            if action_sequence is None:
                print(f"[real_pipeline] C sync capture ready: {output_dir}")
            else:
                print(f"[real_pipeline] C sync capture ready: {output_dir}; action sequence={action_sequence}")
        except Exception as exc:
            self.status_text.setText(f"C sync failed: {exc}. Press C to retry.")
            self.connection_status_text.setText("Connections: FAIL (see console)")
            print(f"[real_pipeline] C sync failed: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False

    def run_execute(self) -> None:
        if self.running:
            print("[real_pipeline] Another operation is already running; ignoring O key.")
            return
        if self.environment_stale:
            self.status_text.setText("Environment changed after O. Press C before executing again.")
            return
        if self.planning_result is None or self.planning_result.rtde_plan is None:
            self.status_text.setText("No RTDE plan yet. Press P after pose estimation first.")
            return
        self.running = True
        completed_action_sequence = self.current_action_sequence()
        dry_run = bool(self.args.execute_dry_run or self.args.mock)
        use_move_l_compliant = bool(self.args.use_move_l_compliant)
        mode = "dry-run" if dry_run else "REAL ROBOT"
        compliant_mode = "moveL_compliant" if use_move_l_compliant else "joint-path approach"
        sequence_text = "" if completed_action_sequence is None else f", action sequence {completed_action_sequence}"
        self.status_text.setText(f"O execution starting ({mode}, {compliant_mode}{sequence_text})...")
        if completed_action_sequence is None:
            print(f"[real_pipeline] O execution starting ({mode}, {compliant_mode})...")
        else:
            print(
                f"[real_pipeline] O execution starting ({mode}, {compliant_mode}); "
                f"action sequence={completed_action_sequence}"
            )
        rtde_robot = object()
        try:
            if not dry_run:
                from robot_con.ur.ur7e_dh76_rtde import UR7EDH76_RTDE

                rtde_robot = UR7EDH76_RTDE(robot_ip=self.args.robot_ip, gp_port=self.args.gp_port)
                print("[real_pipeline] Opening gripper before RTDE execution...")
                rtde_robot.open_gripper()
            else:
                print("[real_pipeline] Dry-run: skipping pre-execution gripper open.")
            log = rtde_utils.execute_rtde_execution_plan(
                rtde_robot=rtde_robot,
                plan=self.planning_result.rtde_plan,
                dry_run=dry_run,
                max_start_joint_error=np.radians(float(self.args.max_start_joint_error_deg)),
                use_move_l_compliant=use_move_l_compliant,
            )
            next_action_sequence = self.advance_action_sequence()
            if completed_action_sequence is None or next_action_sequence is None:
                completion_text = f"O execution complete ({mode}, {compliant_mode}): {len(log)} segment(s)."
            else:
                completion_text = (
                    f"O execution complete ({mode}, {compliant_mode}): action sequence "
                    f"{completed_action_sequence} done; next is {next_action_sequence}. {len(log)} segment(s)."
                )
            self.status_text.setText(f"{completion_text} Press C before any new D/P/O.")
            print(f"[real_pipeline] {completion_text}")
            for entry in log:
                print(f"[real_pipeline]   {entry}")
        except Exception as exc:
            self.status_text.setText(f"O execution failed: {exc}. Press C before retrying D/P/O.")
            print(f"[real_pipeline] O execution failed: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            disconnect = getattr(rtde_robot, "disconnect", None)
            if disconnect is not None:
                try:
                    disconnect()
                except Exception as exc:
                    print(f"[real_pipeline] Warning: RTDE disconnect failed: {exc}")
            self.environment_stale = True
            self.running = False
    def run_detection(self) -> None:
        if self.running:
            print("[real_pipeline] Another operation is already running; ignoring D key.")
            return
        if not self.require_fresh_scene("D detection"):
            return
        self.running = True
        self.detect_attempt_count += 1
        self.status_text.setText(f"Attempt {self.detect_attempt_count}: segmenting, completing point cloud, running surface ICP...")
        self.planning_result = None
        self.clear_plan_models()
        self.clear_detection_models()
        try:
            sam_settings = refresh_sam_task_settings(self.args)
            if sam_settings is not None:
                apply_sam_task_settings(self.ctx.args, sam_settings)
            self.args.bottle_template = "surface"
            self.args.bottle_template_prompt_gui = False
            self.args.completion_bottle_template = "surface"
            self.ctx.args.bottle_icp = False
            self.ctx.args.bottle_template = "surface"
            self.ctx.args.bottle_template_prompt_gui = False
            self.ctx.args.bottle_template_ply = self.args.bottle_template_ply
            self.ctx.args.completion_matching = True
            self.ctx.args.completion_bottle_icp = True
            self.ctx.args.completion_bottle_template = "surface"
            self.ctx.args.completion_bottle_template_ply = self.args.completion_bottle_template_ply
            summary, _masks, selected_mask = box_object_icp.run_segmentation_and_bottle_icp_attempt(self.ctx)
            self.attach_detection_result(summary, selected_mask)
            bottle_summary, pose_source = bottle_pose_summary_from_summary(summary)
            template_pointcloud_id = bottle_summary.get("template", "surface")
            print(
                f"[real_pipeline] template_pointcloud_id={template_pointcloud_id} pose_source={pose_source}",
                flush=True,
            )
            self.status_text.setText(
                f"Attempt {self.detect_attempt_count}: completion ICP done. "
                f"Template={template_pointcloud_id} fitness={bottle_summary['icp_fitness']:.4f}. Press P to plan."
            )
        except Exception as exc:
            self.status_text.setText(f"Attempt {self.detect_attempt_count}: failed: {exc}. Press D to retry.")
            print(f"[real_pipeline] Detection attempt {self.detect_attempt_count} failed: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
    def run_plan(self) -> None:
        if self.running:
            print("[real_pipeline] Another operation is already running; ignoring P key.")
            return
        if not self.require_fresh_scene("P planning"):
            return
        if self.args.skip_plan:
            self.status_text.setText("Planning is disabled by --skip-plan.")
            return
        if self.icp_result is None:
            self.status_text.setText("No estimated start pose yet. Press D first.")
            return
        self.running = True
        action_sequence = self.current_action_sequence()
        sequence_text = "" if action_sequence is None else f" for action sequence {action_sequence}"
        self.status_text.setText(f"Planning pick-only path{sequence_text}...")
        if action_sequence is None:
            print("[real_pipeline] Planning pick-only path.")
        else:
            print(f"[real_pipeline] Planning pick-only path for action sequence {action_sequence}.")
        self.clear_plan_models()
        try:
            plan_args = copy.copy(self.args)
            plan_args.dry_run = True
            planning = run_or_skip_plan(plan_args, self.icp_result)
            if planning is None:
                self.status_text.setText("Planning skipped.")
                return
            self.planning_result = planning
            self.attach_plan_result(planning)
            summary_path = write_summary(self.args, self.icp_result, planning)
            rtde_path = "" if planning.rtde_plan_path is None else f" RTDE={planning.rtde_plan_path}"
            done_sequence_text = "" if planning.action_sequence is None else f" for action sequence {planning.action_sequence}"
            self.status_text.setText(
                f"Planning done{done_sequence_text}: {len(planning.mot_data.jv_list)} frames. Press O to execute."
            )
            print(
                f"[real_pipeline] summary after planning{done_sequence_text}: "
                f"{summary_path}{rtde_path}"
            )
        except Exception as exc:
            self.status_text.setText(f"Planning failed: {exc}. Adjust pose/options, then press P again.")
            print(f"[real_pipeline] Planning failed: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
    def attach_plan_result(self, planning: PlanningResult) -> None:
        mot_data = planning.mot_data
        if len(mot_data) == 0:
            print("[real_pipeline] No motion frames to visualize.")
            return

        marker_robot = sim_pick.make_robot()
        marker_robot.backup_state()
        try:
            for index in range(0, len(mot_data), max(1, sim_pick.RESULT_TRAIL_STRIDE)):
                jnt_values, ee_values, _obj_pose, _mesh = mot_data[index]
                marker_robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
                marker = self.mgm.gen_sphere(
                    pos=marker_robot.gl_tcp_pos,
                    radius=0.006,
                    rgb=np.array([1.0, 0.35, 0.05]),
                    alpha=0.75,
                )
                marker.attach_to(self.base)
                self.plan_models.append(marker)
        finally:
            marker_robot.restore_state()
        self.start_motion_animation(planning)

    def start_motion_animation(self, planning: PlanningResult) -> None:
        object_model_path = planning.object_model_path

        class AnimationData:
            def __init__(self, motion_data):
                self.counter = 0
                self.motion_data = motion_data
                self.robot = sim_pick.make_robot()
                self.mesh_model = None
                self.obj_model = None

        data = AnimationData(planning.mot_data)
        self.animation_data = data

        def update(task):
            if data.mesh_model is not None:
                self.detach_models([data.mesh_model])
                data.mesh_model = None
            if data.obj_model is not None:
                self.detach_models([data.obj_model])
                data.obj_model = None
            if data.counter >= len(data.motion_data):
                data.counter = 0

            jnt_values, ee_values, obj_pose, cached_mesh = data.motion_data[data.counter]
            if cached_mesh is not None:
                data.mesh_model = cached_mesh
            else:
                data.robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
                data.mesh_model = data.robot.gen_meshmodel(
                    alpha=0.72,
                    toggle_tcp_frame=True,
                    toggle_flange_frame=False,
                    toggle_jnt_frames=False,
                )
            if data.mesh_model is not None:
                data.mesh_model.attach_to(self.base)

            if obj_pose is not None and cached_mesh is None:
                data.obj_model = sim_pick.make_object_model(
                    object_model_path,
                    (np.asarray(obj_pose[0], dtype=float), np.asarray(obj_pose[1], dtype=float)),
                    name="interactive_animated_held_object",
                    alpha=0.65,
                    rgb=np.array([0.95, 0.72, 0.18]),
                )
                data.obj_model.attach_to(self.base)

            data.counter += 1
            return task.again

        self.base.taskMgr.doMethodLater(
            sim_pick.RESULT_ANIMATION_INTERVAL,
            update,
            self.animation_task_name,
            appendTask=True,
        )

    def run(self) -> None:
        self.base.run()


def should_run_interactive(args: SimpleNamespace) -> bool:
    if args.interactive is not None:
        return bool(args.interactive)
    return args.object_summary is None and not bool(args.dry_run)


def main() -> None:
    args = make_runtime_config()
    normalize_paths(args)
    apply_action_sequence_start_settings(args)

    # Real workflow always starts empty. Press C to create the current scene.
    args.capture_dir = None
    args.ply = None
    args.image = None
    args.box_transform = None
    args.object_summary = None
    args.object_output_dir = None
    preload_point_hint_model(args)
    preload_yolo_model(args)
    app = InteractiveBottlePickPlaceApp(args, ctx=None, initial_icp=None)
    app.run()


if __name__ == "__main__":
    main()