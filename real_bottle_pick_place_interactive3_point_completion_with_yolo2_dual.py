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
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import threading
import queue
from typing import Any, Optional
import numpy as np
import cv2

from time import perf_counter, strftime
import open3d as o3d
import traceback

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, Notify, Vec3, Point3
from wrs import mgm
from wrs.robot_con.ur.ur7e_dh76_rtde import UR7EDH76_RTDE
import wrs.visualization.panda.world as wd
from yanjiuyuan import point_hint_segment as phs
from yanjiuyuan.yolo_detect2 import BottleDetector

REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from yanjiuyuan.constants import BOX_CAPTURE_ROOT, BOTTLE_ROBOT_SIDE_PLACE_POS, BOTTLE_ROBOT_SIDE_PLACE_POSE_pos, CAMERA_TO_WORLD, PICK_LIFT_MAX_Z # noqa: E402
from yanjiuyuan import box_object_pointcloud_sam_completion_template_icp_with_yolo2 as box_object_icp  # noqa: E402
from yanjiuyuan import connection_status as conn_status  # noqa: E402
from yanjiuyuan import pick_place_rtde_utils as rtde_utils  # noqa: E402
from yanjiuyuan import sim_pick_and_place as sim_pick  # noqa: E402
from yanjiuyuan import sync_real_ur7e_mech_eye_box_env as sync_scene  # noqa: E402
from yanjiuyuan.constants import REAL_PIPELINE_CONFIG
from types1 import ObjectIcpResult, PlanningResult, RtdeObjectPose
import utils
from wrs.drivers.devices.Mech_eye.Mech_camera import CaptureImage
import real_bottle_pick_place_interactive3_point_completion_with_yolo2_dual as dual_pipeline  # noqa: E402
from wrs import ppp
from PIL import Image as _PILImage

BOX_OBJECT_SCRIPT = Path(__file__).resolve().parent / "box_object_pointcloud_sam_completion_template_icp_with_yolo2.py"
DEFAULT_OBJECT_MODEL_PATH = REAL_PIPELINE_CONFIG["bottle_stl"]
DEFAULT_GRASP_PICKLE_PATH = REAL_PIPELINE_CONFIG["grasp_pickle"]
DEFAULT_GRASP_DIR = DEFAULT_GRASP_PICKLE_PATH.parent
DEFAULT_SAM_TASK_CONFIG_PATH = REAL_PIPELINE_CONFIG["sam_task_config"]
DEFAULT_COMPLETION_ADAPOINTR_SCRIPT = REAL_PIPELINE_CONFIG["completion_adapointr_script"]
DEFAULT_COMPLETION_ADAPOINTR_CHECKPOINT = REAL_PIPELINE_CONFIG["completion_adapointr_checkpoint"]

sync_scene.CAM_TO_WORLD = CAMERA_TO_WORLD   # 相机外参来自于constants.py


def make_runtime_config() -> SimpleNamespace:
    config = copy.deepcopy(REAL_PIPELINE_CONFIG)
    return SimpleNamespace(**config)


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


def current_action_sequence_key(args: SimpleNamespace) -> Optional[str]:
    action_sequence = getattr(args, "action_sequence", None)
    if action_sequence is None:
        return None
    try:
        return str(int(action_sequence))
    except (TypeError, ValueError):
        return str(action_sequence).strip()


def action_sequence_settings_for_args(args: SimpleNamespace) -> dict[str, Any]:
    # 已移除对 action_sequence_config.json 的依赖：抓取规划不再依据配置文件挑选特定
    # 抓取/位姿，而是对任意检测到的瓶子位姿进行抓取（arbitrary-pose grasping）。
    return {}


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
        config_path = utils.resolve_path(Path(config_path))
        if config_path is not None:
            candidates.append((config_path.parent / raw_path).resolve())
    candidates.append(utils.resolve_path(raw_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
        "config_path": None,
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
    config_path = utils.resolve_path(Path(config_path))
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
        # 抓取顺序优先级推理模型（grasp_sequence）
        priority_order=args.priority_order,
        priority_checkpoint=args.priority_checkpoint,
        priority_config=args.priority_config,
        priority_yolo=args.priority_yolo,
        priority_device=args.priority_device,
        priority_show=args.priority_show,
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
        start_conf_deg=getattr(args, "start_conf_deg", None),
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


def capture_synced_context(
    args: SimpleNamespace,
    fixed_start_conf_deg: Optional[Any] = None,
    camera_instance: object = None,  # 传入相机实例避免重复连接
    robot_state_override: Optional[tuple] = None,  # (jnt_rad, tcp_pos, jaw_width) 或 None
)-> tuple[box_object_icp.PipelineContext, dict]:
    """拍照 + 箱子检测 + 构建 PipelineContext。

    Args:
        args: 运行时配置。
        fixed_start_conf_deg: 当提供此参数时（双线程/并行模式），跳过机器人连接，
            直接使用该值（角度）作为 start_conf_deg。避免与机器人线程的 RTDE 连接冲突。
        camera_instance: 复用的相机长连接实例。
        robot_state_override: 当提供时（并行模式），跳过临时 RTDE 连接，
            直接使用 (jnt_rad, tcp_pos, jaw_width) 作为 summary 所需的机器人状态，
            避免与主线程的 RTDE 回移产生寄存器冲突。
    输出:
        ctx:这一张照片的核心上下文, 只活这一张照片的处理周期。
        metadata:同拍的旁路诊断快照（原始点云、箱子检测结果、拍照时机器人关节/位姿/夹爪、连接状态）

    """
    # 相机拍照
    output_dir = sync_scene.make_output_dir(args.capture_root, None)
    sync_args = make_sync_capture_args(args, output_dir)
    box_detection = None
    current_jnt_values = None
    current_tcp_pos = None
    current_jaw_width = None
    robot_status = None
    camera_status = None

    if fixed_start_conf_deg is not None:
        # ===== 双线程模式：跳过机器人连接，仅拍照 + 检测 =====
        print("[real_pipeline] (dual-thread) 跳过机器人连接，直接拍照。")
        with utils.timed_step("Mech capture"):
            pcd, rgb_path, colored_ply_path, camera_status, pixel_indices, depth = conn_status.capture_mech_eye_pointcloud_checked(
                output_dir=output_dir,
                ply_out=sync_args.ply_out,
                depth_scale=sync_args.depth_scale,
                depth_trunc=sync_args.depth_trunc,
                save_ply=sync_args.save_ply,
                return_pixel_indices=True,
                return_depth=True,

                camera=camera_instance,  # 传入相机实例，避免重复连接
            )
    else:
        # ===== 原始模式：连接机器人 + 拍照 + 读取状态 =====
        provider = sync_scene.make_robot_provider(sync_args)
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
            # 相机拍照
            with utils.timed_step("Mech capture"):    # 1.3733s
                pcd, rgb_path, colored_ply_path, camera_status, pixel_indices, depth = conn_status.capture_mech_eye_pointcloud_checked(
                    output_dir=output_dir,
                    ply_out=sync_args.ply_out,
                    depth_scale=sync_args.depth_scale,
                    depth_trunc=sync_args.depth_trunc,
                    save_ply=sync_args.save_ply,
                    return_pixel_indices=True,
                    return_depth=True,

                    camera=camera_instance,  # 传入相机实例，避免重复连接
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
            # 箱子位姿检测
            with utils.timed_step("Detect Box"):  # 0.16s
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

    # ===== 公共后处理：点云转换、箱子检测（双线程模式）、上下文构建 =====
    if fixed_start_conf_deg is not None:
        # 双线程模式下在此处做点云转换和箱子检测
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
        with utils.timed_step("Detect Box"):
            box_detection = sync_scene.detect_box(sync_args, scene_data)
        if box_detection is None:
            raise RuntimeError("C sync did not detect the box pose.")
        detected_box_transform = sync_args.box_transform_out
        if detected_box_transform is None or not Path(detected_box_transform).exists():
            raise FileNotFoundError(f"C sync did not write detected box transform: {detected_box_transform}")

        # 双线程/并行模式：优先使用已知抓取起点机器人状态，避免与主线程 RTDE 回移冲突
        if robot_state_override is not None:
            current_jnt_values, current_tcp_pos, current_jaw_width = robot_state_override
        else:
            try:
                rtde_robot_temp = UR7EDH76_RTDE(robot_ip=args.robot_ip, gp_port=args.gp_port)
                current_jnt_values = np.asarray(rtde_robot_temp.get_jnt_values(), dtype=float)
                current_tcp_pos, _ = rtde_robot_temp.fk(jnt_values=current_jnt_values)
                current_jaw_width = float(np.asarray(rtde_robot_temp.get_gripper_width(), dtype=float))
                rtde_robot_temp.disconnect()
            except Exception as e:
                print(f"[real_pipeline] Warning: Failed to read robot state for summary: {e}")
                current_jnt_values = np.zeros(6, dtype=float)
                current_tcp_pos = np.zeros(3, dtype=float)
                current_jaw_width = 0.0

        sync_scene.write_summary(
            sync_args,
            scene_data,
            current_jnt_values,
            current_tcp_pos,
            current_jaw_width,
            box_detection,
        )

    args.capture_dir = output_dir
    args.image = scene_data.rgb_path if scene_data.rgb_path is not None else args.image
    args.ply = scene_data.world_ply_path if scene_data.world_ply_path is not None else scene_data.colored_ply_path
    detected_box_transform = output_dir / "detected_box_transform.txt"
    args.box_transform = detected_box_transform if detected_box_transform.exists() else None
    args.object_summary = None
    args.object_output_dir = output_dir / "box_object_extraction"
    if fixed_start_conf_deg is not None:
        args.start_conf_deg = np.asarray(fixed_start_conf_deg, dtype=float)
    elif current_jnt_values is not None:
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
        depth=depth,
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
        return utils.load_homomat(Path(bottle_summary["icp_transform_path"]), "bottle ICP")

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
        return utils.load_homomat(Path(world_path), "completion bottle world ICP")

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

    yolo_model_key = str(args.yolo_model)
    if getattr(args, "_yolo_detector", None) is not None and getattr(args, "_yolo_model_key", None) == yolo_model_key:
        return
    print(f"[real_pipeline] Loading YOLO model at startup: {args.yolo_model}")
    yolo_detector = BottleDetector(
        model_path=str(args.yolo_model),
    )
    # 覆盖默认阈值（yolo_detect2 构造函数不接受 conf/iou 参数）
    yolo_detector.conf = args.yolo_conf
    yolo_detector.iou = args.yolo_iou
    args._yolo_detector = yolo_detector
    args._yolo_model_key = yolo_model_key
    print("[real_pipeline] YOLO model loaded.")


def move_to_grasp_start(args: SimpleNamespace) -> None:
    """在拍照/抓取前将机器人运动到抓取起点（抓取起点 = 拍照位）关节位置。

    配置 grasp_start_conf_rad 已是弧度，直接作为弧度传给 move_jnts（不再做 np.radians）。

    Args:
        args: 运行时配置（需包含 grasp_start_conf_rad, robot_ip, gp_port, mock 等字段）。
    """
    home_conf_rad = getattr(args, "grasp_start_conf_rad", None)
    if home_conf_rad is None:
        return
    home_conf_rad = np.asarray(home_conf_rad, dtype=float).reshape(-1)

    dry_run = bool(getattr(args, "execute_dry_run", False) or getattr(args, "mock", False))
    mode = "dry-run" if dry_run else "REAL ROBOT"
    print(f"[real_pipeline] 回归抓取起点 ({mode})，目标关节角(弧度): {np.round(home_conf_rad, 4).tolist()}")

    if dry_run:
        print("[real_pipeline] Dry-run: 跳过实际运动。")
        return
    rtde_robot = UR7EDH76_RTDE(
        robot_ip=args.robot_ip,
        gp_port=args.gp_port,
    )
    try:
        rtde_robot.move_jnts(home_conf_rad, vel=0.5, acc=0.5, wait=True)
        print("[real_pipeline] 已运动到抓取起点。")
    finally:
        disconnect = getattr(rtde_robot, "disconnect", None)
        if disconnect is not None:
            try:
                disconnect()
            except Exception as exc:
                print(f"[real_pipeline] Warning: RTDE disconnect failed: {exc}")

def move_to_capture_point(args: SimpleNamespace) -> None:
    """在拍照前将机器人运动到独立的拍照点（capture_conf_rad）关节位置。

    拍照点应与抓取起点区分：机器人先到拍照点（机械臂后撤、不遮挡视野）完成拍照，
    再移到抓取起点执行 RTDE 抓取。配置 capture_conf_rad 已是弧度，直接作为弧度传给
    move_jnts（不做 np.radians）。

    Args:
        args: 运行时配置（需包含 capture_conf_rad, robot_ip, gp_port, mock 等字段）。
    """
    conf_rad = getattr(args, "capture_conf_rad", None)
    if conf_rad is None:
        return
    conf_rad = np.asarray(conf_rad, dtype=float).reshape(-1)

    dry_run = bool(getattr(args, "execute_dry_run", False) or getattr(args, "mock", False))
    mode = "dry-run" if dry_run else "REAL ROBOT"
    print(f"[real_pipeline] 回归拍照点 ({mode})，目标关节角(弧度): {np.round(conf_rad, 4).tolist()}")

    if dry_run:
        print("[real_pipeline] Dry-run: 跳过实际运动。")
        return
    rtde_robot = UR7EDH76_RTDE(
        robot_ip=args.robot_ip,
        gp_port=args.gp_port,
    )
    try:
        rtde_robot.move_jnts(conf_rad, vel=0.5, acc=0.5, wait=True)
        print("[real_pipeline] 已运动到拍照点。")
    finally:
        disconnect = getattr(rtde_robot, "disconnect", None)
        if disconnect is not None:
            try:
                disconnect()
            except Exception as exc:
                print(f"[real_pipeline] Warning: RTDE disconnect failed: {exc}")


def execute_rtde_plan_direct(planning, args: SimpleNamespace, on_transfer_to_place: callable = None) -> list:
    """在真实机器人上执行 RTDE 抓取计划（同步执行）。

    Args:
        planning: PlanningResult 对象，包含 rtde_plan。
        args: 运行时配置（需包含 robot_ip, gp_port, mock 等字段）。
        on_transfer_to_place: 可选回调，在机器人【刚要放下】时调用一次——即第三段搬运 moveL
            （transfer_role=3，lower to place）开始执行前。此时机械臂已到放置点正上方（远离取料台），
            即将竖直下落，用于在该时刻并行启动下一轮拍照+处理，与机器人下落+松手+离开重叠，省掉一轮等待。
            若搬运段不是三段 moveL 形态（无 transfer_role=3 段），回退为进入 open_gripper（松手）段时触发。
            回调在主线程 RTDE 执行期间被调用，注意其内部不要创建 RTDE 连接（CaptureProcessWorker
            已通过 fixed_start_conf_deg + robot_state_override 规避，不会与主线程的 RTDE 执行抢输入寄存器）。

            说明：触发点选在下落段开始前，是因为搬运前两段（上移/水平）机器人仍握着物体在取料台上空
            或途中移动，会遮挡/带偏桌面上待抓的下一物体；而到达放置点正上方后，机械臂已离开取料台上空，
            取料台视野干净，此时拍照质量最好，且拍照与下落+松手并行、互不影响。

    Returns:
        执行日志列表（每个 segment 一个 dict）。
    """
    if planning is None or planning.rtde_plan is None:
        raise RuntimeError("No RTDE plan available to execute.")

    dry_run = bool(getattr(args, "execute_dry_run", False) or getattr(args, "mock", False))
    use_move_l_compliant = bool(getattr(args, "use_move_l_compliant", False))
    mode = "dry-run" if dry_run else "REAL ROBOT"
    compliant_mode = "moveL_compliant" if use_move_l_compliant else "joint-path approach"
    print(f"[real_pipeline] RTDE execution starting ({mode}, {compliant_mode})...")

    rtde_robot = object()
    try:
        if not dry_run:
            rtde_robot = UR7EDH76_RTDE(
                robot_ip=args.robot_ip,
                gp_port=args.gp_port,
            )
            print("[real_pipeline] Opening gripper before RTDE execution...")
            rtde_robot.open_gripper()
        else:
            print("[real_pipeline] Dry-run: skipping pre-execution gripper open.")

        # 逐 segment 执行 RTDE 计划
        log = []
        transfer_trigger_fired = False
        # 物料搬运段：标记了 anchor_to_actual_tcp 的三段 moveL，执行时以
        # 【实际 TCP 位姿】为基准重算目标位姿（仿真 fk 位姿与真实机器人有标定偏差）。
        transfer_segments = [
            s for s in planning.rtde_plan.segments if s.metadata.get("anchor_to_actual_tcp")
        ]
        transfer_anchored = False
        for segment_idx, segment in enumerate(planning.rtde_plan.segments):
            # ⭐ 关键：机器人【刚要放下】（第三段 moveL：lower to place 开始前）时，并行触发下一轮拍照+处理。
            # 此时机械臂已在放置点正上方、离开取料台上空、视野干净，即将下落，拍照与下落+松手并行，比等到
            # open_gripper 再触发多省一段下落时间。若搬运段不是三段 moveL 形态（无 transfer_role=3），
            # 回退为进入 open_gripper 段时触发。
            if (
                on_transfer_to_place is not None
                and not transfer_trigger_fired
                and (segment.metadata.get("transfer_role") == 3 or segment.name == "open_gripper")
            ):
                transfer_trigger_fired = True
                _trig_where = "刚要放下（lower to place 前）" if segment.metadata.get("transfer_role") == 3 else "open_gripper（松手前，回退触发）"
                print(f"[real_pipeline] 🎥 {_trig_where}，触发下一轮拍照+处理（与下落/松手并行）...")
                on_transfer_to_place()

            # ---- 以【实际 TCP 位姿】为基准重算物料搬运三段 moveL 目标位姿 ----
            # 机器人在 close_gripper 之后已到达抓取点，此刻读取真实 TCP 位姿，
            # 由其派生：① 竖直 +Z 抬起 ② 水平移到放置点正上方（姿态保持抓取姿态）
            # ③ 竖直下落到放置点（姿态保持抓取姿态）。这样搬运路径锚定在真实位姿，
            # 不受仿真/真实标定偏差影响。仅在实机（非 dry-run）下执行一次。
            if (
                not dry_run
                and segment.metadata.get("anchor_to_actual_tcp")
                and not transfer_anchored
            ):
                # 用规划阶段预计算的真实抓取点位姿
                _pred = getattr(planning, "predicted_grasp_real_tcp", None)
                if _pred is None:
                    print("[real_pipeline] ⚠️ 无预计算抓取点真实位姿，回退使用规划（仿真）航点")
                else:
                    try:
                        _actual = np.asarray(_pred, dtype=float).reshape(6)  # 基坐标系 6D: x,y,z,rx,ry,rz
                        _ax, _ay, _az = _actual[:3] # 实际位置
                        _arx, _ary, _arz = _actual[3:]  # 实际轴角
                        _dep = float(transfer_segments[0].metadata["depart_distance"])  # 上移距离
                        _px, _py, _pz = transfer_segments[0].metadata["place_pos_base"]
                        _lift_z = min(_az + _dep, PICK_LIFT_MAX_Z) # 限制高度，以免过高到不了
                        print(
                            f"[real_pipeline] 📍 预计算真实位姿重算物料 moveL 航点: "
                            f"grasp=({_ax:.4f},{_ay:.4f},{_az:.4f}), "
                            f"lift_z={_lift_z:.4f}, above/place=({_px:.4f},{_py:.4f},{_pz:.4f})"
                        )
                        _wp1 = [_ax, _ay, _lift_z, _arx, _ary, _arz]
                        _wp2 = [_px, _py, min(_lift_z, 0.63), _arx, _ary, _arz]
                        _wp3 = [_px, _py, _pz, _arx, _ary, _arz]
                        for _s in transfer_segments:
                            _role = _s.metadata.get("transfer_role")
                            _s.pose = list(_wp1 if _role == 1 else _wp2 if _role == 2 else _wp3)
                        print(
                            f"[real_pipeline] 📍 预计算真实位姿重算物料 moveL 航点: "
                            f"grasp=({_ax:.4f},{_ay:.4f},{_az:.4f}), "
                            f"lift_z={_lift_z:.4f}, above/place=({_px:.4f},{_py:.4f},{_pz:.4f})"
                        )
                    except Exception as _exc:
                        print(f"[real_pipeline] ⚠️ 使用预计算抓取点真实位姿失败，回退使用规划（仿真）航点: {_exc}")
                transfer_anchored = True

            entry = {"name": segment.name, "command": segment.command}

            # ---- push 离开段：以【实际 TCP 位姿】为基准竖直上抬（与抓取上移一致）----
            # 注意：不能复用上面的“搬运段重算”逻辑（它要求 place_pos_base 元数据，push 没有，
            # 会 KeyError 回退并把 FK 规划位姿误当实际位姿，导致 moveL 冲向错误位置触发保护停）。
            # 这里单独在 push_leave 段执行前，读取真实 TCP 位姿（基系 6D），仅抬 Z 分量
            # depart_distance、保持姿态，保证离开方向竖直向上、且从机器人“此刻真实所在”出发。
            if (
                not dry_run
                and segment.metadata.get("path_type") == "push_leave"
            ):
                # 用规划阶段预计算的 push 末端真实位姿
                _pred = segment.metadata.get("predicted_real_tcp")
                # ⭐ 若本计划含“往箱子中心推”的力控段（push_to_center_compliant），力控推完后
                # 实际 XY 已偏离接触点且行程不确定（可能提前触力软停），预计算的接触点位姿
                # 不再是机器人“此刻真实所在”——改为运行时读取实际 TCP 位姿，仅抬 Z 竖直离开，
                # 避免 moveL 横向拖回接触点。读取失败时回退预计算位姿。
                _has_center_push = any(
                    s.metadata.get("path_type") == "push_center" for s in planning.rtde_plan.segments
                )
                if _has_center_push:
                    try:
                        _pred = np.asarray(rtde_robot.getActualTCPPose(), dtype=float).reshape(6).tolist()
                        print("[real_pipeline] 📍 push_leave 存在中心推段，改用运行时实际 TCP 位姿锚定上抬")
                    except Exception as _exc:
                        print(f"[real_pipeline] ⚠️ push_leave 读取实际 TCP 失败，回退预计算位姿: {_exc}")
                if _pred is None:
                    print("[real_pipeline] ⚠️ push_leave 无预计算真实位姿，回退使用规划航点")
                else:
                    try:
                        _actual = np.asarray(_pred, dtype=float).reshape(6)  # 基坐标系 6D: x,y,z,rx,ry,rz
                        _dep = float(segment.metadata.get("depart_distance", 0.0))
                        _lift_z = min(float(_actual[2]) + _dep, PICK_LIFT_MAX_Z)  # 限制高度，避免超出可达
                        segment.pose = [
                            float(_actual[0]), float(_actual[1]), _lift_z,
                            float(_actual[3]), float(_actual[4]), float(_actual[5]),
                        ]
                        print(
                            f"[real_pipeline] 📍 push_leave 以预计算真实位姿竖直上抬: "
                            f"z {float(_actual[2]):.4f} → {_lift_z:.4f}"
                        )
                    except Exception as _exc:
                        print(f"[real_pipeline] ⚠️ push_leave 使用预计算真实位姿失败，回退使用规划航点: {_exc}")

            print(f"[real_pipeline]   执行 segment {segment_idx}: {segment.name}")
            segment_start = perf_counter()

            # 执行当前 segment
            segment_log = rtde_utils.execute_rtde_execution_plan(
                rtde_robot=rtde_robot,
                plan=rtde_utils.RtdeExecutionPlan(segments=[segment]),
                dry_run=dry_run,
                max_start_joint_error=np.radians(float(args.max_start_joint_error_deg)),
                use_move_l_compliant=use_move_l_compliant,  # 是否使用力控
                # 力控段允许的正常停下原因：到达距离 / 接触物体(max_tcp_force) / 卡住(stalled) / 侧向顺从(lateral_deviation)
                allowed_compliant_stop_reasons=("distance", "max_tcp_force", "stalled", "lateral_deviation"),
            )

            segment_elapsed = perf_counter() - segment_start
            entry["elapsed"] = segment_elapsed
            log.append(entry)

            print(f"[real_pipeline]   ✅ {segment.name} 完成 ({segment_elapsed:.2f}s)")

        print(f"[real_pipeline] RTDE execution complete ({mode}, {compliant_mode}): {len(log)} segment(s).")
        for entry in log:
            print(f"[real_pipeline]   {entry['name']}: {entry.get('elapsed', 0):.2f}s")
        return log

    finally:
        disconnect = getattr(rtde_robot, "disconnect", None)
        if disconnect is not None:
            try:
                disconnect()
            except Exception as exc:
                print(f"[real_pipeline] Warning: RTDE disconnect failed: {exc}")


def run_push_phase(
    args: SimpleNamespace,
    grasp_start_conf_rad: Any = None,
    ctx: Any = None,
) -> int:
    """所有物体都没有抓取时，按「检测顺序」逐个运行时规划并推开（不重新拍照、不重复 YOLO）。

    行为（与抓取 gen_pick_place_path 同构：遍历候选 + 筛选可行解）：
        1. 加载 push_pickle 中的全部候选 push 位姿（物体局部系，与抓取同格式 jaw_width/ac_pos/ac_rotmat）。
        2. 复用抓取规划阶段已算好的检测结果与逐物体 3D 位姿：直接取 ctx.detections / ctx.push_object_poses，
           不再调用 detect_and_rank（抓取规划内部已跑过 YOLO+ICP，结果挂在 ctx 上）。
        3. 对每个检测到的物体 i：取其 3D 位姿 obj_pose，先用与抓取规划【完全相同】的筛选函数
           planner.filter_grasps_sequence（障碍体=桌子+箱子壁+放置箱体、周围点云=remaining_pointcloud.ply）
           对【全部】候选 push 位姿做轻量可行性筛选（手爪-障碍碰撞→IK→点云碰撞→机器人-障碍碰撞），
           筛掉不可行候选；再仅对筛选通过的候选按 obj_pose 变换成基系 TCP、逐个调用 plan_push 尝试
           RRT 接近路径，第一个成功（返回非 None）即执行并 break。
           —— 这就是「遍历所有抓取姿态候选 + 与抓取同函数筛选 + 筛选通过者再规划」的步骤，与抓取同构。
        4. 运动剖面：RRT 接近 → 力控接触(moveL_compliant) → 手爪闭合 → moveL 竖直离开 → 手爪张开。
           不搬运到放置点、不放置力控下压。
        5. 不维护“已推开”去重集合：每次外层循环重新检测，按当前检测顺序推；
           若某物体被推开后变为可抓取，下一轮会回到抓取分支；同一位姿被反复检测到的风险
           由外层 max_outer_cycles 安全上限兜底。

    Args:
        ctx: 本 cycle 的 PipelineContext（抓取规划已写入 ctx.detections / ctx.push_object_poses）；
             为 None 时退化为按 pickle 全部顺序推（无法按实时检测排序）。

    Returns:
        成功推开的物体数量。
    """
    push_path = Path(REAL_PIPELINE_CONFIG.get("push_pickle"))
    if not push_path.exists():
        print(f"[push] ⚠️ 未找到 push pickle: {push_path}，跳过推开阶段")
        return 0

    # 加载 push 位姿（结构同抓取 pickle：jaw_width, ac_pos, ac_rotmat, hnd_pos, hnd_rotmat）
    sim_robot = sim_pick.make_robot()
    try:
        push_grasps = sim_pick.load_grasps(sim_robot, push_path)
    except Exception as e:
        print(f"[push] ❌ 加载 push pickle 失败: {e}")
        traceback.print_exc()
        return 0
    if not push_grasps:
        print("[push] ⚠️ push pickle 为空，跳过推开阶段")
        return 0
    print(f"[push] 已加载 {len(push_grasps)} 个候选 push 位姿（来自 {push_path.name}）")

    # push 计划的力控接触段（approach_pick_compliant）是 moveL_compliant，必须走力控执行。
    # 强制开启 use_move_l_compliant（影响 execute_rtde_plan_direct 对 moveL_compliant 段的处理）。
    args.use_move_l_compliant = True

    # push 计划从 sim_pick.DEFAULT_HOME_CONF 起步；把它对齐到真实抓取起点，
    # 使执行端机器人（已在抓取起点）与计划首帧一致，避免 start-pose 不匹配被 fail-fast。
    if grasp_start_conf_rad is not None:
        sim_pick.DEFAULT_HOME_CONF = np.asarray(grasp_start_conf_rad, dtype=float).reshape(-1)

    from yanjiuyuan import real_pipeline_planning as real_planning  # 延迟导入，避免与 dual 的循环依赖

    # ===== 复用抓取规划阶段已算好的检测结果与逐物体 3D 位姿（不重新拍照 / 不重复 YOLO）=====
    # 抓取规划 run_seg_icp_grasp 内部已跑过 detect_and_rank，并对每个物体算了 ICP 位姿，
    # 结果分别挂在 ctx.detections / ctx.push_object_poses 上；推开阶段直接取用即可。
    detections = getattr(ctx, "detections", None) if ctx is not None else None
    per_object_poses = getattr(ctx, "push_object_poses", {}) if ctx is not None else {}
    if detections is None:
        # 兜底：极少触发（抓取规划一定跑过 detect_and_rank）。无 ctx 时退化为按 pickle 全部顺序推。
        if ctx is not None:
            det_result = box_object_icp.detect_and_rank(ctx, show=False)
            detections = det_result[3] if det_result is not None else []
        else:
            detections = []
    print(f"[push] 复用抓取规划检测结果：{len(detections)} 个物体；"
          f"逐物体 3D 位姿可用 {len(per_object_poses)} 个")

    # 碰撞体：复用抓取规划同一套（箱子壁 / 目标瓶子 / 桌子 / 放置箱体）。
    # ctx 含实时检测到的箱子位姿 box_transform，与抓取规划 build_obstacle_lists 保持一致。
    box_homomat = None
    if ctx is not None and getattr(ctx, "box_transform", None) is not None:
        try:
            box_homomat = np.asarray(ctx.box_transform, dtype=float).reshape(4, 4)
        except Exception:
            box_homomat = None
            print("[push] ⚠️ ctx.box_transform 形状异常，退化为 桌面+放置箱体 碰撞体")

    # push 候选是否记录为「物体局部系」（与抓取候选一致，需按检测物体位姿变换成基系 TCP）。
    # 若为 False 则把 pickle 中的位姿直接当基系固定 TCP 用（旧行为）。
    candidates_object_local = bool(REAL_PIPELINE_CONFIG.get("push_candidates_object_local", True))

    # ===== 在路径规划前，先对所有推开候选做“与抓取规划相同”的筛选 =====
    # 复用抓取规划同一套筛选函数 planner.filter_grasps_sequence（wrs PickPlacePlanner）：
    #   对每个 goal_pose（此处 = 物体 3D 位姿 obj_pose）依次做：
    #     ① 手爪与障碍 mesh 碰撞 → ② IK 求解 → ③ 周围点云碰撞(raycasting) → ④ 机器人整体与障碍碰撞
    #   返回对全部 goal_pose 都通过的候选 gid 列表。这一步只做“轻量可行性”判断（IK + 碰撞），
    #   不做昂贵的 RRT 接近路径规划；仅对筛选通过的候选才调用 plan_push 做 RRT，显著提速。
    # 筛选所用障碍体 / 周围点云与抓取规划（gen_pick_approach_only_path）完全一致：
    #   障碍体 = build_obstacle_lists(args, box_homomat)（桌子+箱子壁+放置箱体）；
    #   周围点云 = remaining_pointcloud.ply（除被操作物体外的场景点云，不含被推瓶子本身）。

    push_collection = sim_pick.make_grasp_collection(sim_robot, push_grasps)

    # 障碍列表：与 plan_push / 抓取规划一致（一次构建，所有物体复用）
    if box_homomat is not None:
        obstacle_list, _ = build_obstacle_lists(args, box_homomat, include_display=False)
    else:
        obstacle_list = []
        if not getattr(args, "no_env", False):
            obstacle_list.append(sim_pick.make_table_obstacle())
        obstacle_list.append(real_planning.make_robot_side_place_box_collision_obstacle(show_cdprim=False))
    print(f"[push] 筛选碰撞体数: {len(obstacle_list)}（与抓取规划一致）")

    push_planner = ppp.PickPlacePlanner(sim_robot)

    # ===== 逐物体：先筛选全部候选，再对“筛选通过者”做 RRT 路径规划并取首个成功者执行 =====
    # 与抓取 gen_pick_place_path 同构：先 filter（这里用 filter_grasps_sequence）得到候选子集，
    # 再逐候选尝试生成完整路径，首个成功即执行。
    pushed_count = 0
    for i, detection in enumerate(detections):
        obj_pose = per_object_poses.get(i)
        if candidates_object_local and obj_pose is None:
            print(f"[push] 第 {i} 个物体无可用 3D 位姿（ICP 未成功），跳过")
            continue
        # ---- 步骤1：用与抓取规划相同的筛选函数，筛选该物体的全部 push 候选 ----
        if candidates_object_local and obj_pose is not None:
            # 周围点云必须逐物体取：每个物体的 remaining_pointcloud.ply 是在抓取规划阶段
            # 按「该物体的掩码」从场景裁剪出来的（场景减去该物体），互不相同。
            # 不能像之前那样在循环外读一个静态文件——否则所有物体共用同一份、筛选失真。
            # 优先直接用抓取规划阶段留在内存里的逐物体周围点云（ctx.remaining_pcd_by_obj），避免重新读盘。
            surrounding_pcd = getattr(ctx, "remaining_pcd_by_obj", {}).get(i) if ctx is not None else None
            src = "内存(ctx.remaining_pcd_by_obj)"
            if surrounding_pcd is None:
                # 兜底：从磁盘读取该物体专属的 remaining_pointcloud.ply（抓取规划 save_ply=True 时才落盘）
                remaining_ply_path = getattr(ctx, "push_object_remaining_pcd", {}).get(i) if ctx is not None else None
                if remaining_ply_path is None:
                    remaining_ply_path = getattr(args, "remaining_pointcloud_path", None)
                if remaining_ply_path is not None and Path(remaining_ply_path).exists():
                    surrounding_pcd = o3d.io.read_point_cloud(remaining_ply_path)
                    src = f"磁盘({remaining_ply_path})"
            print(f"[push] 第 {i} 个物体周围点云来源: "
                  f"{src if surrounding_pcd is not None else '无（筛选将跳过点云碰撞项）'}")
            sim_robot.backup_state()
            try:
                candidate_indices = push_planner.filter_grasps_sequence(
                    push_collection,
                    [obj_pose],                 # goal_pose = 物体 3D 位姿（与抓取 pick_pose 同义）
                    obstacle_list,              # 桌子+箱子壁+放置箱体
                    surrounding_pcd,            # 该物体专属的周围场景点云（已减去该物体本身）
                    sim_robot,
                    sim_robot.end_effector,
                    log_func=print,
                    toggle_dbg=False,
                )
            finally:
                sim_robot.restore_state()
            print(f"[push] ▶ 第 {i} 个物体：筛选通过 {len(candidate_indices)}/{len(push_grasps)} 个候选"
                  f"（其余被 手爪/障碍碰撞、IK 不可解、点云碰撞、机器人碰撞 筛掉）")
            if not candidate_indices:
                print(f"[push] ⚠️ 第 {i} 个物体所有候选均未通过筛选，跳过")
                continue
        else:
            # 非物体局部系（罕见兜底）：无 obj_pose 可作 goal_pose，退化为不筛选、逐个尝试
            candidate_indices = list(range(len(push_grasps)))

        # ---- 步骤2：仅对筛选通过的候选做 RRT 路径规划，首个成功即执行并 break ----
        print(f"[push] ▶ 第 {i} 个物体：对 {len(candidate_indices)} 个筛选候选逐个尝试 RRT 规划...")
        pushed_this = False
        for cand_idx in candidate_indices:
            g = push_grasps[cand_idx]
            # 候选（物体局部系）→ 机器人基系 TCP
            if candidates_object_local and obj_pose is not None:
                base_pos, base_rotmat = sim_pick.tcp_pose_from_object_pose(obj_pose, g)
            else:
                base_pos, base_rotmat = (np.asarray(g.ac_pos, dtype=float).reshape(3),
                                         np.asarray(g.ac_rotmat, dtype=float).reshape(3, 3))
            jaw = grasp_jaw_width(g)
            try:
                rtde_plan = real_planning.plan_push(
                    args,
                    push_pose=(base_pos, base_rotmat),
                    jaw_width=jaw,
                    object_model_path=sim_pick.OBJECT_MODEL_PATH,  # 目标瓶子（被推物体）模型，作为 obj_cmodel 传入规划器
                    box_homomat=box_homomat,  # 复用抓取规划的箱子碰撞体（箱子壁+目标瓶子+桌子+放置箱体）
                )
            except Exception as e:
                print(f"[push] ⚠️ 第 {i} 个物体候选#{cand_idx} plan_push 异常: {e}")
                traceback.print_exc()
                continue
            if rtde_plan is None:
                # 该候选 RRT 接近路径未生成（不可行）→ 筛选掉，尝试下一个候选
                continue
            # 包装为 PlanningResult 供 execute_rtde_plan_direct 执行（其期望 planning.rtde_plan）
            planning = PlanningResult(
                selected_grasp_index=None,
                action_sequence=None,
                mot_data=None,
                pick_pose=(base_pos, base_rotmat),
                place_pose=(base_pos, base_rotmat),
                object_model_path=REAL_PIPELINE_CONFIG["bottle_stl"],
                obstacle_names=[],
                place_pos_source="push",
                place_rot_source="push",
                rtde_plan=rtde_plan,
            )
            # 回到抓取起点，与计划首帧（DEFAULT_HOME_CONF=抓取起点）一致后再执行
            move_to_grasp_start(args)
            print(f"[push] 🤖 开始执行第 {i} 个物体（候选#{cand_idx}）推开...")
            try:
                execute_rtde_plan_direct(planning, args, on_transfer_to_place=None)
            except Exception as e:
                print(f"[push] ❌ 第 {i} 个物体候选#{cand_idx} 推开执行失败: {e}")
                traceback.print_exc()
                continue
            pushed_count += 1
            pushed_this = True
            print(f"[push] ✅ 第 {i} 个物体用候选#{cand_idx} 推开完成")
            break  # 该物体已成功推开，进入下一个物体
        if not pushed_this:
            print(f"[push] ⚠️ 第 {i} 个物体所有（筛选通过）候选均无法规划/执行，跳过")
    return pushed_count


class GraspPlanViewer:
    """一键抓取流水线结果查看器。

    显示静态场景点云、障碍物、ICP 配准结果（模板点云、抓取/放置位姿）、
    橙色 TCP 轨迹标记，以及逐帧机器人运动动画。
    """

    def __init__(self, ctx: box_object_icp.PipelineContext, result: dict, runtime_args: SimpleNamespace = None, world=None):

        _task_cat = Notify.ptr().getCategory("task")
        if _task_cat is not None:
            _task_cat.setSeverity(5)

        self.ctx = ctx
        self.args = runtime_args if runtime_args is not None else ctx.args
        self.planning = result["planning"]
        self.icp_result = result["icp_result"]
        self.mgm = mgm
        self.static_models: list[object] = []
        self.obstacle_models: list[object] = []
        self.detection_models: list[object] = []
        self.plan_models: list[object] = []
        self.animation_data = None
        self.animation_task_name = "grasp_plan_animation"
        self.auto_execute_task_name = "auto_execute_rtde"
        self.executed = False
        self.executing = False
        self.owns_world = world is None  # 是否由本 viewer 创建世界

        # 兼容 SimpleNamespace 没有的字段
        if not hasattr(ctx.args, "no_env"):
            ctx.args.no_env = False
        if not hasattr(ctx.args, "place_pos"):
            ctx.args.place_pos = None
        if not hasattr(ctx.args, "place_rpy_deg"):
            ctx.args.place_rpy_deg = None

        scene_points = ctx.capture.points_world[ctx.candidate_mask | ctx.removed_mask]
        if len(scene_points) == 0:
            scene_points = ctx.capture.points_world
        cam_pos, lookat_pos, extent = box_object_icp.compute_camera_from_points(scene_points)

        if world is not None:
            self.base = world
            # 复用已有世界时更新相机位置
            self.base.cam.setPos(cam_pos[0], cam_pos[1], cam_pos[2])
            self.base.cam.lookAt(
                Point3(lookat_pos[0], lookat_pos[1], lookat_pos[2]),
                Vec3(0, 0, 1),
            )
        else:
            self.base = wd.World(cam_pos=cam_pos, lookat_pos=lookat_pos, w=1280, h=720)

        frame_length = max(extent * 0.25, 0.03)
        frame_radius = max(frame_length * 0.015, 0.0005)
        mgm.gen_frame(ax_length=frame_length, ax_radius=frame_radius).attach_to(self.base)

        self._attach_static_pointclouds()
        self._attach_scene_obstacles()
        self._attach_icp_result()
        self._attach_plan_result()

        frame_count = (
            len(self.planning.mot_data.jv_list)
            if hasattr(self.planning.mot_data, "jv_list")
            else len(self.planning.mot_data)
        )
        self.status_text = OnscreenText(
            text=(
                f"Grasp plan ready: {frame_count} frames, "
                f"grasp #{self.planning.selected_grasp_index}. "
                f"RTDE: {self.planning.rtde_plan_path}"
            ),
            pos=(-1.28, 0.92),
            align=TextNode.ALeft,
            scale=0.044,
            fg=(0.02, 0.02, 0.02, 1.0),
            mayChange=True,
        )
        auto_execute = bool(getattr(self.args, "auto_execute", True))
        if auto_execute and self.planning.rtde_plan is not None:
            self.status_text.setText(
                f"Grasp plan ready: {frame_count} frames, "
                f"grasp #{self.planning.selected_grasp_index}. "
                f"Animation will auto-execute RTDE plan after playback."
            )
            print(
                "[real_pipeline] GraspPlanViewer ready. "
                "Orange trail = planned TCP path; animated robot shows motion playback. "
                "After animation, RTDE plan will auto-execute. "
                "Press O to execute immediately, or close the window to exit."
            )
            self._schedule_auto_execute(frame_count)
        else:
            self.status_text.setText(
                f"Grasp plan ready: {frame_count} frames, "
                f"grasp #{self.planning.selected_grasp_index}. "
                f"RTDE: {self.planning.rtde_plan_path}. Press O to execute."
            )
            print(
                "[real_pipeline] GraspPlanViewer ready. "
                "Orange trail = planned TCP path; animated robot shows motion playback. "
                "Press O to execute RTDE plan, or close the window to exit."
            )
        self.base.accept("o", self.execute_rtde_plan)

    # ---- static scene ----
    def _attach_static_pointclouds(self) -> None:
        args = self.ctx.args
        candidate_points = self.ctx.capture.points_world[self.ctx.candidate_mask]
        removed_points = self.ctx.capture.points_world[self.ctx.removed_mask]
        candidate_points, _ = box_object_icp.voxel_downsample_arrays(
            candidate_points, None, args.candidate_voxel,
        )
        removed_points, _ = box_object_icp.voxel_downsample_arrays(
            removed_points, None, args.removed_voxel,
        )
        if len(removed_points) > 0:
            model = self.mgm.gen_pointcloud(
                removed_points,
                rgba=np.array([0.55, 0.55, 0.55, 0.7]),
                point_size=args.point_size,
            )
            model.attach_to(self.base)
            self.static_models.append(model)
        if len(candidate_points) > 0:
            model = self.mgm.gen_pointcloud(
                candidate_points,
                rgba=np.array([0.0, 0.85, 0.15, 0.78]),
                point_size=args.point_size,
            )
            model.attach_to(self.base)
            self.static_models.append(model)

    # ---- scene obstacles (box, table, place box) ----
    def _attach_scene_obstacles(self) -> None:
        try:
            box_homomat = self.ctx.box_transform
            _planning_obstacles, display_obstacles = build_obstacle_lists(
                self.ctx.args, box_homomat, include_display=True,
            )
            for obstacle in display_obstacles:
                obstacle.attach_to(self.base)
                self.obstacle_models.append(obstacle)
        except Exception as exc:
            print(f"[real_pipeline] Warning: could not attach scene obstacles: {exc}")

    # ---- ICP result (registered template, pick/place poses) ----
    def _attach_icp_result(self) -> None:
        try:
            icp = self.icp_result

            # global registered template points
            if icp.global_registered_path is not None:
                registered_pcd = o3d.io.read_point_cloud(str(icp.global_registered_path))
                registered_points = np.asarray(registered_pcd.points, dtype=np.float64)
                if len(registered_points) > 0:
                    model = self.mgm.gen_pointcloud(
                        registered_points,
                        rgba=np.array([*box_object_icp.GLOBAL_REGISTERED_POINTS_RGB, 0.95]),
                        point_size=max(self.ctx.args.point_size, 0.0035),
                    )
                    model.attach_to(self.base)
                    self.detection_models.append(model)

            pick_pose = utils.homomat_to_pose(bottle_homomat_for_icp(icp))
            place_pose, _pos_src, _rot_src = resolve_place_pose(
                self.ctx.args, pick_pose,
            )

            # pick / place bottle models
            start_model = sim_pick.make_object_model(
                icp.bottle_model_path,
                pick_pose,
                name="estimated_pick_start_bottle",
                alpha=0.55,
                rgb=np.array([1.0, 0.76, 0.18]),
            )
            start_model.attach_to(self.base)
            start_model.show_cdprim()
            self.detection_models.append(start_model)

            place_model = sim_pick.make_object_model(
                icp.bottle_model_path,
                place_pose,
                name="planned_place_goal_bottle",
                alpha=0.32,
                rgb=np.array([0.2, 0.9, 0.45]),
            )
            place_model.attach_to(self.base)
            place_model.show_cdprim()
            self.detection_models.append(place_model)

            # coordinate frames and markers
            for pose, color in (
                (pick_pose, np.array([1.0, 0.76, 0.18])),
                (place_pose, np.array([0.2, 0.9, 0.45])),
            ):
                frame = self.mgm.gen_frame(
                    pos=pose[0], rotmat=pose[1], ax_length=0.085, ax_radius=0.002,
                )
                frame.attach_to(self.base)
                self.detection_models.append(frame)
                marker = self.mgm.gen_sphere(
                    pos=pose[0], radius=0.01, rgb=color, alpha=0.85,
                )
                marker.attach_to(self.base)
                self.detection_models.append(marker)

            # arrow from pick to place
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
        except Exception as exc:
            print(f"[real_pipeline] Warning: could not attach ICP result: {exc}")
            traceback.print_exc()

    # ---- plan result: trail markers + animation ----
    def _attach_plan_result(self) -> None:
        mot_data = self.planning.mot_data
        if len(mot_data) == 0:
            print("[real_pipeline] No motion frames to visualize.")
            return

        # draw TCP trail markers
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

        self._start_motion_animation()

    def _start_motion_animation(self) -> None:
        object_model_path = self.planning.object_model_path

        class AnimationData:
            def __init__(self, motion_data):
                self.counter = 0
                self.motion_data = motion_data
                self.robot = sim_pick.make_robot()
                self.mesh_model = None
                self.obj_model = None

        data = AnimationData(self.planning.mot_data)
        self.animation_data = data

        def update(task):
            # 可选：打印帧号以确认动画在运行（调试用）
            # print(f"Animation frame {data.counter}/{len(data.motion_data)}")

            # 移除上一帧的模型
            for attr in ("mesh_model", "obj_model"):
                model = getattr(data, attr, None)
                if model is not None:
                    for method_name in ("detach", "remove"):
                        method = getattr(model, method_name, None)
                        if method is not None:
                            try:
                                method()
                                break
                            except Exception:
                                continue
                    setattr(data, attr, None)

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
                    (
                        np.asarray(obj_pose[0], dtype=float),
                        np.asarray(obj_pose[1], dtype=float),
                    ),
                    name="animated_held_object",
                    alpha=0.65,
                    rgb=np.array([0.95, 0.72, 0.18]),
                )
                data.obj_model.attach_to(self.base)

            data.counter += 1
            # 设置下一次调用的延迟（即帧间隔）
            task.delayTime = sim_pick.RESULT_ANIMATION_INTERVAL
            return task.again

        # 只使用 add，任务首次将在下一帧执行
        self.base.taskMgr.add(update, self.animation_task_name)

    def _schedule_auto_execute(self, frame_count: int) -> None:
        """动画播放一遍后自动执行 RTDE 计划。"""
        interval = getattr(sim_pick, "RESULT_ANIMATION_INTERVAL", 0.05)
        delay = frame_count * interval + 1.0  # 播完一遍 + 1秒缓冲

        def auto_execute_task(task):
            if self.executed or self.executing:
                return task.done
            print(f"[real_pipeline] Auto-executing RTDE plan after {delay:.1f}s animation playback.")
            self.execute_rtde_plan()
            return task.done

        self.base.taskMgr.doMethodLater(
            delay, auto_execute_task, self.auto_execute_task_name, appendTask=True
        )
        print(f"[real_pipeline] Auto-execute scheduled in {delay:.1f}s (after one full animation playback).")

    def execute_rtde_plan(self) -> None:
        """在真实机器人上执行 RTDE 抓取计划。"""
        if self.executed:
            print("[real_pipeline] RTDE plan already executed. Close window to exit.")
            return
        if self.executing:
            print("[real_pipeline] Execution already in progress, please wait.")
            return
        if self.planning is None or self.planning.rtde_plan is None:
            self.status_text.setText("No RTDE plan available. Cannot execute.")
            print("[real_pipeline] No RTDE plan available to execute.")
            return

        self.executing = True
        self.status_text.setText("Executing RTDE plan on real robot...")

        # 停止动画
        try:
            self.base.taskMgr.remove(self.animation_task_name)
        except Exception:
            pass

        dry_run = bool(getattr(self.args, "execute_dry_run", False) or getattr(self.args, "mock", False))
        use_move_l_compliant = bool(getattr(self.args, "use_move_l_compliant", False))
        mode = "dry-run" if dry_run else "REAL ROBOT"
        compliant_mode = "moveL_compliant" if use_move_l_compliant else "joint-path approach"
        print(f"[real_pipeline] O execution starting ({mode}, {compliant_mode})...")

        rtde_robot = object()
        try:
            if not dry_run:
                rtde_robot = UR7EDH76_RTDE(
                    robot_ip=self.args.robot_ip,
                    gp_port=self.args.gp_port,
                )
                print("[real_pipeline] Opening gripper before RTDE execution...")
                rtde_robot.open_gripper()
            else:
                print("[real_pipeline] Dry-run: skipping pre-execution gripper open.")

            log = rtde_utils.execute_rtde_execution_plan(
                rtde_robot=rtde_robot,
                plan=self.planning.rtde_plan,
                dry_run=dry_run,
                max_start_joint_error=np.radians(float(self.args.max_start_joint_error_deg)),
                use_move_l_compliant=use_move_l_compliant,
                # 力控段允许的正常停下原因：到达距离 / 接触物体(max_tcp_force) / 卡住(stalled) / 侧向顺从(lateral_deviation)
                allowed_compliant_stop_reasons=("distance", "max_tcp_force", "stalled", "lateral_deviation"),
            )
            completion_text = f"O execution complete ({mode}, {compliant_mode}): {len(log)} segment(s)."
            self.status_text.setText(f"{completion_text} Close window to exit.")
            print(f"[real_pipeline] {completion_text}")
            for entry in log:
                print(f"[real_pipeline]   {entry}")
        except Exception as exc:
            self.status_text.setText(f"O execution failed: {exc}. Close window to exit.")
            print(f"[real_pipeline] O execution failed: {exc}")
            traceback.print_exc()
        finally:
            disconnect = getattr(rtde_robot, "disconnect", None)
            if disconnect is not None:
                try:
                    disconnect()
                except Exception as exc:
                    print(f"[real_pipeline] Warning: RTDE disconnect failed: {exc}")
            self.executed = True
            self.executing = False

    def run(self) -> None:
        self.base.run()


def should_run_interactive(args: SimpleNamespace) -> bool:
    if args.interactive is not None:
        return bool(args.interactive)
    return args.object_summary is None and not bool(args.dry_run)


class DualThreadPipeline:
    """双线程流水线：相机线程负责拍照+检测+规划，机器人线程负责运动+执行。

    流程：
      1. 机器人运动到拍照位 → 触发相机拍照
      2. 相机线程：拍照 → YOLO → SAM → 点云补全 → ICP → 抓取规划 → 放入队列
      3. 机器人线程：从队列获取计划 → 执行抓取放置 → 放手后触发下次拍照
      4. 机器人运动回拍照位（与相机拍照+处理并行）→ 等待下一个计划
      5. 循环 2-4 直到没有物体

    并行收益：相机拍照+处理（~12s）与机器人运动回拍照位（~3s）并行执行。
    """

    def __init__(self, args: SimpleNamespace):
        self.args = args
        self._plan_queue: queue.Queue = queue.Queue(maxsize=1)
        self._capture_trigger = threading.Event()
        self._stop_flag = threading.Event()
        self._pick_count = 0
        self._camera_error: Optional[BaseException] = None
        self._robot_error: Optional[BaseException] = None

    def camera_thread_func(self) -> None:
        """相机线程：等待触发 → 拍照 → 检测 → 规划 → 放入队列。"""
        try:
            while not self._stop_flag.is_set():
                print("[camera_thread] 等待拍照触发...")
                triggered = self._capture_trigger.wait(timeout=600)
                if not triggered or self._stop_flag.is_set():
                    print("[camera_thread] 超时或收到停止信号，退出。")
                    break
                self._capture_trigger.clear()

                # 使用 grasp_start_conf_rad（弧度）作为 start_conf_deg（机器人将在抓取起点执行）
                # fixed_start_conf_deg 内部按角度处理，因此先转为角度。
                _gs = getattr(self.args, "grasp_start_conf_rad", None)
                fixed_conf = np.degrees(np.asarray(_gs, dtype=float)).tolist() if _gs is not None else None
                print("[camera_thread] 开始拍照...")
                #
                ctx, _metadata = capture_synced_context(self.args, fixed_start_conf_deg=fixed_conf)
                print("[camera_thread] 拍照完成，开始 YOLO+SAM+ICP+规划...")
                result = box_object_icp.run_seg_icp_grasp(ctx, pipeline_module=dual_pipeline)

                if result is None:
                    print("[camera_thread] 未检测到物体，发送结束信号。")
                    self._plan_queue.put(None)
                    break

                self._pick_count += 1
                planning = result["planning"]
                print(f"[camera_thread] 第 {self._pick_count} 个物体规划完成，放入队列。")
                self._plan_queue.put(planning)
        except Exception:
            traceback.print_exc()
            self._camera_error = traceback.format_exc()
            self._plan_queue.put(None)  # 解除机器人线程阻塞

    def robot_thread_func(self) -> None:
        """机器人线程：运动到拍照位 → 触发拍照 → 等待计划 → 执行 → 触发下次拍照。"""
        try:
            # 首次：运动到拍照位，然后触发相机拍照
            print("[robot_thread] 首次运动到拍照位...")
            move_to_grasp_start(self.args)
            self._capture_trigger.set()

            while True:
                # 等待相机线程的计划
                print("[robot_thread] 等待相机线程的计划...")
                planning = self._plan_queue.get()  # 阻塞直到有计划

                if planning is None:
                    print(f"[robot_thread] 收到结束信号，共抓取 {self._pick_count} 个。流水线结束。")
                    break

                # 执行抓取放置（包含：接近 → 抓取 → 搬运 → 放置 → 放手 → 离开）
                print(f"[robot_thread] 第 {self._pick_count} 个物体 RTDE 计划已就绪，开始执行...")
                execute_rtde_plan_direct(planning, self.args)
                print(f"[robot_thread] 第 {self._pick_count} 个物体抓取完成（已放手）。")

                # 立即触发相机拍照（用于下次抓取）
                # 相机拍照+处理与机器人运动回拍照位并行执行
                self._capture_trigger.set()

                # 机器人运动回拍照位（与相机线程并行）
                move_to_capture_point(self.args)
                move_to_grasp_start(self.args)
        except Exception:
            traceback.print_exc()
            self._robot_error = traceback.format_exc()
            self._stop_flag.set()
            self._capture_trigger.set()  # 解除相机线程阻塞

    def run(self) -> None:
        """启动双线程并等待完成。"""
        camera_t = threading.Thread(target=self.camera_thread_func, name="camera_thread", daemon=True)
        robot_t = threading.Thread(target=self.robot_thread_func, name="robot_thread", daemon=True)
        camera_t.start()
        robot_t.start()
        robot_t.join()
        # 机器人线程结束后，确保相机线程也退出
        self._stop_flag.set()
        self._capture_trigger.set()
        camera_t.join(timeout=10)
        if self._camera_error:
            print(f"[real_pipeline] 相机线程异常:\n{self._camera_error}")
        if self._robot_error:
            print(f"[real_pipeline] 机器人线程异常:\n{self._robot_error}")
        print(f"[real_pipeline] 双线程流水线结束，共抓取 {self._pick_count} 个物体。")


class CaptureProcessWorker(threading.Thread):
    """并行拍照处理线程。

    在机器人松手后移到拍照点、再移到抓取起点的同时，异步完成：拍照 → YOLO检测 →
    SAM分割 → 点云补全 → ICP匹配 → 抓取规划。结果（ctx, planning_dict）存入 self.result。

    使用 fixed_start_conf_deg 模式（跳过机器人 RTDE 连接）：
      - fixed_start_conf_deg = 抓取起点（计划的 RTDE 执行起点）
      - robot_state_override = 拍照点机器人状态（拍照时机器人位于拍照点）
    因此本线程不创建任何 RTDE 连接，与主线程的回移不会产生寄存器冲突。
    """

    def __init__(self, args: SimpleNamespace, camera_instance: object, capture_conf_rad: Any, grasp_start_conf_rad: Any):
        super().__init__(name="CaptureProcessWorker", daemon=True)
        self.args = args
        self.camera_instance = camera_instance
        self.capture_conf_rad = None if capture_conf_rad is None else np.asarray(capture_conf_rad, dtype=float).reshape(-1)
        self.grasp_start_conf_rad = None if grasp_start_conf_rad is None else np.asarray(grasp_start_conf_rad, dtype=float).reshape(-1)
        self.result = None  # (ctx, planning_dict) | None
        self.error = None

    def run(self) -> None:
        try:
            if self.grasp_start_conf_rad is None:
                # 未配置抓取起点：走普通模式（连接 RTDE 读取状态）
                fixed_conf_deg = None
                override = None
            else:
                # 计划的 RTDE 执行起点 = 抓取起点（内部按角度处理）
                fixed_conf_deg = np.degrees(self.grasp_start_conf_rad).tolist()
                # 拍照时机器人位于拍照点：用拍照点状态写 summary，避免临时 RTDE 连接冲突
                if self.capture_conf_rad is not None:
                    override = (self.capture_conf_rad.copy(), None, 0.0)
                else:
                    override = (self.grasp_start_conf_rad.copy(), None, 0.0)
            ctx, _metadata = capture_synced_context(
                self.args,
                fixed_start_conf_deg=fixed_conf_deg,
                camera_instance=self.camera_instance,
                robot_state_override=override,
            )
            self.ctx = ctx  # 始终暴露 ctx（即使后续规划抛错），供主线程推开阶段复用已拍帧

            result = box_object_icp.run_seg_icp_grasp(ctx, pipeline_module=dual_pipeline)
            # result 可能为 None（未检测到物体）
            self.result = (ctx, result)
        except Exception as e:
            print(f"[CaptureProcessWorker] ❌ 拍照处理失败: {e}")
            self.error = e
            traceback.print_exc()


def main() -> None:
    args = make_runtime_config()
    utils.normalize_paths(args)

    # Real workflow always starts empty. Press C to create the current scene.
    args.capture_dir = None
    args.ply = None
    args.image = None
    args.box_transform = None
    args.object_summary = None
    args.object_output_dir = None
    preload_point_hint_model(args)
    preload_yolo_model(args)

    if getattr(args, "run_grasp", True):
        # ===== 预热相机并保持长连接 =====
        print("[real_pipeline] === 初始化相机（长连接模式）===")
        camera_instance = CaptureImage(save_directory=str(BOX_CAPTURE_ROOT))
        if not camera_instance.is_connected():
            raise ConnectionError("相机连接失败！请检查以太网连接。")
        print("[real_pipeline] ✅ 相机已连接，将复用此实例避免重复连接")
        # ============================================================
        # 抓取流水线（多物体模式）：
        #   首轮：到拍照点 → 拍照+处理（与移到抓取起点并行）→ RTDE执行(pick+place+松手)
        #   后续轮：RTDE执行时，在 transfer_to_place 结束后、open_gripper（松手）开头并行启动下一轮
        #           拍照+处理；机器人完成松手/离开后，在等待拍照处理线程期间并行回拍照点（不遮挡视野）
        #           再到抓取起点，使回移与处理重叠，处理一完成即可执行下一轮
        # ============================================================
        print("[real_pipeline] === 抓取流水线启动（多物体模式：下放时并行拍照处理）===")

        pick_count = 0
        capture_rad = getattr(args, "capture_conf_rad", None)
        grasp_start_rad = getattr(args, "grasp_start_conf_rad", None)

        # 并行拍照处理线程（在机器人下放期间运行，提前重叠下一轮拍照+处理）
        worker = None

        def _spawn_worker() -> None:
            """启动下一轮拍照+处理线程（不建 RTDE 连接，与主线程 RTDE 执行并行）。"""
            nonlocal worker
            worker = CaptureProcessWorker(args, camera_instance, capture_rad, grasp_start_rad)
            worker.start()

        try:
            # ===== 首轮：先到拍照点同步拍照+处理（无上一轮可复用）=====
            move_to_capture_point(args)  # 拍照点（机械臂后撤、不遮挡视野）
            _spawn_worker() #  拍照
            move_to_grasp_start(args)    # 拍照点 → 抓取起点（与拍照处理并行）
            worker.join(timeout=180.0)
            if worker.error is not None:
                print(f"[real_pipeline] ❌ 并行拍照处理线程异常: {worker.error}")
                raise worker.error
            worker_result = worker.result  # (ctx, planning_dict) | None
            worker = None

            # ===== 外层循环：检测 →（抓取循环 或 推开阶段）→ 再检测 =====
            # 每轮都重新拍照检测；若可抓取则进入抓取循环，否则进入推开阶段；
            # 抓取或推开完成后回到本循环重新检测，直到“本轮既无可抓取也无可推开”为止。
            push_count = 0
            max_outer_cycles = int(REAL_PIPELINE_CONFIG.get("max_outer_cycles", 40))    # 最大外层循环次数
            outer_cycle = 0
            redetect_pending = False  # 仅在“推开改动了场景”后才需重新拍照检测

            while outer_cycle < max_outer_cycles:
                outer_cycle += 1

                # ---- 检测阶段：仅在“上一轮推开改动了场景”后重新拍照+处理 ----
                # 抓取循环结束后 worker_result 已是本轮回移/松手期间并行拍好的新鲜结果（worker_A），
                # 无需再拍；若直接判定无抓取且无推开，则结束，不再无谓回拍照点补拍一次（worker_B）。
                if outer_cycle > 1 and redetect_pending:
                    move_to_capture_point(args)      # 拍照点（机械臂后撤、不遮挡视野）
                    _spawn_worker()                  # 拍照+处理（并行线程，不连 RTDE）
                    move_to_grasp_start(args)        # 拍照点 → 抓取起点（与处理并行）
                    worker.join(timeout=180.0)
                    if worker.error is not None:
                        print(f"[real_pipeline] ❌ 并行拍照处理线程异常: {worker.error}")
                        raise worker.error
                    worker_result = worker.result
                    worker = None
                    redetect_pending = False

                # 路由：有抓取路径 → 抓取循环；否则 → 推开阶段
                if worker_result is not None and worker_result[1] is not None:
                    # ===== 抓取循环：逐个抓取可抓物体 =====
                    while worker_result is not None and worker_result[1] is not None:
                        ctx, planning_result = worker_result
                        planning = planning_result["planning"]
                        pick_count += 1
                        print(f"[real_pipeline] 📋 第 {pick_count} 个物体规划完成")

                        # RTDE 执行（pick+place+松手）；在 open_gripper 开头并行启动下一轮拍照+处理
                        print(f"[real_pipeline] 🤖 开始执行第 {pick_count} 个物体抓取...")
                        exec_start = perf_counter()
                        execute_rtde_plan_direct(planning, args, on_transfer_to_place=_spawn_worker)
                        exec_elapsed = perf_counter() - exec_start
                        print(f"[real_pipeline] ✅ 第 {pick_count} 个物体抓取完成 (执行:{exec_elapsed:.2f}s)")

                        # 机器人回移（与拍照处理线程并行）
                        move_to_capture_point(args)  # 放置位 → 拍照点
                        move_to_grasp_start(args)    # 拍照点 → 抓取起点

                        # 等待下一轮拍照+处理完成（在 open_gripper 时已并行启动）
                        if worker is not None:
                            worker.join(timeout=180.0)
                            if worker.error is not None:
                                print(f"[real_pipeline] ❌ 并行拍照处理线程异常: {worker.error}")
                                raise worker.error
                            worker_result = worker.result
                            worker = None
                        else:
                            # 兜底：本轮计划没有 open_gripper 段（异常计划），拍照线程未启动。
                            # 回移已完成，这里同步启动并等待拍照处理（无并行重叠）。
                            print("[real_pipeline] ⚠️ 本轮未触发并行拍照（无 open_gripper 段），回退到同步拍照处理")
                            _spawn_worker()
                            worker.join(timeout=180.0)
                            if worker.error is not None:
                                print(f"[real_pipeline] ❌ 并行拍照处理线程异常: {worker.error}")
                                raise worker.error
                            worker_result = worker.result
                            worker = None

                    # 抓取循环结束（本轮已无可抓取物体）→ 回到外层循环重新检测
                    continue

                # ===== 无抓取路径 → 推开阶段：复用本 cycle 已拍帧按检测顺序推开（不重新拍照）=====
                did_push = 0
                if REAL_PIPELINE_CONFIG.get("push_enabled", False):
                    # ctx 来源：本轮 worker_result 成功时为 (ctx, None)；worker 抛错时为 worker.ctx
                    ctx_for_push = (
                        worker_result[0] if worker_result is not None
                        else (worker.ctx if worker is not None else None)
                    )
                    try:
                        did_push = run_push_phase(
                            args,
                            grasp_start_conf_rad=grasp_start_rad,
                            ctx=ctx_for_push,
                        )
                        push_count += did_push
                    except Exception as e:
                        print(f"[real_pipeline] ❌ 推开阶段异常: {e}")
                        traceback.print_exc()

                # 终止判定：本轮既无可抓取（worker_result[1] is None）也无可推开（did_push==0）→ 无可做，结束
                if did_push == 0:
                    print("[real_pipeline] 本轮无抓取且无可推开物体，流水线结束。")
                    break
                # 否则：推开后场景已变，下一轮需重新拍照检测（看推开后是否能抓取）
                redetect_pending = True
                print(f"[real_pipeline] 本轮推开 {did_push} 个物体，回到检测循环重新判断抓取/推开...")

            print(f"[real_pipeline] 流水线结束：共抓取 {pick_count} 个，推开 {push_count} 个，外层循环 {outer_cycle} 轮。")

        except KeyboardInterrupt:
            print("\n[real_pipeline] ⚠️  用户中断，正在停止...")
        except Exception as e:
            print(f"[real_pipeline] ❌ 主线程异常: {e}")
            traceback.print_exc()
        finally:
            # 等待可能仍在运行的拍照处理线程退出
            if worker is not None and worker.is_alive():
                print("[real_pipeline] 等待拍照处理线程退出...")
                worker.join(timeout=5.0)
            # 关闭相机长连接
            print("[real_pipeline] 关闭相机连接...")
            conn_status.close_camera_quietly(camera_instance)

            print("[real_pipeline] 🛑 流水线已停止")

    else:
        # ============================================================
        # 交互模式（C/D/P/O 键盘交互）：
        #   C: 同步机器人状态 + 拍照 + 检测箱子
        #   D: 分割 + 点云补全 + ICP 位姿估计
        #   P: 规划抓取路径（带可视化动画）
        #   O: 执行 RTDE 抓取放置
        # 创建 Panda3D 世界用于可视化与碰撞检测；相机在按下 C 时按需连接，
        # 不在此预建长连接（避免与交互内按需连接冲突）。
        # 进入方式：constants.py 的 REAL_PIPELINE_CONFIG 中将 run_grasp 设为 False。
        # ============================================================
        print("[real_pipeline] === 交互模式启动（C/D/P/O）===")
        app = InteractiveBottlePickPlaceApp(args, ctx=None, initial_icp=None)
        app.run()


# ============================================================================
# 拆分后导入被抽离的模块（置于所有 def 之后、__main__ 之前，避免循环导入）
# ============================================================================
from interactive_app import InteractiveBottlePickPlaceApp
from real_pipeline_planning import (
    build_obstacle_lists,                      # 原模块内部裸用 + box_object(pipeline_module) 调用
    grasp_jaw_width,
    bottle_homomat_for_icp,
    resolve_place_pose,
)
if __name__ == "__main__":
    main()