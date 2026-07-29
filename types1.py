from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import numpy as np
from yanjiuyuan import pick_place_rtde_utils as rtde_utils

# 自定义类
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
    # 仿真预计算的“抓取点真实法兰盘位姿” [x,y,z,rx,ry,rz]（get_real_tcp_pose_from_conf 推算），
    # 运行时替代 getActualTCPPose()，锚定搬运 moveL 航点。
    predicted_grasp_real_tcp: Optional[np.ndarray] = None


@dataclass
class RtdeObjectPose:
    pos: np.ndarray
    rotmat: np.ndarray