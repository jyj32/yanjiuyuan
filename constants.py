from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# 参数配置文件
YANJIUYUAN_DIR = Path(__file__).resolve().parent
MODEL_DIR = YANJIUYUAN_DIR / "models"
BOX_MODEL_PATH = MODEL_DIR / "box.STL"
BOX_TEMPLATE_PLY = MODEL_DIR / "box_z_parallel_surface_points.ply"
BOX_CAPTURE_ROOT = YANJIUYUAN_DIR / "captures"


# 相机外参矩阵
CAMERA_TO_WORLD = np.array(
    [
        [-0.998885, 0.022034, -0.041751, 0.647500],
        [-0.021796, -0.999744, -0.006148, 0.018000],
        [-0.041876, -0.005231, 0.999109, 1.267000],
        [0.000000, 0.000000, 0.000000, 1.000000],
    ],
    dtype=float,
)

# Default world-frame bottle place goal used by the offline ICP -> pick-and-place scripts.
# Tune this when you want to put the placed bottle closer to/farther from the robot.
BOTTLE_ROBOT_SIDE_PLACE_POS = (0.2, -0.5, 0.15) # 放置位置（世界坐标系下）
BOTTLE_ROBOT_SIDE_PLACE_POSE_pos = (0.46, -0.073, 0.35) # moveL的放置位置

# 抓取/放置规划距离参数（集中管理，供 sim_pick_and_place / real_pipeline_planning / dual 等调用）
PICK_APPROACH_DEPART_DISTANCE = 0.35  # 抓起后竖直抬离距离 (m)：抓取点 TCP 沿世界 +Z 抬离
PICK_LIFT_MAX_Z = 0.65  # 抓起后抬升航点的最大 Z 高度 (m)

Range3D = Optional[Tuple[Optional[float], Optional[float]]]

# Box template generation used by sample_box_surface.py.
BOX_TEMPLATE_N_SAMPLES = 10000
BOX_TEMPLATE_NORMAL_Z_TOLERANCE = 1e-6
BOX_TEMPLATE_SURFACE_Z_MIN = 0.21
BOX_TEMPLATE_INCLUDE_NORMALS = True
BOX_TEMPLATE_CENTER_BOTTOM = False
BOX_TEMPLATE_RANDOM_SEED: Optional[int] = 0

# Target preprocessing used by box_icp_from_saved_capture.py.
BOX_USE_BLUE_RGB_SEGMENTATION = True
BOX_BLUE_HUE_RANGE_DEG = (170.0, 280.0)
BOX_BLUE_MIN_SATURATION = 0.08
BOX_BLUE_MIN_VALUE = 0.02
BOX_BLUE_DOMINANCE_MARGIN = 0.0
# 箱子点云过滤范围
BOX_X_RANGE: Range3D = (0.30, 1.2) # 箱子长0.6m
BOX_Y_RANGE: Range3D = (-0.35, 0.35)    # 箱子宽0.4m
BOX_Z_RANGE: Range3D = (0.02, 0.35) # 箱子高0.23m
BOX_TARGET_VOXEL_DOWNSAMPLE = 0.008
BOX_MATCH_TARGET_Z_MIN = 0.21
BOX_MATCH_TARGET_Z_MAX = 0.24

BOX_KEEP_LARGEST_BLUE_CLUSTER = False
BOX_BLUE_CLUSTER_EPS = 0.035
BOX_BLUE_CLUSTER_MIN_POINTS = 80
BOX_REMOVE_STATISTICAL_OUTLIERS = False
BOX_OUTLIER_NB_NEIGHBORS = 30
BOX_OUTLIER_STD_RATIO = 2.0

# Registration settings.
BOX_REGISTRATION_VOXEL_SIZE = 0.02
BOX_TEMPLATE_TOP_Z_MIN = BOX_TEMPLATE_SURFACE_Z_MIN
BOX_TEMPLATE_NORMAL_Z_ABS_MIN = 0.95
BOX_OBB_LONG_SHORT_MIN_RATIO = 1.05
BOX_OBB_REQUIRE_LONG_SHORT_RATIO = False
BOX_OBB_HORIZONTAL_AXIS_MIN_NORM = 0.25
BOX_ICP_MAX_ITERATION = 40

# Output, viewer, and progress settings.
BOX_OUTPUT_DIR = None
BOX_SHOW_RESULT_VIEWER = True
BOX_TARGET_POINT_SIZE = 0.002
BOX_MODEL_POINT_SIZE = 0.003
BOX_FINAL_TARGET_VOXEL_DOWNSAMPLE = 0.006
BOX_FINAL_MODEL_VOXEL_DOWNSAMPLE = 0.006
BOX_WRITE_DEBUG_POINTCLOUDS = True
BOX_LOG_STEPS = True
BOX_OPEN3D_PRINT_PROGRESS = False
# Object extraction inside the detected box.
BOX_OBJECT_OUTPUT_DIR = None
BOX_OBJECT_INNER_XY_MARGIN = 0.0
# # 箱子内部可放置物体的几何区域,去掉箱子内壁的点云
BOX_OBJECT_CONCAVE_X_RANGE = (-0.28, 0.28)
BOX_OBJECT_CONCAVE_Y_RANGE = (-0.18, 0.18)
BOX_OBJECT_CONCAVE_Z_RANGE = (0.02, 0.30)

BOX_OBJECT_CONCAVE_REGION_RGB = (0.0, 0.75, 1.0)
BOX_OBJECT_TOP_OVERHANG_XY_MARGIN = 0.045
BOX_OBJECT_ABOVE_TOP_MARGIN = 0.12
BOX_OBJECT_REMOVE_BLUE_BOX_POINTS = False
BOX_OBJECT_SHOW_REMOVED_CONTEXT = True
BOX_OBJECT_REMOVED_CONTEXT_XY_MARGIN = 0.08
BOX_OBJECT_REMOVED_CONTEXT_Z_MARGIN = 0.04
BOX_OBJECT_AUTO_SEGMENT_BOX = False
BOX_OBJECT_PIXEL_COLOR_TOLERANCE = 2
BOX_OBJECT_PIXEL_MAPPING_MIN_RATIO = 0.70
BOX_OBJECT_CANDIDATE_VOXEL_DOWNSAMPLE = 0.003
BOX_OBJECT_REMOVED_VOXEL_DOWNSAMPLE = 0.006
BOX_OBJECT_SELECTED_VOXEL_DOWNSAMPLE = 0.003
BOX_OBJECT_SHOW_VIEWER = True
BOX_OBJECT_SHOW_BOX_MODEL = False
BOX_OBJECT_POINT_SIZE = 0.002

# sim_pick_and_place 必须在 MODEL_DIR 等常量定义之后再导入，否则会与 constants 形成循环导入
# （constants 曾在此文件顶部直接导入 sim_pick，而 sim_pick 顶层又需要 constants.MODEL_DIR）。
from yanjiuyuan import sim_pick_and_place as sim_pick  # noqa: E402

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
    "sam_task_config": YANJIUYUAN_DIR / "sam_task_config.json",
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
    "yolo_model": YANJIUYUAN_DIR / "models" / "bottle_detect2.pt",
    "yolo_conf": 0.528,
    "yolo_iou": 0.70,
    "no_yolo": False,
    # ---- 抓取顺序优先级推理模型（grasp_sequence）----
    "priority_order": True, # 是否使用抓取顺序优先级推理模型
    "priority_checkpoint": YANJIUYUAN_DIR / "grasp_sequence" / "best.pt",
    "priority_config": YANJIUYUAN_DIR / "grasp_sequence" / "deploy_config.json",
    "priority_yolo": YANJIUYUAN_DIR / "grasp_sequence" / "yolo11s.pt",
    "priority_device": "auto",
    "priority_show": False,
    "show_object_viewer": False,
    "show_box_model": False,
    "bottle_template": "surface",
    "bottle_template_ply": None,
    "bottle_template_prompt_gui": False,
    "bottle_stl": YANJIUYUAN_DIR / "models" / "bottle.stl",
    "bottle_voxel": None,
    "bottle_template_voxel": None,
    "bottle_global_ransac_n": None,
    "bottle_global_ransac_attempts": None,
    "bottle_icp_max_iteration": None,

    # SAM point completion and final bottle pose estimation.
    "completion_matching": True,
    "completion_template": "surface",
    "completion_template_ply": None,
    "completion_adapointr_script": YANJIUYUAN_DIR.parent / "poind_cloud_completion" / "v2" / "pcn_train" / "AdapoinTr" / "infer_AdaPoinTr.py",
    "completion_adapointr_checkpoint": YANJIUYUAN_DIR.parent / "poind_cloud_completion" / "v2" / "pcn_train" / "AdapoinTr" / "log" / "train_AdaPoinTr_corrosion" / "checkpoints" / "best_combo.pth",
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
    "use_move_l_compliant": True,  # 是否使用力控

    # Compliant pick approach. Tune these here, not from the command line.
    "compliant_force": 40.0,    # 力控的力
    "compliant_vel": 0.06,    # 力控的速度
    "compliant_lateral_tolerance": 0.01,
    "compliant_lateral_stop_tolerance": "auto",
    "compliant_force_frame": "direction",
    "compliant_axes": None,
    "compliant_zero_ft_sensor": True,   # 力控前清零 FT 传感器基线，使接触力检测准确
    "compliant_max_tcp_force": 20.0,    # 接触反力阈值，达到后将退出力控
    "compliant_timeout": 15.0,    # 比默认(max(2.0,3*d/vel))更宽松的安全兜底，避免慢速接近误杀
    "compliant_dwell_after_stop": 1.0,    # 力控接触/到位软停后继续保压的时长(s)

    # Compliant place press（放置力控下压坐实/压实）。Tune these here, not from the command line.
    "place_press_enabled": True,
    "place_press_distance": 0.20,
    "place_press_force": 40.0,
    "place_press_vel": 0.08,
    "place_press_lateral_tolerance": 0.01,
    "place_press_lateral_stop_tolerance": "auto",
    "place_press_zero_ft_sensor": True,
    "place_press_max_tcp_force": 20.0,
    "place_press_timeout": 15.0,
    "place_press_dwell_after_stop": 0,    # 放置力控下压软停后继续保压的时长(s)

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
    "grasp_pickle": sim_pick.GRASP_PICKLE_PATH, # 抓取pickle文件
    "push_pickle": YANJIUYUAN_DIR / "grasps" / "bottle_dh76_push.pickle", # 推开pickle文件

    # Push（推开）阶段配置：所有物体都没有抓取时，按检测顺序重新检测每个物体，
    # 查 push_pickle 中对应检测序号的推开位姿，有则运行时路径规划（RRT 接近 + 力控接触 + 闭合 + 张开 + moveL 离开）并推开。
    "push_enabled": True,            # 是否启用“无抓取→推开”阶段（默认关闭，避免改变既有抓取流水线行为）
    "push_depart_distance": 0.35,     # 推开后竖直离开距离 (m)，复用 compliant 的“接触/保压”设置
    "push_leave_vel": 0.1,            # 离开 moveL 速度 (m/s)
    "push_leave_acc": 0.3,            # 离开 moveL 加速度 (m/s^2)
    "max_outer_cycles": 20,          # 抓取/推开外层循环最大轮数（安全上限，正常会因“本轮无抓取且无可推开”提前结束）

    # Push（推开）阶段力控接触参数：独立于“抓取 Compliant pick approach”的力控设置，
    # 统一在此调参。不再从命令行 args.compliant_* 读取（那批键本未定义，运行期会 AttributeError）。
    "push_compliant_force": 30.0,                 # 推开力控的力
    "push_compliant_vel": 0.08,                   # 推开力控的速度
    "push_compliant_lateral_tolerance": 0.02,     # 侧向容差
    "push_compliant_lateral_stop_tolerance": "auto",
    "push_compliant_force_frame": "direction",    # 力控方向帧
    "push_compliant_axes": None,                  # 限制力控轴（None=全向）
    "push_compliant_zero_ft_sensor": True,        # 力控前清零 FT 传感器基线，使接触力检测准确
    "push_compliant_max_tcp_force": 25.0,         # 接触反力阈值，达到后将退出力控
    "push_compliant_timeout": 15.0,               # 力控安全超时(s)，比默认更宽松避免慢速接近误杀
    "push_compliant_dwell_after_stop": 2.0,       # 推开动作力控接触/到位软停后继续保压的时长(s)

    # Push（推开）阶段追加动作：力控接触推开 + 闭合手爪后，再沿“接触点→箱子中心”的
    # 【水平】方向力控推一段（把瓶子从箱壁/角落拖回箱子中央），然后张开手爪、竖直上抬离开。
    # 力控方向经 robot.get_real_tcp_pose_sim 换算到真实机器人基系（moveL_compliant 按真实基系解释 direction）。
    "push_center_enabled": True,                  # 是否启用“往箱子中心推”力控段
    "push_center_distance": 0.10,                 # 往箱子中心推的距离 (m)
    "push_center_force": 40.0,                    # 力控推力 (N)
    "push_center_vel": 0.06,                      # 力控速度 (m/s)
    "push_center_lateral_tolerance": 0.02,        # 侧向容差
    "push_center_lateral_stop_tolerance": "auto",
    "push_center_zero_ft_sensor": True,           # 力控前清零 FT 传感器基线
    "push_center_max_tcp_force": 30.0,            # 接触反力阈值，达到后软停（视为正常结束）
    "push_center_timeout": 10.0,                  # 力控安全超时 (s)
    "push_center_dwell_after_stop": 2.0,          # 软停后继续保压时长 (s)

    # push 候选坐标系：True=物体局部系（与抓取候选一致，需按检测物体 3D 位姿变换成基系 TCP 后逐个尝试）；
    # False=直接把 pickle 中的位姿当机器人基系固定 TCP 用（旧行为）。ac_pos 量级 ±0.1m 已证实为局部系，默认 True。
    "push_candidates_object_local": True,

    "dry_run": False,
    "use_rrt": True,
    "no_env": False,
    "box_obstacle": True,
    "visualize_failure": False,
    "summary_out": None,

    # 一键抓取流水线：捕获 → YOLO检测 → SAM分割 → 点云补全 → ICP匹配 → 抓取规划 → 轨迹动画
    "run_grasp": True,  # True=双线程自动抓取流水线（dual.py 为 headless 自动脚本）；改为 False 进入 C/D/P/O 交互模式
    # 动画播放一遍后是否自动执行 RTDE 计划（全自动模式）
    "auto_execute": True,

    # 拍照点（独立拍照位）关节角，单位：弧度，6个值。
    "capture_conf_rad": [2.435166835784912, -1.524255597298481, 1.4446890989886683, -1.5398729604533692, -1.5186150709735315, 1.5936657190322876],

    # 抓取起点关节角，单位：弧度，6个值。
    "grasp_start_conf_rad": [1.2610626220703125, -1.5542540115169068, 1.4155376593219202, -1.680861612359518, -1.4960697332965296, 1.3155012130737305],
}


