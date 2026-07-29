"""Generate a UR7e + DH76 bottle pick-and-place path with WRS PickPlacePlanner."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import pickle
import sys
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

import wrs.basis.robot_math as rm  # noqa: E402

from yanjiuyuan.constants import MODEL_DIR, PICK_APPROACH_DEPART_DISTANCE  # noqa: E402
from yanjiuyuan.sync_real_ur7e_to_wrs import DEFAULT_HOME_CONF  # noqa: E402


CUSTOM_BOTTLE_MODEL_PATH = Path(__file__).resolve().parent / "meshes" / "bottle.STL"
OBJECT_MODEL_PATH = CUSTOM_BOTTLE_MODEL_PATH if CUSTOM_BOTTLE_MODEL_PATH.exists() else MODEL_DIR / "bottle.stl"
GRASP_PICKLE_PATH = Path(__file__).resolve().parent / "grasps" / "bottle_dh76_4.pickle" # 抓取pickle文件

PICK_POS = np.array([0.6, 0.0, 0.08], dtype=float)
PICK_RPY_DEG = np.array([0.0, 90, 0], dtype=float)
PLACE_POS = np.array([0.6, -0.12, 0.08], dtype=float)
PLACE_RPY_DEG = np.array([0.0, 90, 0], dtype=float)

IK_SOLVER = 'ikfast'
OPEN_JAW_WIDTH = 0.118
APPROACH_DISTANCE = 0.1    # 抓取路径距离0.1
# PICK_APPROACH_DISTANCE = 0.13


LINEAR_GRANULARITY = 0.015
USE_RRT = True
USE_REASONED_COMMON_GRASPS = True
VALIDATE_COLLISIONS = True
DEBUG_WRS_LOG_LINES = 4
VISUALIZE_FAILURE = False
VISUALIZE_RESULT = True
RESULT_ANIMATION_INTERVAL = 0.04
RESULT_TRAIL_STRIDE = 6
VISUALIZE_FAILURE_GRASP_INDEX = None
VERBOSE = True
PRINT_WRS_SUMMARY = True
DEBUG_IKFAST_FRAME = False


def debug_print(message: object = "") -> None:
    if VERBOSE:
        print(message, flush=True)


def format_vec(values, digits: int = 5) -> list[float]:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def format_jnts_deg(values, digits: int = 2) -> list[float]:
    return np.round(np.degrees(np.asarray(values, dtype=float)), digits).tolist()

def pose_from_pos_rpy(pos: Iterable[float], rpy_deg: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(pos, dtype=float), rm.rotmat_from_euler(*np.radians(np.asarray(rpy_deg, dtype=float)))


def make_robot():
    from wrs.robot_sim.robots.ur7e.ur7e_withouttable_dh76 import UR7EDH76

    robot = UR7EDH76(enable_cc=True, ik_solver=IK_SOLVER)
    robot.goto_given_conf(DEFAULT_HOME_CONF)
    robot.jaw_to(jawwidth=clamp_jaw_width(robot, OPEN_JAW_WIDTH))
    return robot


def debug_ikfast_frame_conversion(robot) -> None:
    if not DEBUG_IKFAST_FRAME or not getattr(robot, "_prefer_ikfast", False):
        return
    debug_print("IKFast frame check:")
    debug_print(f"  arm base world pos: {format_vec(robot.arm.pos)}")
    debug_print(f"  arm base world z: {format_vec(robot.arm.rotmat[:, 2])}")
    debug_print(f"  flange->tcp loc pos: {format_vec(robot.arm.loc_tcp_pos)}")
    debug_print(f"  flange->tcp loc z: {format_vec(robot.arm.loc_tcp_rotmat[:, 2])}")
    robot.backup_state()
    try:
        robot.goto_given_conf(DEFAULT_HOME_CONF)
        tgt_pos = robot.gl_tcp_pos.copy()
        tgt_rotmat = robot.gl_tcp_rotmat.copy()
        rel_pos, rel_rotmat = robot.get_tgt_flange_pose_in_arm_base(tgt_pos, tgt_rotmat)
        solutions = robot.ik_all(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=DEFAULT_HOME_CONF)
        debug_print(f"  home tcp world pos: {format_vec(tgt_pos)}")
        debug_print(f"  home flange in ikfast arm base: pos={format_vec(rel_pos)}, z={format_vec(rel_rotmat[:, 2])}")
        debug_print(f"  roundtrip ikfast solutions: {len(solutions)}")
        if solutions:
            robot.goto_given_conf(solutions[0])
            pos_err = float(np.linalg.norm(robot.gl_tcp_pos - tgt_pos))
            rot_err = float(np.linalg.norm(robot.gl_tcp_rotmat - tgt_rotmat))
            debug_print(
                f"  roundtrip best joints(deg): {format_jnts_deg(solutions[0])}, "
                f"pos_err={pos_err:.3e}, rot_err={rot_err:.3e}"
            )
    finally:
        robot.restore_state()

def clamp_jaw_width(robot, jaw_width: float) -> float:
    low, high = robot.hnd.jaw_range
    return float(np.clip(jaw_width, low, high))


def load_grasps(robot, grasp_pickle_path: Path):
    from wrs import gg

    with grasp_pickle_path.open("rb") as file:
        raw_grasps = pickle.load(file)

    grasps = []
    for index, entry in enumerate(raw_grasps):
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            raise ValueError(f"Unsupported grasp entry {index}: {type(entry).__name__}")
        jaw_width = clamp_jaw_width(robot, float(np.asarray(entry[0], dtype=float).reshape(-1)[0]))
        ac_pos = np.asarray(entry[1], dtype=float)
        ac_rotmat = np.asarray(entry[2], dtype=float)
        if ac_pos.shape != (3,) or ac_rotmat.shape != (3, 3):
            raise ValueError(f"Invalid grasp entry {index}: ac_pos/ac_rotmat shape mismatch")
        grasps.append(gg.Grasp(ee_values=jaw_width, ac_pos=ac_pos, ac_rotmat=ac_rotmat))
    if not grasps:
        raise ValueError(f"No grasps found in {grasp_pickle_path}")
    return grasps


def make_grasp_collection(robot, grasps):
    from wrs import gg

    grasp_collection = gg.GraspCollection(end_effector=robot.end_effector)
    for grasp in grasps:
        grasp_collection.append(grasp)
    return grasp_collection


def get_bottle_cdprim(name="bottle_cdprim", ex_radius=0):
    from panda3d.core import CollisionBox, CollisionNode, NodePath, Point3

    pdcnd = CollisionNode(name + "_cnode")
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
    cdprim = NodePath(name + "_cdprim")
    cdprim.attachNewNode(pdcnd)
    return cdprim


def make_object_model(
    model_path: Path,
    pose: tuple[np.ndarray, np.ndarray],
    name: str,
    alpha: float = 0.7,
    rgb: np.ndarray | None = None,
):
    from wrs import mcm

    bottle_model_path = CUSTOM_BOTTLE_MODEL_PATH if CUSTOM_BOTTLE_MODEL_PATH.exists() else model_path
    obj = mcm.CollisionModel(
        initor=str(bottle_model_path),
        name=name,
        cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
        ex_radius=0.003,
        userdef_cdprim_fn=get_bottle_cdprim,
        rgb=np.array([0.55, 0.82, 1.0]) if rgb is None else np.asarray(rgb, dtype=float),
        alpha=alpha,
    )
    obj.pose = pose
    return obj


def make_table_obstacle():
    from wrs import mcm
    from yanjiuyuan.mech_eye_ur7e_pointcloud_env import OBSTACLE_SPECS, resolve_mesh_path

    for spec in OBSTACLE_SPECS:
        if spec["name"] != "table":
            continue
        mesh_path = resolve_mesh_path(spec["mesh"])
        if mesh_path is None:
            raise FileNotFoundError(f"Table mesh not found: {spec['mesh']}")
        rgba = np.asarray(spec["rgba"], dtype=float)
        table = mcm.CollisionModel(
            initor=str(mesh_path),
            name=spec["name"],
            cdprim_type=mcm.const.CDPrimType.AABB,
            ex_radius=spec["ex_radius"],
            rgb=rgba[:3],
            alpha=float(rgba[3]),
        )
        if spec["pos"] is not None:
            table.pos = spec["pos"]
        return table
    raise RuntimeError("Table obstacle spec not found")


def tcp_pose_from_object_pose(object_pose: tuple[np.ndarray, np.ndarray], grasp) -> tuple[np.ndarray, np.ndarray]:
    obj_pos, obj_rotmat = object_pose
    return obj_pos + obj_rotmat.dot(grasp.ac_pos), obj_rotmat.dot(grasp.ac_rotmat)


def nearest_joint_equivalent(jnt_values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    jnt_values = np.asarray(jnt_values, dtype=float).copy()
    for i, value in enumerate(jnt_values):
        candidates = value + 2.0 * np.pi * np.arange(-2, 3)
        jnt_values[i] = candidates[np.argmin(np.abs(candidates - reference[i]))]
    return jnt_values


def find_planner_seed(robot, grasps, pick_pose, place_pose, obstacle_list):
    debug_print("Searching planner seed from pick/place TCP poses...")
    best_conf = None
    best_jaw_width = None
    best_score = np.inf
    ik_ok_count = 0
    collision_free_count = 0
    robot.backup_state()
    try:
        for object_name, object_pose in (("pick", pick_pose), ("place", place_pose)):
            for grasp_index, grasp in enumerate(grasps):
                tcp_pos, tcp_rotmat = tcp_pose_from_object_pose(object_pose, grasp)
                jnt_values = robot.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rotmat)
                if jnt_values is None:
                    continue
                ik_ok_count += 1
                jnt_values = nearest_joint_equivalent(jnt_values, DEFAULT_HOME_CONF)
                robot.goto_given_conf(jnt_values=jnt_values, ee_values=grasp.ee_values)
                if robot.is_collided(obstacle_list=obstacle_list, toggle_contacts=False):
                    continue
                collision_free_count += 1
                score = float(np.linalg.norm(jnt_values - DEFAULT_HOME_CONF))
                if score < best_score:
                    best_score = score
                    best_conf = jnt_values.copy()
                    best_jaw_width = grasp.ee_values
                    debug_print(
                        f"  seed candidate: {object_name} pickle_grasp_{grasp_index}, "
                        f"score={best_score:.4f}, joints(deg)={format_jnts_deg(best_conf)}"
                    )
    finally:
        robot.restore_state()
    if best_conf is None:
        debug_print(
            f"Seed search: no collision-free IK seed found "
            f"(ik_ok={ik_ok_count}, collision_free={collision_free_count})."
        )
    else:
        debug_print(
            f"Seed search: selected joints(deg)={format_jnts_deg(best_conf)}, "
            f"jaw={best_jaw_width:.5f}, ik_ok={ik_ok_count}, collision_free={collision_free_count}."
        )
    return best_conf, best_jaw_width

def summarize_wrs_log(log_text: str) -> str:
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    if not lines:
        return "no WRS log"
    unique_lines = []
    for line in lines:
        if line not in unique_lines:
            unique_lines.append(line)
    return " | ".join(unique_lines[-DEBUG_WRS_LOG_LINES:])


def add_ikfast_target_detail(robot, detail: dict) -> dict:
    if not hasattr(robot, "get_tgt_flange_pose_in_arm_base"):
        return detail
    try:
        rel_pos, rel_rotmat = robot.get_tgt_flange_pose_in_arm_base(detail["tcp_pos"], detail["tcp_rotmat"])
    except Exception as exc:
        detail["ikfast_frame_error"] = repr(exc)
        return detail
    detail["ikfast_flange_pos"] = np.asarray(rel_pos, dtype=float).copy()
    detail["ikfast_flange_z"] = np.asarray(rel_rotmat, dtype=float)[:, 2].copy()
    return detail

def check_linear_segment(robot, label, start_pos, start_rotmat, goal_pos, goal_rotmat, ee_values, obstacle_list):
    start_pos = np.asarray(start_pos, dtype=float)
    start_rotmat = np.asarray(start_rotmat, dtype=float)
    goal_pos = np.asarray(goal_pos, dtype=float)
    goal_rotmat = np.asarray(goal_rotmat, dtype=float)
    poses = list(
        rm.interplate_pos_rotmat(
            start_pos=start_pos,
            start_rotmat=start_rotmat,
            goal_pos=goal_pos,
            goal_rotmat=goal_rotmat,
            granularity=LINEAR_GRANULARITY,
        )
    )
    seed = None
    robot.backup_state()
    try:
        for step_index, (pos, rotmat) in enumerate(poses):
            pos = np.asarray(pos, dtype=float)
            rotmat = np.asarray(rotmat, dtype=float)
            jnt_values = robot.ik(tgt_pos=pos, tgt_rotmat=rotmat, seed_jnt_values=seed)
            if jnt_values is None:
                detail = {
                    "kind": "ik",
                    "label": label,
                    "step_index": step_index,
                    "step_count": len(poses) - 1,
                    "tcp_pos": pos.copy(),
                    "tcp_rotmat": rotmat.copy(),
                    "segment_start_pos": start_pos.copy(),
                    "segment_start_rotmat": start_rotmat.copy(),
                    "segment_goal_pos": goal_pos.copy(),
                    "segment_goal_rotmat": goal_rotmat.copy(),
                    "ee_values": ee_values,
                    "jnt_values": None,
                }
                add_ikfast_target_detail(robot, detail)
                detail["message"] = format_diagnostic_detail(detail)
                return detail
            robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
            self_collided = bool(robot.is_collided(obstacle_list=None, toggle_contacts=False))
            env_collided = bool(robot.is_collided(obstacle_list=obstacle_list, toggle_contacts=False))
            if self_collided or env_collided:
                detail = {
                    "kind": "collision",
                    "label": label,
                    "collision_type": "self" if self_collided else "environment",
                    "step_index": step_index,
                    "step_count": len(poses) - 1,
                    "tcp_pos": pos.copy(),
                    "tcp_rotmat": rotmat.copy(),
                    "segment_start_pos": start_pos.copy(),
                    "segment_start_rotmat": start_rotmat.copy(),
                    "segment_goal_pos": goal_pos.copy(),
                    "segment_goal_rotmat": goal_rotmat.copy(),
                    "ee_values": ee_values,
                    "jnt_values": np.asarray(jnt_values, dtype=float).copy(),
                }
                add_ikfast_target_detail(robot, detail)
                detail["message"] = format_diagnostic_detail(detail)
                return detail
            seed = jnt_values
    finally:
        robot.restore_state()
    return None


def diagnose_grasp_detail(grasp, pick_pose, place_pose, obstacle_list) -> dict:
    robot = make_robot()
    pick_tcp_pos, pick_tcp_rotmat = tcp_pose_from_object_pose(pick_pose, grasp)
    place_tcp_pos, place_tcp_rotmat = tcp_pose_from_object_pose(place_pose, grasp)
    pre_pick_pos = pick_tcp_pos - rm.unit_vector(pick_tcp_rotmat[:, 2]) * APPROACH_DISTANCE
    post_pick_pos = pick_tcp_pos + rm.const.z_ax * APPROACH_DISTANCE
    pre_place_pos = place_tcp_pos + rm.const.z_ax * APPROACH_DISTANCE
    segments = [
        ("pick approach", pre_pick_pos, pick_tcp_rotmat, pick_tcp_pos, pick_tcp_rotmat, OPEN_JAW_WIDTH),
        ("pick depart", pick_tcp_pos, pick_tcp_rotmat, post_pick_pos, pick_tcp_rotmat, grasp.ee_values),
        ("place approach", pre_place_pos, place_tcp_rotmat, place_tcp_pos, place_tcp_rotmat, grasp.ee_values),
        ("place depart", place_tcp_pos, place_tcp_rotmat, pre_place_pos, place_tcp_rotmat, OPEN_JAW_WIDTH),
    ]
    for segment in segments:
        failure = check_linear_segment(robot, *segment, obstacle_list=obstacle_list)
        if failure is not None:
            return failure
    return {
        "kind": "planner",
        "label": "planner",
        "message": "linear approach/depart segments passed; failure is likely RRT transfer or planner internal state",
        "tcp_pos": pick_tcp_pos.copy(),
        "tcp_rotmat": pick_tcp_rotmat.copy(),
        "segment_start_pos": pre_pick_pos.copy(),
        "segment_start_rotmat": pick_tcp_rotmat.copy(),
        "segment_goal_pos": pick_tcp_pos.copy(),
        "segment_goal_rotmat": pick_tcp_rotmat.copy(),
        "ee_values": grasp.ee_values,
        "jnt_values": None,
    }


def format_diagnostic_detail(detail: dict) -> str:
    label = detail.get("label", "planner")
    tcp_pos = np.round(detail.get("tcp_pos", np.zeros(3)), 5).tolist()
    step_index = detail.get("step_index")
    step_count = detail.get("step_count")
    ikfast_extra = ""
    if "ikfast_flange_pos" in detail:
        ikfast_extra = (
            f", ikfast_flange_pos={format_vec(detail['ikfast_flange_pos'])}, "
            f"ikfast_flange_z={format_vec(detail['ikfast_flange_z'])}"
        )
    elif "ikfast_frame_error" in detail:
        ikfast_extra = f", ikfast_frame_error={detail['ikfast_frame_error']}"
    if detail.get("kind") == "ik":
        return f"{label}: no IK at step {step_index}/{step_count}, tcp={tcp_pos}{ikfast_extra}"
    if detail.get("kind") == "collision":
        jnt_deg = np.round(np.degrees(detail["jnt_values"]), 2).tolist()
        return (
            f"{label}: {detail['collision_type']} collision at step {step_index}/{step_count}, "
            f"tcp={tcp_pos}, joints(deg)={jnt_deg}{ikfast_extra}"
        )
    return detail.get("message", "unknown planner failure")

def diagnose_grasp(grasp, pick_pose, place_pose, obstacle_list) -> str:
    return diagnose_grasp_detail(grasp, pick_pose, place_pose, obstacle_list)["message"]


def format_collision(collision: dict) -> str:
    jnt_deg = np.round(np.degrees(collision["jnt_values"]), 2).tolist()
    return f"frame={collision['index']} type={collision['type']} joints(deg)={jnt_deg}"


def attach_debug_model(model, base) -> None:
    try:
        model.copy().attach_to(base)
    except Exception:
        model.attach_to(base)


def select_failure_debug_record(records: list[dict]) -> dict | None:
    if not records:
        return None
    if VISUALIZE_FAILURE_GRASP_INDEX is not None:
        for record in records:
            if record["grasp_index"] == VISUALIZE_FAILURE_GRASP_INDEX:
                return record
    return records[-1]


def visualize_failure(grasp_index: int, grasp, pick_pose, place_pose, obstacle_list, detail: dict) -> None:
    from wrs import mgm, wd

    fail_pos = np.asarray(detail.get("tcp_pos", tcp_pose_from_object_pose(pick_pose, grasp)[0]), dtype=float)
    fail_rotmat = np.asarray(detail.get("tcp_rotmat", tcp_pose_from_object_pose(pick_pose, grasp)[1]), dtype=float)
    start_pos = np.asarray(detail.get("segment_start_pos", fail_pos), dtype=float)
    start_rotmat = np.asarray(detail.get("segment_start_rotmat", fail_rotmat), dtype=float)
    goal_pos = np.asarray(detail.get("segment_goal_pos", fail_pos), dtype=float)
    goal_rotmat = np.asarray(detail.get("segment_goal_rotmat", fail_rotmat), dtype=float)

    base = wd.World(cam_pos=[2.0, -1.6, 1.2], lookat_pos=[0.4, -0.05, 0.25], w=1280, h=720)
    mgm.gen_frame(ax_length=0.25, ax_radius=0.004).attach_to(base)

    for obstacle in obstacle_list:
        attach_debug_model(obstacle, base)

    make_object_model(
        OBJECT_MODEL_PATH,
        pick_pose,
        name="debug_pick_object",
        alpha=0.45,
        rgb=np.array([0.55, 0.82, 1.0]),
    ).attach_to(base)
    make_object_model(
        OBJECT_MODEL_PATH,
        place_pose,
        name="debug_place_object",
        alpha=0.25,
        rgb=np.array([0.2, 0.9, 0.45]),
    ).attach_to(base)

    if np.linalg.norm(goal_pos - start_pos) > 1e-8:
        mgm.gen_arrow(
            spos=start_pos,
            epos=goal_pos,
            rgb=np.array([1.0, 0.05, 0.05]),
            alpha=1.0,
            stick_radius=0.005,
        ).attach_to(base)
    mgm.gen_sphere(pos=start_pos, radius=0.012, rgb=np.array([0.0, 0.35, 1.0]), alpha=1.0).attach_to(base)
    mgm.gen_sphere(pos=goal_pos, radius=0.012, rgb=np.array([0.0, 0.8, 0.2]), alpha=1.0).attach_to(base)
    mgm.gen_sphere(pos=fail_pos, radius=0.014, rgb=np.array([1.0, 0.0, 0.0]), alpha=1.0).attach_to(base)
    mgm.gen_frame(pos=start_pos, rotmat=start_rotmat, ax_length=0.08, ax_radius=0.002).attach_to(base)
    mgm.gen_frame(pos=goal_pos, rotmat=goal_rotmat, ax_length=0.08, ax_radius=0.002).attach_to(base)
    mgm.gen_frame(pos=fail_pos, rotmat=fail_rotmat, ax_length=0.10, ax_radius=0.0025).attach_to(base)

    debug_robot = make_robot()
    jnt_values = detail.get("jnt_values")
    if jnt_values is not None:
        debug_robot.goto_given_conf(jnt_values=jnt_values, ee_values=detail.get("ee_values", grasp.ee_values))
    debug_robot.gen_meshmodel(
        alpha=0.55,
        toggle_tcp_frame=True,
        toggle_flange_frame=True,
        toggle_jnt_frames=False,
    ).attach_to(base)

    try:
        debug_robot.end_effector.grip_at_by_pose(
            jaw_center_pos=fail_pos,
            jaw_center_rotmat=fail_rotmat,
            jaw_width=float(detail.get("ee_values", grasp.ee_values)),
        )
        debug_robot.end_effector.gen_meshmodel(rgb=rm.const.magenta, alpha=0.35, toggle_tcp_frame=True).attach_to(base)
    except Exception as exc:
        debug_print(f"Could not draw debug gripper ghost: {exc}")

    debug_print("Opening WRS debug scene for planner failure.")
    debug_print(f"Debug grasp: pickle_grasp_{grasp_index}")
    debug_print(f"Failure: {detail.get('message', 'unknown planner failure')}")
    debug_print(f"Approach start TCP: {format_vec(start_pos)}")
    debug_print(f"Approach target TCP: {format_vec(goal_pos)}")
    debug_print(f"First failed TCP: {format_vec(fail_pos)}")
    debug_print("Blue sphere: approach start; green sphere: target; red sphere: first failed TCP.")
    debug_print("Close the WRS window to continue to the RuntimeError summary.")
    base.run()

def diagnose_failure_candidates(grasps, grasp_indices, pick_pose, place_pose, obstacle_list) -> list[dict]:
    if not VISUALIZE_FAILURE:
        return []
    pairs = list(zip(grasp_indices, grasps))
    if VISUALIZE_FAILURE_GRASP_INDEX is not None:
        selected = [pair for pair in pairs if pair[0] == VISUALIZE_FAILURE_GRASP_INDEX]
        if selected:
            pairs = selected
    pairs = pairs[:1]
    records = []
    if not pairs:
        return records
    debug_print(f"Collecting failure visualization details for pickle_grasp_{pairs[0][0]}...")
    for grasp_index, grasp in pairs:
        detail = diagnose_grasp_detail(grasp, pick_pose, place_pose, obstacle_list)
        records.append({"grasp_index": grasp_index, "grasp": grasp, "detail": detail})
        debug_print(f"  pickle_grasp_{grasp_index}: {detail['message']}")
    return records


def gen_pick_place_path(robot, obj_cmodel, grasps, pick_pose, place_pose, obstacle_list):
    from wrs import ppp

    debug_print("Starting PickPlacePlanner.gen_pick_and_place...")
    debug_print(f"  grasps={len(grasps)}, use_rrt={USE_RRT}, validate_collisions={VALIDATE_COLLISIONS}")
    debug_print(f"  reason_common_grasps={USE_REASONED_COMMON_GRASPS}")
    debug_print(f"  approach/depart distance={APPROACH_DISTANCE:.3f} m, linear_granularity={LINEAR_GRANULARITY:.3f} m")
    debug_print("  directions: pick approach=grasp TCP +Z, pick depart=+Z, place approach=-Z, place depart=+Z")

    planner = ppp.PickPlacePlanner(robot)
    candidate_indices = list(range(len(grasps)))
    candidate_grasps = list(grasps)

    if USE_REASONED_COMMON_GRASPS:
        debug_print("Reasoning common grasps for pick and place poses...")
        full_grasp_collection = make_grasp_collection(robot, grasps)
        common_gid_list = list(
            planner.reason_common_gids(
                grasp_collection=full_grasp_collection,
                goal_pose_list=[pick_pose, place_pose],
                obstacle_list=obstacle_list,
                toggle_dbg=False,
            )
        )
        candidate_indices = common_gid_list
        candidate_grasps = [grasps[index] for index in candidate_indices]
        debug_print(f"  reasoned common grasps: {len(candidate_indices)}/{len(grasps)}")
        if candidate_indices:
            preview = candidate_indices[:20]
            suffix = "..." if len(candidate_indices) > len(preview) else ""
            debug_print(f"  common grasp indices: {preview}{suffix}")
        if not candidate_grasps:
            debug_print("No common grasps survived pick/place reasoning.")
            diagnostics = diagnose_failure_candidates(grasps, list(range(len(grasps))), pick_pose, place_pose, obstacle_list)
            debug_record = select_failure_debug_record(diagnostics)
            if VISUALIZE_FAILURE and debug_record is not None:
                visualize_failure(
                    debug_record["grasp_index"],
                    debug_record["grasp"],
                    pick_pose,
                    place_pose,
                    obstacle_list,
                    debug_record["detail"],
                )
            joined = "\n  ".join(
                f"pickle_grasp_{record['grasp_index']}: {record['detail']['message']}" for record in diagnostics
            )
            message = "PickPlacePlanner failed: no reasoned common grasps."
            if joined:
                message = f"{message}\n  {joined}"
            raise RuntimeError(message)

    grasp_collection = make_grasp_collection(robot, candidate_grasps)
    robot.goto_given_conf(DEFAULT_HOME_CONF, ee_values=clamp_jaw_width(robot, OPEN_JAW_WIDTH))

    mot_data = planner.gen_pick_and_place(
        obj_cmodel=obj_cmodel,
        grasp_collection=grasp_collection,
        goal_pose_list=[place_pose],
        start_jnt_values=DEFAULT_HOME_CONF,
        end_jnt_values=None,
        pick_approach_jaw_width=clamp_jaw_width(robot, OPEN_JAW_WIDTH),
        pick_approach_direction=None,  # PickPlacePlanner: None means grasp/TCP +Z ("handz").
        pick_approach_distance=APPROACH_DISTANCE,
        pick_depart_direction=rm.const.z_ax,
        pick_depart_distance=PICK_APPROACH_DEPART_DISTANCE,
        place_approach_direction_list=[-rm.const.z_ax],
        place_approach_distance_list=[APPROACH_DISTANCE],
        place_depart_direction_list=[rm.const.z_ax],
        place_depart_distance_list=[APPROACH_DISTANCE],
        place_depart_jaw_width=clamp_jaw_width(robot, OPEN_JAW_WIDTH),
        linear_granularity=LINEAR_GRANULARITY,
        obstacle_list=obstacle_list,
        use_rrt=USE_RRT,
        reason_grasps=not USE_REASONED_COMMON_GRASPS,
        toggle_dbg=False,
    )
    if mot_data is None:
        debug_print("PickPlacePlanner returned None.")
        diagnostics = diagnose_failure_candidates(candidate_grasps, candidate_indices, pick_pose, place_pose, obstacle_list)
        debug_record = select_failure_debug_record(diagnostics)
        if VISUALIZE_FAILURE and debug_record is not None:
            debug_print(
                f"Opening failure visualization for pickle_grasp_{debug_record['grasp_index']}: "
                f"{debug_record['detail'].get('message', 'unknown planner failure')}"
            )
            visualize_failure(
                debug_record["grasp_index"],
                debug_record["grasp"],
                pick_pose,
                place_pose,
                obstacle_list,
                debug_record["detail"],
            )
        joined = "\n  ".join(
            f"pickle_grasp_{record['grasp_index']}: {record['detail']['message']}" for record in diagnostics
        )


    frame_count = len(mot_data.jv_list) if hasattr(mot_data, "jv_list") else len(mot_data)
    debug_print(f"PickPlacePlanner produced {frame_count} frame(s); validating collision...")
    collisions = validate_motion(make_robot(), mot_data, obstacle_list) if VALIDATE_COLLISIONS else []
    if collisions:
        first_collision = format_collision(collisions[0])
        debug_print(f"Collision validation failed: {len(collisions)} frame(s); first={first_collision}")
        raise RuntimeError(
            f"Collision validation failed after PickPlacePlanner succeeded: {len(collisions)} frame(s).\n"
            f"  first={first_collision}"
        )

    selected_grasp_index = candidate_indices[0] if len(candidate_indices) == 1 else None
    debug_print("PickPlacePlanner succeeded.")
    if selected_grasp_index is None:
        debug_print("  selected grasp index is not exposed by PickPlacePlanner in reasoned-common mode.")
    else:
        debug_print(f"  selected grasp: pickle_grasp_{selected_grasp_index}")
    return selected_grasp_index, mot_data

def visualize_result(mot_data, pick_pose, place_pose, obstacle_list) -> None:
    from wrs import mgm, wd

    if len(mot_data) == 0:
        debug_print("No motion frames to visualize.")
        return

    base = wd.World(cam_pos=[2.0, -1.6, 1.2], lookat_pos=[0.58, -0.12, 0.28], w=1280, h=720)
    mgm.gen_frame(ax_length=0.25, ax_radius=0.004).attach_to(base)

    for obstacle in obstacle_list:
        attach_debug_model(obstacle, base)

    make_object_model(
        OBJECT_MODEL_PATH,
        pick_pose,
        name="result_pick_object",
        alpha=0.25,
        rgb=np.array([0.55, 0.82, 1.0]),
    ).attach_to(base)
    make_object_model(
        OBJECT_MODEL_PATH,
        place_pose,
        name="result_place_object",
        alpha=0.25,
        rgb=np.array([0.2, 0.9, 0.45]),
    ).attach_to(base)

    marker_robot = make_robot()
    marker_robot.backup_state()
    try:
        for index in range(0, len(mot_data), max(1, RESULT_TRAIL_STRIDE)):
            jnt_values, ee_values, _obj_pose, _mesh = mot_data[index]
            marker_robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
            mgm.gen_sphere(
                pos=marker_robot.gl_tcp_pos,
                radius=0.006,
                rgb=np.array([1.0, 0.35, 0.05]),
                alpha=0.75,
            ).attach_to(base)
    finally:
        marker_robot.restore_state()

    class AnimationData:
        def __init__(self, motion_data):
            self.counter = 0
            self.motion_data = motion_data
            self.robot = make_robot()
            self.mesh_model = None
            self.obj_model = None

    anime_data = AnimationData(mot_data)

    def update(data, task):
        if data.mesh_model is not None:
            data.mesh_model.detach()
            data.mesh_model = None
        if data.obj_model is not None:
            data.obj_model.detach()
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
            data.mesh_model.attach_to(base)

        if obj_pose is not None and cached_mesh is None:
            data.obj_model = make_object_model(
                OBJECT_MODEL_PATH,
                (np.asarray(obj_pose[0], dtype=float), np.asarray(obj_pose[1], dtype=float)),
                name="animated_held_object",
                alpha=0.65,
                rgb=np.array([0.95, 0.72, 0.18]),
            )
            data.obj_model.attach_to(base)

        data.counter += 1
        return task.again
    debug_print("Opening WRS result visualization.")
    debug_print(f"Result animation frames: {len(mot_data)}")
    debug_print("Close the WRS window to finish the script.")
    base.taskMgr.doMethodLater(
        RESULT_ANIMATION_INTERVAL,
        update,
        "sim_pick_place_result_animation",
        extraArgs=[anime_data],
        appendTask=True,
    )
    base.run()

def validate_motion(robot, mot_data, obstacle_list: list[object]) -> list[dict]:
    collisions = []
    robot.backup_state()
    try:
        for index in range(len(mot_data)):
            jnt_values, ee_values, _obj_pose, _mesh = mot_data[index]
            robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
            self_collided = bool(robot.is_collided(obstacle_list=None, toggle_contacts=False))
            env_collided = bool(robot.is_collided(obstacle_list=obstacle_list, toggle_contacts=False))
            if self_collided or env_collided:
                collisions.append(
                    {
                        "index": index,
                        "type": "self" if self_collided else "environment",
                        "jnt_values": np.asarray(jnt_values, dtype=float).copy(),
                    }
                )
    finally:
        robot.restore_state()
    return collisions


def print_summary(mot_data, grasp_count: int, selected_grasp_index: int, pick_pose, place_pose, obstacle_names: list[str]) -> None:
    debug_print("PickPlacePlanner path ready.")
    debug_print(f"Loaded grasps: {grasp_count}")
    if selected_grasp_index is None:
        debug_print("Selected grasp: unknown (PickPlacePlanner does not expose the final gid)")
    else:
        debug_print(f"Selected grasp: pickle_grasp_{selected_grasp_index}")
    debug_print(f"Frames: {len(mot_data.jv_list)}")
    if len(mot_data.jv_list) > 1:
        max_step = max(
            float(np.max(np.abs(np.asarray(mot_data.jv_list[i + 1]) - np.asarray(mot_data.jv_list[i]))))
            for i in range(len(mot_data.jv_list) - 1)
        )
        debug_print(f"Max adjacent joint step(deg): {np.degrees(max_step):.3f}")
    debug_print(f"Collision obstacles: {obstacle_names}")
    debug_print(f"Pick object pos(m): {format_vec(pick_pose[0], digits=6)}")
    debug_print(f"Place object pos(m): {format_vec(place_pose[0], digits=6)}")
    debug_print(f"Start conf(deg): {format_jnts_deg(mot_data.jv_list[0], digits=3)}")
    debug_print(f"End conf(deg): {format_jnts_deg(mot_data.jv_list[-1], digits=3)}")

def main():
    debug_print("=== sim_pick_and_place: PickPlacePlanner debug run ===")
    debug_print(f"Object model: {OBJECT_MODEL_PATH}")
    debug_print(f"Grasp pickle: {GRASP_PICKLE_PATH}")
    debug_print(f"Default home conf(deg): {format_jnts_deg(DEFAULT_HOME_CONF)}")
    debug_print(f"Open jaw width: {OPEN_JAW_WIDTH:.5f} m")
    if not OBJECT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Object model not found: {OBJECT_MODEL_PATH}")
    if not GRASP_PICKLE_PATH.exists():
        raise FileNotFoundError(f"Grasp pickle not found: {GRASP_PICKLE_PATH}")

    pick_pose = pose_from_pos_rpy(PICK_POS, PICK_RPY_DEG)
    place_pose = pose_from_pos_rpy(PLACE_POS, PLACE_RPY_DEG)
    debug_print(f"Pick pos: {format_vec(pick_pose[0])}, rpy(deg): {format_vec(PICK_RPY_DEG, digits=3)}")
    debug_print(f"Place pos: {format_vec(place_pose[0])}, rpy(deg): {format_vec(PLACE_RPY_DEG, digits=3)}")

    robot = make_robot()
    debug_print(f"Robot: UR7EDH76(enable_cc=True, ik_solver={IK_SOLVER!r})")
    debug_ikfast_frame_conversion(robot)
    grasps = load_grasps(robot, GRASP_PICKLE_PATH)
    debug_print(f"Loaded grasps: {len(grasps)}")
    table = make_table_obstacle()
    obstacle_list = [table]
    debug_print(f"Collision obstacles: {[obstacle.name for obstacle in obstacle_list]}")

    obj_cmodel = make_object_model(OBJECT_MODEL_PATH, pick_pose, name="pick_place_object")
    selected_grasp_index, mot_data = gen_pick_place_path(robot, obj_cmodel, grasps, pick_pose, place_pose, obstacle_list)
    print_summary(mot_data, len(grasps), selected_grasp_index, pick_pose, place_pose, [table.name])

    if VALIDATE_COLLISIONS:
        debug_print("Collision check: passed (self/environment).")

    if VISUALIZE_RESULT:
        visualize_result(mot_data, pick_pose, place_pose, obstacle_list)

    return mot_data

if __name__ == "__main__":
    main()
