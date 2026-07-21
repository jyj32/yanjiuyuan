"""Offline bottle push-only planner driven by action_sequence_config.json.

This is the push-only companion to sim_bottle_pick_place_from_box_object_icp.py.
It reuses the saved-capture/ICP pipeline, then plans only the JSON-configured
push path for action sequences 7 and 9 by default.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from yanjiuyuan import pick_place_rtde_utils as rtde_utils  # noqa: E402
from yanjiuyuan import real_bottle_pick_place_interactive2 as real_push  # noqa: E402
from yanjiuyuan import sim_bottle_pick_place_from_box_object_icp as sim_pipeline  # noqa: E402
from yanjiuyuan import sim_pick_and_place as sim_pick  # noqa: E402


DEFAULT_ACTION_SEQUENCE_CONFIG_PATH = Path(__file__).resolve().parent / "action_sequence_config.json"
DEFAULT_PUSH_ACTION_SEQUENCES = (7, 9)


@dataclass
class PushPlanningResult:
    action_sequence: int
    selected_grasp_index: Optional[int]
    mot_data: Any
    pick_pose: tuple[np.ndarray, np.ndarray]
    place_pose: tuple[np.ndarray, np.ndarray]
    object_model_path: Path
    obstacle_names: list[str]
    grasp_pickle_path: Path
    push_settings: dict[str, Any]
    rtde_plan: Optional[rtde_utils.RtdeExecutionPlan] = None
    rtde_plan_path: Optional[Path] = None


@dataclass
class RtdeObjectPose:
    pos: np.ndarray
    rotmat: np.ndarray


def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def parse_action_sequence_list(value: str) -> list[int]:
    sequences: list[int] = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        sequences.append(int(item))
    if not sequences:
        raise argparse.ArgumentTypeError("expected at least one action sequence id")
    return sequences


def parse_args() -> argparse.Namespace:
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument(
        "--action-sequence-config",
        type=Path,
        default=DEFAULT_ACTION_SEQUENCE_CONFIG_PATH,
        help="JSON file containing push settings for action sequences 7 and 9.",
    )
    wrapper_parser.add_argument(
        "--push-action-sequences",
        default=",".join(str(item) for item in DEFAULT_PUSH_ACTION_SEQUENCES),
        help="Comma-separated action sequence ids to plan. Default: 7,9.",
    )
    wrapper_args, remaining_argv = wrapper_parser.parse_known_args()
    requested_no_interactive = "--no-interactive" in remaining_argv

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining_argv]
        args = sim_pipeline.parse_args()
    finally:
        sys.argv = original_argv

    sim_pipeline.normalize_paths(args)
    args.action_sequence_config = resolve_path(wrapper_args.action_sequence_config)
    args.push_action_sequences = parse_action_sequence_list(wrapper_args.push_action_sequences)
    if not bool(args.dry_run) and not requested_no_interactive:
        args.interactive = True
    return args


def make_action_args(args: argparse.Namespace, action_sequence: int) -> argparse.Namespace:
    action_args = copy.copy(args)
    action_args.action_sequence = int(action_sequence)
    action_args.action_sequence_config = args.action_sequence_config
    return action_args


def configure_push_planning(action_args: argparse.Namespace, object_model_path: Path) -> None:
    sim_pipeline.configure_pick_place(action_args, object_model_path)
    sim_pick.GRASP_PICKLE_PATH = real_push.action_sequence_grasp_pickle_path(action_args)
    sim_pick.VALIDATE_COLLISIONS = False


def load_push_settings(action_args: argparse.Namespace) -> dict[str, Any]:
    push_settings = real_push.action_sequence_push_settings(action_args)
    if push_settings is None:
        raise RuntimeError(
            f"Action sequence {action_args.action_sequence} has no push.direction in "
            f"{action_args.action_sequence_config}."
        )
    return push_settings


def reason_candidate_grasps(robot, planner, grasps, pick_pose, obstacle_list, action_args: argparse.Namespace) -> list[int]:
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
        if not candidate_indices:
            raise RuntimeError("Push planner failed: no pick-feasible grasps.")
        preview = candidate_indices[:20]
        suffix = "..." if len(candidate_indices) > len(preview) else ""
        sim_pick.debug_print(f"  pick grasp indices: {preview}{suffix}")
    return real_push.filter_candidate_grasps_for_action_sequence(action_args, candidate_indices, len(grasps))


def gen_push_only_path(
    robot,
    obj_cmodel,
    grasps,
    pick_pose: tuple[np.ndarray, np.ndarray],
    obstacle_list: list[object],
    action_args: argparse.Namespace,
) -> tuple[int, Any, dict[str, Any]]:
    from wrs import ppp

    push_settings = load_push_settings(action_args)
    sim_pick.debug_print(f"Starting push-only planner for action sequence {action_args.action_sequence}...")
    sim_pick.debug_print(
        "  push settings: "
        f"direction={sim_pick.format_vec(push_settings['direction'], digits=4)}, "
        f"distance={push_settings['distance']:.4f} m, "
        f"lift_distance={push_settings['lift_distance']:.4f} m, "
        f"{real_push.push_clearance_text(push_settings)}"
    )

    planner = ppp.PickPlacePlanner(robot)
    candidate_indices = reason_candidate_grasps(robot, planner, grasps, pick_pose, obstacle_list, action_args)
    for grasp_index in candidate_indices:
        grasp = grasps[grasp_index]
        grasp_width = real_push.grasp_jaw_width(grasp)
        pick_approach_jaw_width = sim_pick.clamp_jaw_width(robot, grasp_width)
        robot.goto_given_conf(sim_pick.DEFAULT_HOME_CONF, ee_values=pick_approach_jaw_width)
        sim_pick.debug_print(
            f"  pickle_grasp_{grasp_index}: grasp jaw={grasp_width:.5f} m, "
            f"push pre-approach close jaw={pick_approach_jaw_width:.5f} m (grasp jaw)"
        )
        mot_data, push_error = real_push.gen_pick_push_path_for_grasp(
            planner=planner,
            robot=robot,
            obj_cmodel=obj_cmodel.copy(),
            grasp=grasp,
            pick_pose=pick_pose,
            pick_approach_jaw_width=pick_approach_jaw_width,
            push_settings=push_settings,
            obstacle_list=obstacle_list,
        )
        if mot_data is None:
            sim_pick.debug_print(f"  pickle_grasp_{grasp_index}: push path failed: {push_error}")
            continue
        frame_count = len(mot_data.jv_list)
        sim_pick.debug_print(f"Push-only planner succeeded: {frame_count} frame(s), selected pickle_grasp_{grasp_index}.")
        return grasp_index, mot_data, real_push.serializable_push_settings(push_settings)

    raise RuntimeError(
        f"Push planner failed: action sequence {action_args.action_sequence} produced no valid push path."
    )


def default_rtde_plan_path(base_dir: Path, action_sequence: int, args: argparse.Namespace) -> Path:
    if args.rtde_plan_out is None:
        return base_dir / f"push_rtde_plan_action_sequence_{action_sequence}.json"
    if len(args.push_action_sequences) == 1:
        return args.rtde_plan_out
    stem = args.rtde_plan_out.stem
    suffix = args.rtde_plan_out.suffix or ".json"
    return args.rtde_plan_out.with_name(f"{stem}_action_sequence_{action_sequence}{suffix}")


def build_and_save_rtde_plan(
    args: argparse.Namespace,
    planning: PushPlanningResult,
    default_dir: Path,
) -> tuple[rtde_utils.RtdeExecutionPlan, Path]:
    robot = sim_pick.make_robot()
    grasps = sim_pick.load_grasps(robot, planning.grasp_pickle_path)
    grasp_index = planning.selected_grasp_index
    if grasp_index is None or grasp_index < 0 or grasp_index >= len(grasps):
        raise RuntimeError(f"Invalid selected grasp index for RTDE plan: {grasp_index}")
    action_args = make_action_args(args, planning.action_sequence)
    grasp = real_push.decorate_grasp_for_rtde(robot, grasps[grasp_index], grasp_index, action_args)
    rtde_plan = rtde_utils.build_pick_place_rtde_plan(
        robot=robot,
        mot_data=planning.mot_data,
        pick_pose=RtdeObjectPose(pos=planning.pick_pose[0], rotmat=planning.pick_pose[1]),
        place_pose=RtdeObjectPose(pos=planning.place_pose[0], rotmat=planning.place_pose[1]),
        grasp=grasp,
        planner_name=f"WRS sim push-only planner action sequence {planning.action_sequence}",
    )
    rtde_plan.metadata["path_type"] = "push"
    rtde_plan.metadata["action_sequence"] = int(planning.action_sequence)
    rtde_plan.metadata["push_settings"] = planning.push_settings
    rtde_plan_path = default_rtde_plan_path(default_dir, planning.action_sequence, args)
    rtde_utils.save_rtde_execution_plan(rtde_plan, rtde_plan_path)
    sim_pick.debug_print(f"Saved push RTDE execution plan: {rtde_plan_path}")
    return rtde_plan, rtde_plan_path


def plan_one_action_sequence(
    args: argparse.Namespace,
    icp: sim_pipeline.ObjectIcpResult,
    action_sequence: int,
) -> PushPlanningResult:
    action_args = make_action_args(args, action_sequence)
    object_model_path = action_args.object_model if action_args.object_model is not None else icp.bottle_model_path
    object_model_path = object_model_path.resolve()
    if not object_model_path.exists():
        raise FileNotFoundError(f"Object model not found: {object_model_path}")

    bottle_homomat = sim_pipeline.load_homomat(icp.bottle_transform_path, "bottle ICP")
    box_homomat = sim_pipeline.load_homomat(icp.box_transform_path, "box")
    pick_pose = sim_pipeline.homomat_to_pose(bottle_homomat)
    place_pose, _place_pos_source, _place_rot_source = sim_pipeline.resolve_place_pose(action_args, pick_pose)

    configure_push_planning(action_args, object_model_path)
    if not sim_pick.GRASP_PICKLE_PATH.exists():
        raise FileNotFoundError(f"Grasp pickle not found: {sim_pick.GRASP_PICKLE_PATH}")

    sim_pick.debug_print("=== sim_bottle_push_from_box_object_icp: push-only planner run ===")
    sim_pick.debug_print(f"Action sequence: {action_sequence}")
    sim_pick.debug_print(f"Action sequence config: {action_args.action_sequence_config}")
    sim_pick.debug_print(f"Object model: {object_model_path}")
    sim_pick.debug_print(f"Grasp pickle: {sim_pick.GRASP_PICKLE_PATH}")
    sim_pick.debug_print(f"Pick pose from ICP: pos={sim_pick.format_vec(pick_pose[0], digits=6)}")

    robot = sim_pick.make_robot()
    sim_pick.debug_ikfast_frame_conversion(robot)
    grasps = sim_pick.load_grasps(robot, sim_pick.GRASP_PICKLE_PATH)
    obstacle_list, display_obstacle_list = sim_pipeline.build_obstacle_lists(action_args, box_homomat)
    obstacle_names = [getattr(obstacle, "name", type(obstacle).__name__) for obstacle in obstacle_list]
    sim_pick.debug_print(f"Collision obstacles: {obstacle_names}")
    sim_pipeline.precheck_start_collision(robot, obstacle_list)

    obj_cmodel = sim_pick.make_object_model(object_model_path, pick_pose, name=f"push_action_sequence_{action_sequence}")
    selected_grasp_index, mot_data, push_settings = gen_push_only_path(
        robot,
        obj_cmodel,
        grasps,
        pick_pose,
        obstacle_list,
        action_args,
    )
    sim_pick.debug_print(f"Frames: {len(mot_data.jv_list)}")
    sim_pick.debug_print(f"Start conf(deg): {sim_pick.format_jnts_deg(mot_data.jv_list[0], digits=3)}")
    sim_pick.debug_print(f"End conf(deg): {sim_pick.format_jnts_deg(mot_data.jv_list[-1], digits=3)}")

    planning = PushPlanningResult(
        action_sequence=int(action_sequence),
        selected_grasp_index=selected_grasp_index,
        mot_data=mot_data,
        pick_pose=pick_pose,
        place_pose=place_pose,
        object_model_path=object_model_path,
        obstacle_names=obstacle_names,
        grasp_pickle_path=sim_pick.GRASP_PICKLE_PATH,
        push_settings=push_settings,
    )
    planning.rtde_plan, planning.rtde_plan_path = build_and_save_rtde_plan(args, planning, icp.output_dir)

    if sim_pick.VISUALIZE_RESULT:
        sim_pick.visualize_result(mot_data, pick_pose, place_pose, display_obstacle_list)
    return planning


def write_summary(
    args: argparse.Namespace,
    icp: sim_pipeline.ObjectIcpResult,
    planning_results: list[PushPlanningResult],
) -> Path:
    summary_path = args.summary_out if args.summary_out is not None else icp.output_dir / "sim_bottle_push_summary.json"
    summary = {
        "object_summary_path": str(icp.summary_path),
        "bottle_transform_path": str(icp.bottle_transform_path),
        "box_transform_path": str(icp.box_transform_path),
        "bottle_model_path": str(icp.bottle_model_path),
        "action_sequence_config": str(args.action_sequence_config),
        "push_action_sequences": [int(item) for item in args.push_action_sequences],
        "planned": bool(planning_results),
        "results": [],
    }
    for planning in planning_results:
        summary["results"].append(
            {
                "action_sequence": int(planning.action_sequence),
                "object_model_path": str(planning.object_model_path),
                "selected_grasp_index": planning.selected_grasp_index,
                "grasp_pickle_path": str(planning.grasp_pickle_path),
                "path_type": "push",
                "push_settings": planning.push_settings,
                "frame_count": len(planning.mot_data.jv_list),
                "collision_obstacle_names": planning.obstacle_names,
                "pick_pos": np.asarray(planning.pick_pose[0], dtype=float).tolist(),
                "pick_rotmat": np.asarray(planning.pick_pose[1], dtype=float).tolist(),
                "rtde_plan_path": None if planning.rtde_plan_path is None else str(planning.rtde_plan_path),
                "rtde_segment_count": None if planning.rtde_plan is None else len(planning.rtde_plan.segments),
                "visualized_result": not bool(args.dry_run),
            }
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


class InteractiveBottlePushApp(sim_pipeline.InteractiveBottlePickPlaceApp):
    def __init__(
        self,
        args: argparse.Namespace,
        ctx: Optional[Any] = None,
        initial_icp: Optional[sim_pipeline.ObjectIcpResult] = None,
    ):
        super().__init__(args=args, ctx=ctx, initial_icp=initial_icp)
        self.push_planning_results: list[PushPlanningResult] = []
        self.animation_task_name = "sim_bottle_push_interactive_animation"
        self.status_text.setText("Offline push scene loaded. Press D for ICP, then P to plan push actions 7/9.")
        print("Offline push viewer is ready. Press D for segmentation/template/ICP, P to plan push actions 7/9.")

    def attach_push_plan_results(self, planning_results: list[PushPlanningResult]) -> None:
        if not planning_results:
            print("[sim_push_pipeline] No push motion frames to visualize.")
            return
        colors = (
            np.array([1.0, 0.35, 0.05]),
            np.array([0.1, 0.42, 1.0]),
            np.array([0.15, 0.7, 0.28]),
            np.array([0.85, 0.2, 0.7]),
        )
        marker_robot = sim_pick.make_robot()
        marker_robot.backup_state()
        try:
            for result_index, planning in enumerate(planning_results):
                mot_data = planning.mot_data
                rgb = colors[result_index % len(colors)]
                for index in range(0, len(mot_data), max(1, sim_pick.RESULT_TRAIL_STRIDE)):
                    jnt_values, ee_values, _obj_pose, _mesh = mot_data[index]
                    marker_robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
                    marker = self.mgm.gen_sphere(
                        pos=marker_robot.gl_tcp_pos,
                        radius=0.006,
                        rgb=rgb,
                        alpha=0.78,
                    )
                    marker.attach_to(self.base)
                    self.plan_models.append(marker)
        finally:
            marker_robot.restore_state()
        self.start_motion_animation(planning_results[0])

    def run_plan(self) -> None:
        if self.running:
            print("[sim_push_pipeline] Another operation is already running; ignoring P key.")
            return
        if self.args.skip_plan:
            self.status_text.setText("Planning is disabled by --skip-plan.")
            return
        if self.icp_result is None:
            self.status_text.setText("No estimated start pose yet. Press D first.")
            return
        self.running = True
        sequence_text = ",".join(str(item) for item in self.args.push_action_sequences)
        self.status_text.setText(f"Planning push paths for action sequences {sequence_text}...")
        self.clear_plan_models()
        try:
            plan_args = copy.copy(self.args)
            plan_args.dry_run = True
            planning_results = [
                plan_one_action_sequence(plan_args, self.icp_result, sequence)
                for sequence in self.args.push_action_sequences
            ]
            self.push_planning_results = planning_results
            self.planning_result = planning_results[0] if planning_results else None
            self.attach_push_plan_results(planning_results)
            summary_path = write_summary(self.args, self.icp_result, planning_results)
            rtde_paths = [str(item.rtde_plan_path) for item in planning_results if item.rtde_plan_path is not None]
            rtde_text = "" if not rtde_paths else " RTDE=" + "; ".join(rtde_paths)
            frame_text = ", ".join(
                f"{item.action_sequence}:{len(item.mot_data.jv_list)}" for item in planning_results
            )
            self.status_text.setText(
                f"Push planning done for {sequence_text}: frames {frame_text}. Offline animation is running."
            )
            print(f"[sim_push_pipeline] push summary after planning: {summary_path}{rtde_text}")
        except Exception as exc:
            self.status_text.setText(f"Push planning failed: {exc}. Adjust pose/options, then press P again.")
            print(f"[sim_push_pipeline] Push planning failed: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False

def main() -> None:
    args = parse_args()
    if args.skip_plan:
        raise RuntimeError("This push-only script is for planning push paths; do not pass --skip-plan.")
    if args.action_sequence_config is None or not args.action_sequence_config.exists():
        raise FileNotFoundError(f"Action sequence config not found: {args.action_sequence_config}")

    if sim_pipeline.should_run_interactive(args):
        if args.object_summary is not None:
            icp = sim_pipeline.read_object_summary(args.object_summary)
            app = InteractiveBottlePushApp(args, initial_icp=icp)
        else:
            ctx = sim_pipeline.prepare_interactive_pipeline_context(args)
            app = InteractiveBottlePushApp(args, ctx=ctx)
        app.run()
        return

    icp = sim_pipeline.run_or_reuse_object_icp(args)
    planning_results = [plan_one_action_sequence(args, icp, sequence) for sequence in args.push_action_sequences]
    summary_path = write_summary(args, icp, planning_results)

    print("[sim_push_pipeline] done")
    print(f"[sim_push_pipeline] action sequence config: {args.action_sequence_config}")
    print(f"[sim_push_pipeline] bottle transform: {icp.bottle_transform_path}")
    print(f"[sim_push_pipeline] box transform: {icp.box_transform_path}")
    for planning in planning_results:
        print(
            f"[sim_push_pipeline] action sequence {planning.action_sequence}: "
            f"frames={len(planning.mot_data.jv_list)}, "
            f"selected_grasp=pickle_grasp_{planning.selected_grasp_index}, "
            f"rtde={planning.rtde_plan_path}"
        )
    print(f"[sim_push_pipeline] summary: {summary_path}")


if __name__ == "__main__":
    main()