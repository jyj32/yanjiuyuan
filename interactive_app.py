from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import threading
import queue
import time
from typing import Any, Optional

import numpy as np

from time import perf_counter
import open3d as o3d
import traceback

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, Notify, Vec3, Point3
from wrs import mcm, mgm, ppp
from wrs.robot_con.ur.ur7e_dh76_rtde import UR7EDH76_RTDE
from wrs.robot_sim.robots.ur7e.ur7e_withouttable_dh76 import UR7EDH76
import wrs.visualization.panda.world as wd
from yanjiuyuan import point_hint_segment as phs
from yanjiuyuan.yolo_detect2 import BottleDetector

REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from yanjiuyuan.constants import BOX_CAPTURE_ROOT, BOTTLE_ROBOT_SIDE_PLACE_POS, CAMERA_TO_WORLD # noqa: E402
from yanjiuyuan import box_object_pointcloud_sam_completion_template_icp_with_yolo2 as box_object_icp  # noqa: E402
from yanjiuyuan import connection_status as conn_status  # noqa: E402
from yanjiuyuan import pick_place_rtde_utils as rtde_utils  # noqa: E402
from yanjiuyuan import sim_pick_and_place as sim_pick  # noqa: E402
from yanjiuyuan import sync_real_ur7e_mech_eye_box_env as sync_scene  # noqa: E402
from yanjiuyuan.box_collision import (  # noqa: E402
    make_concave_box_collision_obstacles,
    make_detected_box_visual_model,
)
from yanjiuyuan.constants import REAL_PIPELINE_CONFIG
from types1 import ObjectIcpResult, PlanningResult, RtdeObjectPose
import utils
from wrs.drivers.devices.Mech_eye.Mech_camera import CaptureImage

# 原文件独有的符号（interactive_app 不搬走，从原文件导入；原文件在末尾再导入本模块，避免循环）
from real_bottle_pick_place_interactive3_point_completion_with_yolo2_dual import (
    apply_sam_task_settings,
    bottle_pose_summary_from_summary,
    capture_synced_context,
    move_to_capture_point,
    object_icp_result_from_summary,
    refresh_sam_task_settings,
)
from real_pipeline_planning import (
    build_obstacle_lists,
    run_or_skip_plan,
    write_summary,
    # 下列路径规划辅助函数已统一集中到 real_pipeline_planning.py
    bottle_homomat_for_icp,
    resolve_place_pose,
)


class InteractiveBottlePickPlaceApp:    # 交互式瓶子抓取CDPO
    def __init__(
        self,
        args: SimpleNamespace,
        ctx: Optional[box_object_icp.PipelineContext] = None,
        initial_icp: Optional[ObjectIcpResult] = None,
    ):
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
        self._failed_det_indices: set[int] = set()   # 路径规划失败的物体索引
        self._current_obj_index: int | None = None   # 当前检测成功的物体索引
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

        initial_text = "Entering interactive CDPO mode; robot will move to capture point. Then press C to sync/capture."
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
                pick_pose = utils.homomat_to_pose(bottle_homomat_for_icp(self.icp_result))
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
                box_homomat = utils.load_homomat(self.icp_result.box_transform_path, "box")
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
        pick_pose = utils.homomat_to_pose(bottle_homomat)
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
        self.status_text.setText("C sync: moving robot to capture point (retract, clear view)...")
        self.connection_status_text.setText("Connections: pending...")
        try:
            # 先让机器人后撤到拍照点（机械臂远离取料台上方、不遮挡视野），再拍照，
            # 保证采集到的点云不含机械臂遮挡。与自动模式主循环首步 move_to_capture_point 一致。
            move_to_capture_point(self.args)
            self.status_text.setText("C sync: checking connections, reading robot state, capturing Mech-Eye point cloud...")
            self.connection_status_text.setText("Connections: checking...")
            ctx, metadata = capture_synced_context(self.args)
            self.ctx = ctx
            self.icp_result = None
            self.planning_result = None
            self.environment_stale = False
            self.detect_attempt_count = 0
            self._failed_det_indices.clear()
            self._current_obj_index = None
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
            traceback.print_exc()
        finally:
            self.running = False

    def run_execute(self) -> None:
        # 抓取执行
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

    def _draw_detection_preview(self, image_bgr, detections):
        """在 RGB 上绘制每个 YOLO OBB（红框）与框号码（绿字），保存预览 PNG，返回路径。"""
        import cv2 as _cv2

        vis = np.asarray(image_bgr, dtype=np.uint8).copy()
        for i, d in enumerate(detections):
            corners = d.get("corners_px")
            if corners is not None:
                pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
                _cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            bbox = d.get("bbox")
            if bbox is not None:
                tx, ty = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
            elif corners is not None:
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                tx, ty = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
            else:
                tx, ty = 10, 20
            _cv2.putText(vis, str(i), (tx, ty), _cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        out_dir = self.ctx.capture.capture_dir
        out_path = out_dir / "yolo_detection_preview.png"
        _cv2.imwrite(str(out_path), vis)
        return out_path

    def _show_image_file(self, path) -> None:
        """用系统默认图片查看器直接弹出显示图片（非阻塞，失败仅告警）。"""
        _path = str(path)
        try:
            if sys.platform == "win32":
                # start 后的第一个空串是窗口标题占位符，避免路径含空格时报错
                subprocess.run(["cmd", "/c", "start", "", _path], check=False)
            elif sys.platform == "darwin":
                subprocess.run(["open", _path], check=False)
            else:
                subprocess.run(["xdg-open", _path], check=False)
            print(f"[real_pipeline] 已弹出检测预览图（请在图片查看器中查看框号码）: {_path}")
        except Exception as _e:
            print(f"[real_pipeline] 无法自动打开预览图（忽略）: {_e}")

    def run_detection(self) -> None:
        # 检测程序（交互式：YOLO 显示所有候选框 -> 人工输入检测顺序 -> 按序逐个处理指定框并各自保存记录）
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
            # ---- 1. YOLO 检测所有瓶子 ----
            detected = box_object_icp.detect_bottles(self.ctx)
            if detected is None:
                self.status_text.setText("YOLO 未检测到任何瓶子，按 D 重试。")
                print("[real_pipeline] YOLO 未检测到任何瓶子")
                return
            image_bgr, detections = detected
            if not detections:
                self.status_text.setText("YOLO 未检测到任何瓶子，按 D 重试。")
                print("[real_pipeline] YOLO 未检测到任何瓶子")
                return
            # ---- 2. 显示检测结果（框号码）并绘制预览图 ----
            print("[real_pipeline] YOLO 检测到以下候选框：")
            for i, d in enumerate(detections):
                print(f"  框 {i}: class={d.get('class_name')} conf={float(d.get('confidence', 0)):.3f} "
                      f"area={float(d.get('obb_area', 0)):.0f} bbox={d.get('bbox')}")
            try:
                _preview = self._draw_detection_preview(image_bgr, detections)
                if _preview is not None:
                    print(f"[real_pipeline] 检测预览图已保存: {_preview}")
                    self._show_image_file(_preview)
            except Exception as _pe:
                print(f"[real_pipeline] 绘制/显示检测预览失败（忽略）: {_pe}")
            # ---- 3. 自动按检测顺序处理，不再等待人工输入 ----
            # detect_bottles 已调用优先级推理模型（grasp_sequence），返回的 detections
            # 已是 best-first 顺序；直接采用该顺序作为抓取顺序，首个成功框即优先级最高的可抓取框。
            _order = list(range(len(detections)))
            self.status_text.setText(f"已按检测顺序 {_order} 开始处理（每轮仅处理首个可成功抓取的框）...")
            print(f"[real_pipeline] 自动采用检测顺序 {_order}（无需人工输入）")
            self.status_text.setText(f"已按检测顺序 {_order} 开始处理（每轮仅处理顺序中首个可成功抓取的框，其余留待后续轮次）...")
            print(f"[real_pipeline] 用户指定检测顺序 {_order}，本轮只处理首个成功框（不再对后续框做 ICP）")

            # ---- 4. 按人工输入的检测顺序，逐个框做 SAM 分割 + 补全 + ICP，并各自保存记录（跳过其余）----
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

            _last_summary = None
            _last_mask = None
            _primary_set = False
            for _k, _choice in enumerate(_order):
                self.status_text.setText(f"检测顺序 {_k + 1}/{len(_order)}：处理框 {_choice}（SAM 分割 + ICP）...")
                print(f"[real_pipeline] 检测顺序 {_k + 1}/{len(_order)}：处理框 {_choice}")
                _skip = set(range(len(detections))) - {_choice}
                try:
                    # yolo检测（仅当前框），sam分割，补全点云，icp匹配
                    summary, _masks, selected_mask = box_object_icp.run_segmentation_and_bottle_icp_with_fallback(
                        self.ctx, skip_indices=_skip
                    )
                except Exception as _box_exc:
                    print(f"[real_pipeline] 警告: 框 {_choice} 处理失败: {_box_exc}")
                    traceback.print_exc()
                    continue
                _last_summary, _last_mask = summary, selected_mask
                # 首个成功处理的框作为 P 规划的主物体（周围点云 / 物体索引）
                if not _primary_set:
                    self.args.remaining_pointcloud_path = summary.get("remaining_pointcloud_path")
                    self._current_obj_index = summary.get("obj_index")
                    _primary_set = True
                self.attach_detection_result(summary, selected_mask)
                bottle_summary, pose_source = bottle_pose_summary_from_summary(summary)
                template_pointcloud_id = bottle_summary.get("template", "surface")
                print(
                    f"[real_pipeline] 框 {_choice} 完成: template_pointcloud_id={template_pointcloud_id} pose_source={pose_source}",
                    flush=True,
                )
                # 本轮只处理首个成功框：第一个框已有抓取路径即不再对后续框做 ICP，
                # 直接结束本轮（其余框留待下一次 C/D 轮次再处理）。
                break
            if _last_summary is None:
                self.status_text.setText("所有指定框处理均失败，按 D 重试。")
                print("[real_pipeline] 所有指定框处理失败")
                return
            self.status_text.setText(f"已处理顺序 {_order} 中首个可抓取框，可按 P 规划（主物体=该框）。")
        except Exception as exc:
            self.status_text.setText(f"Attempt {self.detect_attempt_count}: failed: {exc}. Press D to retry.")
            print(f"[real_pipeline] Detection attempt {self.detect_attempt_count} failed: {exc}")
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
        _plan_t0 = time.time()
        try:
            # ---- 多物体规划回退循环 ----
            # 当前物体路径规划失败时，自动跳到下一个物体重试检测+规划
            while True:
                plan_args = copy.copy(self.args)
                plan_args.dry_run = True
                try:
                    planning = run_or_skip_plan(plan_args, self.icp_result)
                    break  # 规划成功（或被 skip），跳出循环
                except Exception as plan_exc:
                    # 路径规划失败，尝试下一个物体
                    if self._current_obj_index is None:
                        raise  # 没有物体索引信息，直接报错
                    self._failed_det_indices.add(self._current_obj_index)
                    print(f"[real_pipeline] 物体{self._current_obj_index + 1} 路径规划失败: {plan_exc}")
                    print(f"[real_pipeline] 尝试检测下一个物体（跳过索引 {sorted(self._failed_det_indices)}）...")
                    self.status_text.setText(
                        f"物体{self._current_obj_index + 1} 规划失败，尝试下一个物体..."
                    )
                    # 重新检测，跳过已失败的物体
                    try:
                        next_summary, _masks, next_selected_mask = \
                            box_object_icp.run_segmentation_and_bottle_icp_with_fallback(
                                self.ctx, skip_indices=self._failed_det_indices
                            )
                    except Exception:
                        # 所有物体都已失败
                        raise RuntimeError(
                            f"所有物体路径规划均失败（已尝试 {len(self._failed_det_indices)} 个物体）"
                        ) from plan_exc
                    # 更新检测结果为新物体
                    self.args.remaining_pointcloud_path = next_summary.get("remaining_pointcloud_path")
                    self._current_obj_index = next_summary.get("obj_index")
                    self.clear_detection_models()
                    self.attach_detection_result(next_summary, next_selected_mask)
                    self.icp_result = object_icp_result_from_summary(next_summary)
                    self.attach_global_registered_template_points(self.icp_result)
                    self.attach_start_and_place_models(self.icp_result)
                    bottle_summary, pose_source = bottle_pose_summary_from_summary(next_summary)
                    print(
                        f"[real_pipeline] 切换到物体{self._current_obj_index + 1}: "
                        f"pose_source={pose_source}, "
                        f"pick={sim_pick.format_vec(utils.homomat_to_pose(bottle_homomat_for_icp(self.icp_result))[0], digits=6)}",
                        flush=True,
                    )
                    continue  # 用新物体重试规划

            if planning is None:
                self.status_text.setText("Planning skipped.")
                return
            self.planning_result = planning
            self.attach_plan_result(planning)
            summary_path = write_summary(self.args, self.icp_result, planning)
            rtde_path = "" if planning.rtde_plan_path is None else f" RTDE={planning.rtde_plan_path}"
            done_sequence_text = "" if planning.action_sequence is None else f" for action sequence {planning.action_sequence}"
            _plan_elapsed = time.time() - _plan_t0
            self.status_text.setText(
                f"Planning done{done_sequence_text}: {len(planning.mot_data.jv_list)} frames. Press O to execute."
            )
            print(
                f"[real_pipeline] summary after planning{done_sequence_text}: "
                f"{summary_path}{rtde_path} (planning_step_total={_plan_elapsed:.3f}s)"
            )
        except Exception as exc:
            self.status_text.setText(f"Planning failed: {exc}. Adjust pose/options, then press P again.")
            print(f"[real_pipeline] Planning failed: {exc}")
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

    def _move_to_capture_on_start(self, task):
        """进入 CDPO 交互模式后，先让机器人后撤到拍照点（不遮挡视野），再提示按 C 拍照。

        Panda3D 的 task 回调：窗口已渲染首帧后才执行，避免阻塞窗口创建。
        """
        try:
            move_to_capture_point(self.args)
            self.status_text.setText(
                "Robot at capture point. Press C to sync/capture, D for pose estimation, P to plan, O to execute."
            )
        except Exception as exc:
            self.status_text.setText(f"Init move to capture point failed: {exc}. Press C to retry.")
            print(f"[real_pipeline] Init move to capture point failed: {exc}")
            traceback.print_exc()
        return task.done

    def run(self) -> None:
        # 进入交互模式时，先让机器人在后台后撤到拍照点（不遮挡视野）：窗口先渲染一帧，
        # 再移动机器人；移动完成后提示用户按 C 拍照。满足"最初让机器人到拍摄点再按 C"。
        self.base.taskMgr.doMethodLater(0.5, self._move_to_capture_on_start, "init_move_to_capture")
        self.base.run()

    def run_auto_pipeline(self) -> None:
        """Run the full C->D->P->O pipeline automatically without key presses.

        After completing one full cycle (capture->detection->planning->execution),
        the pipeline loops back to capture the next scene automatically.
        Close the window to stop.
        """

        steps = [
            ("capture", self.run_sync_capture),
            ("detection", self.run_detection),
            ("planning", self.run_plan),
            ("execution", self.run_execute),
        ]

        state = {"index": 0, "cycle": 1, "pre_exec_seq": None}

        def auto_task(task):
            # --- All steps done -> start next cycle ---
            if state["index"] >= len(steps):
                cycle = state["cycle"]
                print(f"[real_pipeline] Auto pipeline: cycle {cycle} complete.")
                state["cycle"] += 1
                state["index"] = 0
                state["pre_exec_seq"] = None
                next_cycle = state["cycle"]
                print(f"[real_pipeline] Auto pipeline: starting cycle {next_cycle} in 3 s ...")
                self.status_text.setText(
                    f"Cycle {cycle} done. Starting cycle {next_cycle} ...")
                # Reschedule after a 3-second delay; cannot return a float
                # from a Panda3D task.
                self.base.taskMgr.doMethodLater(
                    3.0, auto_task, "auto_pipeline", appendTask=True)
                return task.done

            label, fn = steps[state["index"]]

            # Remember action sequence before execution to detect success.
            if label == "execution":
                state["pre_exec_seq"] = self.current_action_sequence()

            print(
                f"[real_pipeline] Auto pipeline: cycle {state['cycle']}, "
                f"step {state['index'] + 1}/{len(steps)} ({label})."
            )
            fn()

            # --- Evaluate whether the step succeeded ---
            if label == "capture":
                if self.ctx is None:
                    print("[real_pipeline] Auto pipeline: capture failed, stopping.")
                    self.status_text.setText("Auto pipeline: capture failed. See console.")
                    return task.done
            elif label == "detection":
                if self.icp_result is None:
                    print("[real_pipeline] Auto pipeline: detection failed, stopping.")
                    self.status_text.setText("Auto pipeline: detection failed. See console.")
                    return task.done
            elif label == "planning":
                if self.planning_result is None:
                    print("[real_pipeline] Auto pipeline: planning failed, stopping.")
                    self.status_text.setText("Auto pipeline: planning failed. See console.")
                    return task.done
                if self.planning_result.rtde_plan is None:
                    print("[real_pipeline] Auto pipeline: no RTDE plan, stopping before execution.")
                    self.status_text.setText("Auto pipeline: no RTDE plan generated. See console.")
                    return task.done
            elif label == "execution":
                # Detect execution failure: if action sequence did not advance,
                # the execution threw an exception internally.
                post_seq = self.current_action_sequence()
                if (
                    state["pre_exec_seq"] is not None
                    and post_seq == state["pre_exec_seq"]
                ):
                    print("[real_pipeline] Auto pipeline: execution failed, stopping.")
                    self.status_text.setText("Auto pipeline: execution failed. See console.")
                    return task.done

            state["index"] += 1
            return task.again

        # Small delay so the Panda3D window renders one frame before the
        # long-running capture step blocks the event loop.
        self.base.taskMgr.doMethodLater(0.5, auto_task, "auto_pipeline", appendTask=True)
        self.base.run()
