"""
Async real-environment interactive pipeline.

This entry point keeps the Panda/WRS viewer responsive while the real pipeline
is busy:

  C: queue capture -> SAM/ICP -> WRS planning in a long-lived worker process.
  O: execute the current RTDE plan in a background thread.

The worker process preloads the SAM/point-hint model once at startup. Hardware
access is still serialized for safety: C pressed during O is queued and starts
after O finishes.
"""

from __future__ import annotations

import atexit
import copy
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import queue
from types import SimpleNamespace
from typing import Any, Optional
import threading
import time
import traceback

import numpy as np

from yanjiuyuan import pick_place_rtde_utils as rtde_utils
from yanjiuyuan import real_bottle_pick_place_interactive3 as ip3


WORKER_SELECTED_POINT_LIMIT = 30000
ASYNC_POLL_INTERVAL = 0.05


@dataclass
class AsyncPlanResult:
    rtde_plan: Optional[rtde_utils.RtdeExecutionPlan]
    rtde_plan_path: Optional[Path]
    frame_count: int
    action_sequence: Optional[int]
    selected_grasp_index: Optional[int]
    summary_path: Path


def _args_payload(args: SimpleNamespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in {"point_hint_model", "point_hint_model_key"}:
            continue
        payload[key] = copy.deepcopy(value)
    return payload


def _args_from_payload(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**copy.deepcopy(payload))


def _reset_live_input_paths(args: SimpleNamespace) -> None:
    args.capture_dir = None
    args.ply = None
    args.image = None
    args.box_transform = None
    args.object_summary = None
    args.object_output_dir = None


def _path_text(path: Optional[Path]) -> Optional[str]:
    return None if path is None else str(Path(path))


def _status_text(metadata: dict) -> str:
    items: list[str] = []
    robot_status = metadata.get("robot_status")
    if robot_status is not None:
        for check in getattr(robot_status, "checks", []):
            name = getattr(check, "name", "")
            ok = "OK" if getattr(check, "ok", False) else "FAIL"
            if name == "robot RTDE control":
                items.append(f"RTDE-C:{ok}")
            elif name == "robot RTDE receive":
                items.append(f"RTDE-R:{ok}")
            elif name == "DH76 gripper":
                items.append(f"Gripper:{ok}")
    camera_status = metadata.get("camera_status")
    if camera_status is not None:
        ok = "OK" if getattr(camera_status, "ok", False) else "FAIL"
        items.append(f"Camera:{ok}")
    return "Connections: " + (" ".join(items) if items else "checked")


def _downsample_for_ipc(points: np.ndarray, limit: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if limit <= 0 or len(points) <= limit:
        return points
    step = max(1, int(np.ceil(len(points) / float(limit))))
    return points[::step][:limit]


def _make_job_args(base_args: SimpleNamespace, payload: dict[str, Any]) -> SimpleNamespace:
    job_args = copy.copy(base_args)
    model = getattr(base_args, "point_hint_model", None)
    model_key = getattr(base_args, "point_hint_model_key", None)
    for key, value in payload.items():
        if key in {"point_hint_model", "point_hint_model_key"}:
            continue
        setattr(job_args, key, copy.deepcopy(value))
    if model is not None:
        if getattr(job_args, "model", None) is None:
            job_args.model = getattr(base_args, "model", None)
        job_args.point_hint_model = model
        job_args.point_hint_model_key = model_key
    ip3.normalize_paths(job_args)
    _reset_live_input_paths(job_args)
    return job_args


def _run_pipeline_job(job_id: int, args: SimpleNamespace) -> dict[str, Any]:
    started = time.perf_counter()
    ctx, metadata = ip3.capture_synced_context(args)

    sam_settings = ip3.refresh_sam_task_settings(args)
    if sam_settings is not None:
        ip3.apply_sam_task_settings(ctx.args, sam_settings)
    ctx.args.bottle_template = args.bottle_template
    ctx.args.bottle_template_prompt_gui = args.bottle_template_prompt_gui
    ctx.args.bottle_template_ply = args.bottle_template_ply

    with ip3.time_stage("ASYNC DETECTION+ICP"):
        summary, _masks, selected_mask = ip3.box_object_icp.run_segmentation_and_bottle_icp_attempt(ctx)
    icp = ip3.object_icp_result_from_summary(summary)

    planning = None
    if not args.skip_plan:
        plan_args = copy.copy(args)
        plan_args.dry_run = True
        with ip3.time_stage("ASYNC PLAN"):
            planning = ip3.run_or_skip_plan(plan_args, icp)
    summary_path = ip3.write_summary(args, icp, planning)

    selected_points = ctx.capture.points_world[np.asarray(selected_mask, dtype=bool)]
    selected_points = _downsample_for_ipc(selected_points, WORKER_SELECTED_POINT_LIMIT)

    result = {
        "job_id": job_id,
        "elapsed": time.perf_counter() - started,
        "output_dir": str(ctx.output_dir),
        "connection_status_text": _status_text(metadata),
        "current_jnt_values": metadata.get("current_jnt_values"),
        "current_jaw_width": metadata.get("current_jaw_width"),
        "summary_path": str(summary_path),
        "selected_points": selected_points,
        "icp": {
            "output_dir": str(icp.output_dir),
            "summary_path": str(icp.summary_path),
            "bottle_transform_path": str(icp.bottle_transform_path),
            "box_transform_path": str(icp.box_transform_path),
            "bottle_model_path": str(icp.bottle_model_path),
            "global_registered_path": _path_text(icp.global_registered_path),
        },
        "planning": None,
    }
    if planning is not None:
        result["planning"] = {
            "rtde_plan_path": _path_text(planning.rtde_plan_path),
            "frame_count": len(planning.mot_data.jv_list),
            "action_sequence": planning.action_sequence,
            "selected_grasp_index": planning.selected_grasp_index,
        }
    return result


def _pipeline_worker_main(input_queue: mp.Queue, output_queue: mp.Queue, initial_payload: dict[str, Any]) -> None:
    base_args = _args_from_payload(initial_payload)
    ip3.normalize_paths(base_args)
    _reset_live_input_paths(base_args)
    try:
        ip3.preload_point_hint_model(base_args)
        output_queue.put({"type": "worker_ready", "ok": True})
    except Exception as exc:
        output_queue.put(
            {
                "type": "worker_ready",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )

    while True:
        message = input_queue.get()
        msg_type = message.get("type")
        if msg_type == "shutdown":
            output_queue.put({"type": "worker_shutdown"})
            return
        if msg_type != "pipeline":
            output_queue.put({"type": "worker_warning", "message": f"Unknown worker message: {msg_type}"})
            continue

        job_id = int(message["job_id"])
        try:
            output_queue.put({"type": "pipeline_started", "job_id": job_id})
            job_args = _make_job_args(base_args, message["args"])
            result = _run_pipeline_job(job_id, job_args)
            output_queue.put({"type": "pipeline_done", "job_id": job_id, "result": result})
        except Exception as exc:
            output_queue.put(
                {
                    "type": "pipeline_error",
                    "job_id": job_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )


def _execute_rtde_thread(plan_path: str, config: dict[str, Any], event_queue: queue.Queue) -> None:
    rtde_robot: object = object()
    dry_run = bool(config["dry_run"])
    try:
        plan = rtde_utils.load_rtde_execution_plan(Path(plan_path))
        if not dry_run:
            from robot_con.ur.ur7e_dh76_rtde import UR7EDH76_RTDE

            rtde_robot = UR7EDH76_RTDE(robot_ip=config["robot_ip"], gp_port=config["gp_port"])
            print("[real_pipeline_async] Opening gripper before RTDE execution...")
            rtde_robot.open_gripper()
        else:
            print("[real_pipeline_async] Dry-run: skipping pre-execution gripper open.")

        log = rtde_utils.execute_rtde_execution_plan(
            rtde_robot=rtde_robot,
            plan=plan,
            dry_run=dry_run,
            jntspace_kwargs=config["jntspace_kwargs"],
            jntspace_kwargs_by_segment=config["jntspace_kwargs_by_segment"],
            max_start_joint_error=config["max_start_joint_error"],
            use_move_l_compliant=bool(config["use_move_l_compliant"]),
        )
        event_queue.put({"type": "execute_done", "log": log})
    except Exception as exc:
        event_queue.put(
            {
                "type": "execute_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        disconnect = getattr(rtde_robot, "disconnect", None)
        if disconnect is not None:
            try:
                disconnect()
            except Exception as exc:
                print(f"[real_pipeline_async] Warning: RTDE disconnect failed: {exc}")


class AsyncInteractiveBottlePickPlaceApp(ip3.InteractiveBottlePickPlaceApp):
    def __init__(self, args: SimpleNamespace):
        self.pipeline_busy = False
        self.execute_busy = False
        self.pending_capture_after_execute = False
        self._pipeline_job_id = 0
        self._pipeline_in: Optional[mp.Queue] = None
        self._pipeline_out: Optional[mp.Queue] = None
        self._pipeline_process: Optional[mp.Process] = None
        self._thread_events: queue.Queue = queue.Queue()
        self._execute_thread: Optional[threading.Thread] = None
        super().__init__(args, ctx=None, initial_icp=None)
        self.connection_status_text.setText("Async worker: starting SAM process...")
        self._start_pipeline_worker()
        self.base.taskMgr.doMethodLater(ASYNC_POLL_INTERVAL, self._poll_async_events, "real_pipeline_async_poll")
        print("[real_pipeline_async] Async viewer is ready. C runs in worker process; O runs in execution thread.")

    def ensure_connections(self) -> tuple[object, object]:
        # The async entry point intentionally does not keep robot/camera handles
        # in the UI process. Pipeline jobs open them in the worker process, and O
        # opens RTDE only inside the execution thread.
        return None, None

    def close_connections(self) -> None:
        self._shutdown_pipeline_worker()

    def _start_pipeline_worker(self) -> None:
        if self._pipeline_process is not None and self._pipeline_process.is_alive():
            return
        ctx = mp.get_context("spawn")
        self._pipeline_in = ctx.Queue()
        self._pipeline_out = ctx.Queue()
        self._pipeline_process = ctx.Process(
            target=_pipeline_worker_main,
            args=(self._pipeline_in, self._pipeline_out, _args_payload(self.args)),
            daemon=True,
        )
        self._pipeline_process.start()
        atexit.register(self._shutdown_pipeline_worker)

    def _shutdown_pipeline_worker(self) -> None:
        process = getattr(self, "_pipeline_process", None)
        input_queue = getattr(self, "_pipeline_in", None)
        if process is None:
            return
        if process.is_alive() and input_queue is not None:
            try:
                input_queue.put({"type": "shutdown"})
                process.join(timeout=2.0)
            except Exception:
                pass
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        self._pipeline_process = None

    def _set_busy_flags(self) -> None:
        self.running = self.pipeline_busy or self.execute_busy

    def _poll_async_events(self, task):
        self._drain_pipeline_events()
        self._drain_thread_events()
        return task.again

    def _drain_pipeline_events(self) -> None:
        output_queue = self._pipeline_out
        if output_queue is None:
            return
        while True:
            try:
                event = output_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_pipeline_event(event)

    def _drain_thread_events(self) -> None:
        while True:
            try:
                event = self._thread_events.get_nowait()
            except queue.Empty:
                break
            self._handle_thread_event(event)

    def _handle_pipeline_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "worker_ready":
            if event.get("ok"):
                self.connection_status_text.setText("Async worker: SAM loaded")
                print("[real_pipeline_async] Worker ready; SAM model is loaded once in the worker process.")
            else:
                self.connection_status_text.setText(f"Async worker failed: {event.get('error')}")
                print(f"[real_pipeline_async] Worker startup failed: {event.get('error')}")
                print(event.get("traceback", ""))
            return
        if event_type == "pipeline_started":
            print(f"[real_pipeline_async] C job {event.get('job_id')} started in worker process.")
            return
        if event_type == "pipeline_done":
            if int(event.get("job_id", -1)) != self._pipeline_job_id:
                print(f"[real_pipeline_async] Ignoring stale C job {event.get('job_id')}.")
                return
            self.pipeline_busy = False
            self._set_busy_flags()
            self._apply_pipeline_result(event["result"])
            return
        if event_type == "pipeline_error":
            if int(event.get("job_id", -1)) != self._pipeline_job_id:
                return
            self.pipeline_busy = False
            self._set_busy_flags()
            self.status_text.setText(f"C async pipeline failed: {event.get('error')}. Press C to retry.")
            self.connection_status_text.setText("Async worker: pipeline failed")
            print(f"[real_pipeline_async] C job failed: {event.get('error')}")
            print(event.get("traceback", ""))
            return
        if event_type == "worker_warning":
            print(f"[real_pipeline_async] Worker warning: {event.get('message')}")
            return
        if event_type == "worker_shutdown":
            print("[real_pipeline_async] Worker shutdown.")

    def _handle_thread_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        completed_action_sequence = getattr(self, "_executing_action_sequence", None)
        mode = getattr(self, "_executing_mode", "REAL ROBOT")
        compliant_mode = getattr(self, "_executing_compliant_mode", "joint-path approach")
        self.execute_busy = False
        self._set_busy_flags()
        self.environment_stale = True
        auto_capture_after_success = False
        if event_type == "execute_done":
            log = event.get("log", [])
            next_action_sequence = self.advance_action_sequence()
            if completed_action_sequence is None or next_action_sequence is None:
                completion_text = f"O execution complete ({mode}, {compliant_mode}): {len(log)} segment(s)."
            else:
                completion_text = (
                    f"O execution complete ({mode}, {compliant_mode}): action sequence "
                    f"{completed_action_sequence} done; next is {next_action_sequence}. {len(log)} segment(s)."
                )
            auto_capture_after_success = bool(getattr(self.args, "auto_capture_after_execute", True))
            followup_text = " Auto capture will start next." if auto_capture_after_success else " Press C to recapture."
            self.status_text.setText(f"{completion_text}{followup_text}")
            print(f"[real_pipeline_async] {completion_text}")
            for entry in log:
                print(f"[real_pipeline_async]   {entry}")
        elif event_type == "execute_error":
            self.status_text.setText(f"O execution failed: {event.get('error')}. Press C before retrying.")
            print(f"[real_pipeline_async] O execution failed: {event.get('error')}")
            print(event.get("traceback", ""))

        if self.pending_capture_after_execute:
            self.pending_capture_after_execute = False
            self.status_text.setText("Queued C starting after O...")
            self.run_sync_capture()
        elif auto_capture_after_success:
            self.schedule_auto_capture(
                "after O",
                float(getattr(self.args, "auto_capture_after_execute_delay", 0.25)),
            )

    def _apply_pipeline_result(self, result: dict[str, Any]) -> None:
        icp_data = result["icp"]
        self.ctx = None
        self.icp_result = ip3.ObjectIcpResult(
            output_dir=Path(icp_data["output_dir"]),
            summary_path=Path(icp_data["summary_path"]),
            bottle_transform_path=Path(icp_data["bottle_transform_path"]),
            box_transform_path=Path(icp_data["box_transform_path"]),
            bottle_model_path=Path(icp_data["bottle_model_path"]),
            global_registered_path=None
            if icp_data.get("global_registered_path") is None
            else Path(icp_data["global_registered_path"]),
        )
        self.planning_result = None
        planning_data = result.get("planning")
        if planning_data is not None and planning_data.get("rtde_plan_path") is not None:
            rtde_plan_path = Path(planning_data["rtde_plan_path"])
            self.planning_result = AsyncPlanResult(
                rtde_plan=rtde_utils.load_rtde_execution_plan(rtde_plan_path),
                rtde_plan_path=rtde_plan_path,
                frame_count=int(planning_data.get("frame_count", 0)),
                action_sequence=planning_data.get("action_sequence"),
                selected_grasp_index=planning_data.get("selected_grasp_index"),
                summary_path=Path(result["summary_path"]),
            )

        self.connection_status_text.setText(result.get("connection_status_text", "Async worker: done"))
        self.clear_synced_scene_models()
        self._attach_selected_points(result.get("selected_points"))
        self.attach_global_registered_template_points(self.icp_result)
        self.attach_start_and_place_models(self.icp_result)
        self.attach_scene_obstacles()
        self.attach_synced_robot(result.get("current_jnt_values"), result.get("current_jaw_width"))

        action_sequence = self.current_action_sequence()
        sequence_text = "" if action_sequence is None else f" Action sequence {action_sequence}."
        planning_text = " Planning skipped." if self.planning_result is None else (
            f" Planning done: {self.planning_result.frame_count} frames. Press O to execute."
        )
        self.environment_stale = False
        self.status_text.setText(
            f"C async done: {Path(result['output_dir']).name}.{sequence_text}{planning_text}"
        )
        print(
            f"[real_pipeline_async] C job done in {float(result.get('elapsed', 0.0)):.2f}s; "
            f"summary={result['summary_path']}"
        )

    def _attach_selected_points(self, selected_points: Any) -> None:
        if selected_points is None:
            return
        points = np.asarray(selected_points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            return
        selected_model = self.mgm.gen_pointcloud(
            points,
            rgba=np.array([1.0, 0.0, 0.0, 1.0]),
            point_size=max(float(getattr(self.args, "point_size", 0.002)), 0.0025),
        )
        selected_model.attach_to(self.base)
        self.detection_models.append(selected_model)

    def run_sync_capture(self) -> None:
        if self.execute_busy:
            self.pending_capture_after_execute = True
            self.status_text.setText("O is executing. C is queued and will start after O finishes.")
            print("[real_pipeline_async] C queued after current O execution.")
            return
        if self.pipeline_busy:
            self.status_text.setText("C pipeline is already running in the worker process.")
            print("[real_pipeline_async] C ignored: pipeline worker is busy.")
            return
        if self._pipeline_process is None or not self._pipeline_process.is_alive():
            self._start_pipeline_worker()

        self._pipeline_job_id += 1
        job_id = self._pipeline_job_id
        self.pipeline_busy = True
        self._set_busy_flags()
        self.planning_result = None
        self.icp_result = None
        self.environment_stale = False
        self.detect_attempt_count = 0
        self.clear_synced_scene_models(force_gc=True)
        sequence = self.current_action_sequence()
        sequence_text = "" if sequence is None else f" for action sequence {sequence}"
        self.status_text.setText(f"C async job {job_id} running{sequence_text}: capture, SAM/ICP, planning...")
        print(f"[real_pipeline_async] Queueing C job {job_id}{sequence_text}.")
        assert self._pipeline_in is not None
        self._pipeline_in.put({"type": "pipeline", "job_id": job_id, "args": _args_payload(self.args)})

    def run_detection_and_plan(self) -> None:
        self.run_sync_capture()

    def run_detection(self) -> None:
        self.run_sync_capture()

    def run_plan(self) -> None:
        self.status_text.setText("Planning is part of the async C pipeline in this entry point.")

    def run_execute(self) -> None:
        if self.execute_busy:
            self.status_text.setText("O execution is already running.")
            print("[real_pipeline_async] O ignored: execution is already running.")
            return
        if self.pipeline_busy:
            self.status_text.setText("C pipeline is still running; wait for the new plan before O.")
            print("[real_pipeline_async] O ignored: pipeline worker is busy.")
            return
        if self.environment_stale:
            self.status_text.setText("Environment changed after O. Press C before executing again.")
            return
        if self.planning_result is None or self.planning_result.rtde_plan_path is None:
            self.status_text.setText("No RTDE plan yet. Press C first.")
            return

        rtde_control_frequency = float(getattr(self.args, "rtde_control_frequency", 0.002))
        rtde_rrt_control_frequency = float(getattr(self.args, "rtde_rrt_control_frequency", rtde_control_frequency))
        if rtde_control_frequency <= 0.0 or rtde_rrt_control_frequency <= 0.0:
            self.status_text.setText("Invalid RTDE control frequency; values must be > 0.")
            return

        completed_action_sequence = self.current_action_sequence()
        dry_run = bool(self.args.execute_dry_run or self.args.mock)
        use_move_l_compliant = bool(self.args.use_move_l_compliant)
        mode = "dry-run" if dry_run else "REAL ROBOT"
        compliant_mode = "moveL_compliant" if use_move_l_compliant else "joint-path approach"
        sequence_text = "" if completed_action_sequence is None else f", action sequence {completed_action_sequence}"
        self.status_text.setText(f"O execution running in background ({mode}, {compliant_mode}{sequence_text})...")
        print(
            f"[real_pipeline_async] O execution starting ({mode}, {compliant_mode}{sequence_text}); "
            f"control_frequency={rtde_control_frequency:.4f}s, "
            f"rrt_control_frequency={rtde_rrt_control_frequency:.4f}s"
        )

        jntspace_kwargs = {"control_frequency": rtde_control_frequency}
        rrt_jntspace_kwargs = {"control_frequency": rtde_rrt_control_frequency}
        config = {
            "dry_run": dry_run,
            "robot_ip": self.args.robot_ip,
            "gp_port": self.args.gp_port,
            "jntspace_kwargs": jntspace_kwargs,
            "jntspace_kwargs_by_segment": {
                "move_to_pre_pick": rrt_jntspace_kwargs,
                "transfer_to_place": rrt_jntspace_kwargs,
                "depart_after_place": rrt_jntspace_kwargs,
                "after_pick_motion": rrt_jntspace_kwargs,
            },
            "max_start_joint_error": np.radians(float(self.args.max_start_joint_error_deg)),
            "use_move_l_compliant": use_move_l_compliant,
        }
        self.execute_busy = True
        self._set_busy_flags()
        self._executing_action_sequence = completed_action_sequence
        self._executing_mode = mode
        self._executing_compliant_mode = compliant_mode
        self._execute_thread = threading.Thread(
            target=_execute_rtde_thread,
            args=(str(self.planning_result.rtde_plan_path), config, self._thread_events),
            daemon=True,
            name="real_pipeline_async_execute",
        )
        self._execute_thread.start()


def main() -> None:
    mp.freeze_support()
    args = ip3.make_runtime_config()
    ip3.normalize_paths(args)
    ip3.preload_runtime_json_configs(args)
    ip3.apply_action_sequence_start_settings(args)
    ip3.run_headless_autonomous(args)


if __name__ == "__main__":
    main()
