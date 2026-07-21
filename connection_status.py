"""Connection status helpers for the UR7e/DH76 robot and Mech-Eye camera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import cv2
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# Keep these in sync with wrs.drivers.devices.Mech_eye.Mech_camera.CaptureImage.
# They are the current Mech-Eye intrinsics used by the existing backend.
MECH_EYE_FX = 2419.864311550494
MECH_EYE_FY = 2419.976305839313
MECH_EYE_CX = 972.6079631929535
MECH_EYE_CY = 562.4548210192155


@dataclass
class ConnectionCheck:
    name: str
    ok: bool
    detail: str = ""
    error: Optional[str] = None

    def line(self) -> str:
        state = "OK" if self.ok else "FAIL"
        suffix = self.detail
        if self.error:
            suffix = f"{suffix}; {self.error}" if suffix else self.error
        return f"{self.name}: {state}" + (f" ({suffix})" if suffix else "")


@dataclass
class LiveConnectionStatus:
    checks: list[ConnectionCheck]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def line(self) -> str:
        return " | ".join(check.line() for check in self.checks)

    def multiline(self) -> str:
        return "\n".join(check.line() for check in self.checks)


def ok(name: str, detail: str = "") -> ConnectionCheck:
    return ConnectionCheck(name=name, ok=True, detail=detail)


def fail(name: str, error: object, detail: str = "") -> ConnectionCheck:
    return ConnectionCheck(name=name, ok=False, detail=detail, error=str(error))


def _is_connected(obj) -> Optional[bool]:
    method = getattr(obj, "isConnected", None)
    if method is None:
        return None
    return bool(method())


def check_robot_provider(provider, mock: bool = False) -> LiveConnectionStatus:
    if mock:
        return LiveConnectionStatus([
            ok("robot RTDE control", "mock"),
            ok("robot RTDE receive", "mock"),
            ok("DH76 gripper", "mock"),
        ])

    checks: list[ConnectionCheck] = []
    robot_x = getattr(provider, "_robot_x", None)

    rtde_c = None if robot_x is None else getattr(robot_x, "rtde_c", getattr(robot_x, "_rtde_c", None))
    rtde_r = None if robot_x is None else getattr(robot_x, "rtde_r", getattr(robot_x, "_rtde_r", None))
    hnd = None if robot_x is None else getattr(robot_x, "hnd", getattr(robot_x, "_hnd", None))

    try:
        connected = _is_connected(rtde_c)
        if connected is False:
            checks.append(fail("robot RTDE control", "isConnected() returned False"))
        elif connected is True:
            checks.append(ok("robot RTDE control", "connected"))
        elif robot_x is not None and hasattr(robot_x, "check_rtdec_is_connected"):
            checks.append(ok("robot RTDE control", "reconnected" if robot_x.check_rtdec_is_connected() else "not connected"))
            if checks[-1].detail == "not connected":
                checks[-1].ok = False
        else:
            checks.append(ok("robot RTDE control", "not directly checkable"))
    except Exception as exc:
        checks.append(fail("robot RTDE control", exc))

    try:
        connected = _is_connected(rtde_r)
        jnt_values = np.asarray(provider.get_jnt_values(), dtype=float).reshape(-1)
        detail = f"connected, joints={np.round(np.degrees(jnt_values), 2).tolist()}deg"
        if connected is False:
            checks.append(fail("robot RTDE receive", "isConnected() returned False", detail=detail))
        else:
            checks.append(ok("robot RTDE receive", detail))
    except Exception as exc:
        checks.append(fail("robot RTDE receive", exc))

    try:
        gripper_connected = None
        if hnd is not None and hasattr(hnd, "check_connection"):
            gripper_connected = bool(hnd.check_connection())
        jaw_width = provider.get_jaw_width() if hasattr(provider, "get_jaw_width") else None
        detail = ""
        if jaw_width is not None:
            detail = f"jaw_width={float(jaw_width):.6f}m"
        if gripper_connected is False:
            checks.append(fail("DH76 gripper", "check_connection() returned False", detail=detail))
        elif jaw_width is None and gripper_connected is None:
            checks.append(fail("DH76 gripper", "jaw width unavailable"))
        else:
            checks.append(ok("DH76 gripper", detail or "connected"))
    except Exception as exc:
        checks.append(fail("DH76 gripper", exc))

    return LiveConnectionStatus(checks)


def check_camera_instance(capture_image) -> ConnectionCheck:
    try:
        if not hasattr(capture_image, "is_connected"):
            return ok("Mech-Eye camera", "is_connected() unavailable after construction")
        if capture_image.is_connected():
            return ok("Mech-Eye camera", "connected")
        return fail("Mech-Eye camera", "is_connected() returned False")
    except Exception as exc:
        return fail("Mech-Eye camera", exc)


def open_camera_for_status(save_directory: Path | str) -> tuple[object, ConnectionCheck]:
    from wrs.drivers.devices.Mech_eye.Mech_camera import CaptureImage

    camera = CaptureImage(save_directory=str(save_directory))
    return camera, check_camera_instance(camera)


def close_camera_quietly(capture_image) -> None:
    camera = getattr(capture_image, "camera", None)
    if camera is None:
        return
    try:
        camera.disconnect()
    except Exception:
        pass


def _generate_mech_eye_pointcloud_fast(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    depth_scale: float,
    depth_trunc: float,
    require_aligned_pixels: bool,
):
    import cv2
    import open3d as o3d

    if color_image is None or depth_image is None:
        raise RuntimeError("Mech-Eye capture returned empty RGB/depth images.")

    depth = np.asarray(depth_image)
    color = np.asarray(color_image)
    height, width = depth.shape[:2]
    if color.shape[:2] != (height, width):
        if require_aligned_pixels:
            raise RuntimeError(
                "RGB/depth dimensions differ; direct point-to-pixel mapping requires aligned images: "
                f"rgb={color.shape[:2]}, depth={depth.shape[:2]}."
            )
        color = cv2.resize(color, (width, height))

    depth_values = depth.astype(np.float32, copy=False).reshape(-1) * float(depth_scale)
    valid_mask = (depth_values > 0.0) & (depth_values < float(depth_trunc)) & np.isfinite(depth_values)
    pixel_indices = np.flatnonzero(valid_mask).astype(np.int64)
    if len(pixel_indices) == 0:
        raise RuntimeError("Point cloud is empty after depth validity filtering.")

    z = depth_values[pixel_indices].astype(np.float64, copy=False)
    u = (pixel_indices % width).astype(np.float64, copy=False)
    v = (pixel_indices // width).astype(np.float64, copy=False)

    points = np.empty((len(pixel_indices), 3), dtype=np.float64)
    points[:, 0] = (u - MECH_EYE_CX) * z / MECH_EYE_FX
    points[:, 1] = -((v - MECH_EYE_CY) * z / MECH_EYE_FY)
    points[:, 2] = -z

    if color.ndim == 2:
        color_rgb = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)
    elif color.shape[2] == 4:
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
    else:
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    colors = color_rgb.reshape(-1, 3)[pixel_indices].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, pixel_indices


def capture_mech_eye_pointcloud_checked(
    output_dir: Path,
    ply_out: Optional[Path],
    depth_scale: float,
    depth_trunc: float,
    save_ply: bool = True,
    return_pixel_indices: bool = False,
    return_rgb: bool = False,
    save_rgb: bool = True,
    camera: object = None,
): # 梅卡曼德相机拍照函数

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / "rgb.png"
    pointcloud_output_path = ply_out if ply_out is not None else output_dir / "colored_pointcloud.ply"
    colored_ply_path = pointcloud_output_path if save_ply else None
    if colored_ply_path is not None:
        colored_ply_path.parent.mkdir(parents=True, exist_ok=True)

    # When a camera is supplied, reuse the live connection and do NOT disconnect
    # it here (the caller owns its lifecycle). Otherwise open/close per call.
    owns_camera = camera is None
    try:
        if camera is None:
            camera, camera_status = open_camera_for_status(output_dir)
        else:
            if hasattr(camera, "save_directory"):
                camera.save_directory = str(output_dir)
            if hasattr(camera, "check_connect_and_reconnect"):
                camera.check_connect_and_reconnect()
            camera_status = check_camera_instance(camera)
        if not camera_status.ok:
            raise ConnectionError(camera_status.line())
        if hasattr(camera, "capture_rgb_and_depth"):
            rgb, depth = camera.capture_rgb_and_depth(save=False, show=False)
            pcd, pixel_indices = _generate_mech_eye_pointcloud_fast(
                rgb,
                depth,
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                require_aligned_pixels=return_pixel_indices,
            )
        else:
            rgb, depth, pcd = camera.capture_and_generate_pointcloud(
                save=False,
                show=False,
                pcb_out_path=str(pointcloud_output_path),
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                keep_invalid=False,
            )
            pixel_indices = None
        if rgb is None or pcd is None:
            raise RuntimeError("Mech-Eye capture did not return RGB data and point cloud data.")
        if return_pixel_indices:
            if pixel_indices is None:
                if depth is None:
                    raise RuntimeError("Mech-Eye capture did not return depth data for pixel index mapping.")
                if np.asarray(rgb).shape[:2] != np.asarray(depth).shape[:2]:
                    raise RuntimeError(
                        "RGB/depth dimensions differ; direct point-to-pixel mapping requires aligned images: "
                        f"rgb={np.asarray(rgb).shape[:2]}, depth={np.asarray(depth).shape[:2]}."
                    )
                depth_values = np.asarray(depth).astype(np.float32, copy=False).reshape(-1) * float(depth_scale)
                valid_mask = (depth_values > 0.0) & (depth_values < float(depth_trunc)) & np.isfinite(depth_values)
                pixel_indices = np.flatnonzero(valid_mask).astype(np.int64)
            point_count = len(np.asarray(pcd.points))
            if len(pixel_indices) != point_count:
                raise RuntimeError(
                    "Depth valid-mask pixel count does not match generated point cloud count: "
                    f"{len(pixel_indices)} != {point_count}."
                )
        if save_rgb:
            cv2.imwrite(str(rgb_path), rgb)
        else:
            rgb_path = None
        if not pcd.has_colors():
            print("Warning: captured point cloud has no color data.")
        if colored_ply_path is not None:
            o3d.io.write_point_cloud(str(colored_ply_path), pcd, write_ascii=False)
        if return_pixel_indices and return_rgb:
            return pcd, rgb_path, colored_ply_path, camera_status, pixel_indices, rgb
        if return_pixel_indices:
            return pcd, rgb_path, colored_ply_path, camera_status, pixel_indices
        if return_rgb:
            return pcd, rgb_path, colored_ply_path, camera_status, rgb
        return pcd, rgb_path, colored_ply_path, camera_status
    finally:
        if owns_camera and camera is not None:
            close_camera_quietly(camera)


def check_mech_eye_camera(save_directory: Path | str = ".") -> LiveConnectionStatus:
    camera = None
    try:
        camera, camera_status = open_camera_for_status(save_directory)
        return LiveConnectionStatus([camera_status])
    except Exception as exc:
        return LiveConnectionStatus([fail("Mech-Eye camera", exc)])
    finally:
        if camera is not None:
            close_camera_quietly(camera)


def print_status(status: LiveConnectionStatus, prefix: str = "[connection]") -> None:
    for check in status.checks:
        print(f"{prefix} {check.line()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check UR7e/DH76 and Mech-Eye connection status.")
    parser.add_argument("--robot-ip", default="192.168.125.30")
    parser.add_argument("--gp-port", default="COM3")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--skip-robot", action="store_true")
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--camera-workdir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_checks: list[ConnectionCheck] = []
    provider = None
    if not args.skip_robot:
        if args.mock:
            robot_status = check_robot_provider(None, mock=True)
        else:
            from yanjiuyuan.sync_real_ur7e_mech_eye_box_env import make_robot_provider

            provider_args = argparse.Namespace(robot_ip=args.robot_ip, gp_port=args.gp_port, mock=False)
            try:
                provider = make_robot_provider(provider_args)
                robot_status = check_robot_provider(provider, mock=False)
            except Exception as exc:
                robot_status = LiveConnectionStatus([fail("robot/DH76", exc)])
        print_status(robot_status)
        all_checks.extend(robot_status.checks)
    if not args.skip_camera:
        camera_status = check_mech_eye_camera(args.camera_workdir)
        print_status(camera_status)
        all_checks.extend(camera_status.checks)
    if provider is not None:
        provider.close()
    if any(not check.ok for check in all_checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()