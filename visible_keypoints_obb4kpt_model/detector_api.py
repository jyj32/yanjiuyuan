#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from obb_pose import OBBPosePredictor


CLASS_NAMES = {0: "occluded", 1: "clear"}
KEYPOINT_NAMES = ("C", "N", "L", "B")


def auto_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# 瓶子的旋转目标检测（OBB）与关键点估计
class BottleOBB4KPTDetector:
    def __init__(
        self,
        weights: str | Path | None = None,
        device: str = "auto",
        confidence: float = 0.528,
        nms_iou: float = 0.70,
        image_size: int = 960,
        keypoint_visibility: float = 0.50,  # 关键点可见性阈值，低于此值视为不可见
    ) -> None:
        self.weights = Path(weights or PACKAGE_ROOT / "model" / "bottle_detect2.pt").resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        self.device = auto_device() if device == "auto" else device
        self.confidence = float(confidence)
        self.nms_iou = float(nms_iou)
        self.image_size = int(image_size)
        self.keypoint_visibility = float(keypoint_visibility)
        self._predictor = OBBPosePredictor(
            overrides={
                "model": str(self.weights),
                "conf": max(0.001, min(self.confidence, 0.05)),
                "iou": self.nms_iou,
                "imgsz": self.image_size,
                "device": self.device,
                "save": False,
                "verbose": False,   # 减少日志
                "task": "obb",
            }
        )

    def predict(self, source: Any) -> list[dict]:
        source_arg = str(Path(source).resolve()) if isinstance(source, (str, Path)) else source
        results = list(self._predictor(source=source_arg))
        return [self._convert_result(result) for result in results]

    # 预测单张图片
    def predict_one(self, source: Any) -> dict:
        results = self.predict(source)
        if len(results) != 1:
            raise RuntimeError(f"Expected one image, received {len(results)} results")
        return results[0]

    @staticmethod
    def best_clear_candidate(frame_result: dict) -> dict | None:
        return next(
            (item for item in frame_result["detections"] if item["class_id"] == 1),
            None,
        )

    # 结果转换方法
    def _convert_result(self, result: Any) -> dict:
        height, width = map(int, result.orig_shape)
        detections: list[dict] = []
        if result.obb is not None and len(result.obb):
            rows = result.obb.data.detach().cpu().numpy()
            corners = result.obb.xyxyxyxy.detach().cpu().numpy()
            keypoints = result.keypoints.data.detach().cpu().numpy()
            for row, polygon, points in zip(rows, corners, keypoints):
                confidence = float(row[5])
                if confidence < self.confidence:
                    continue
                class_id = int(row[6])
                point_records = {}
                for name, point in zip(KEYPOINT_NAMES, points):
                    visibility = float(point[2])
                    point_records[name] = {
                        "xy_px": [float(point[0]), float(point[1])],
                        "visibility": visibility,
                        "visible": visibility >= self.keypoint_visibility,
                    }
                detections.append(
                    {
                        "id": 0,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "confidence": confidence,
                        "is_grasp_candidate": class_id == 1,
                        "obb": {
                            "center_px": [float(row[0]), float(row[1])],
                            "size_px": [float(row[2]), float(row[3])],
                            "angle_rad": float(row[4]),
                            "angle_deg": math.degrees(float(row[4])),
                            "corners_px": [[float(value) for value in point] for point in polygon],
                        },
                        "keypoints": point_records,
                    }
                )
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        for index, detection in enumerate(detections, start=1):
            detection["id"] = index
        best = self.best_clear_candidate({"detections": detections})
        return {
            "image_path": str(result.path),
            "width": width,
            "height": height,
            "detections": detections,
            "best_clear_candidate_id": best["id"] if best else None,
        }
