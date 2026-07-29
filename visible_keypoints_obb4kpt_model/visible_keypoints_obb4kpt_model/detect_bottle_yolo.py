#!/usr/bin/env python3
"""
瓶子 OBB + 4关键点检测器（类形式）。

检测方法直接接收 PIL Image 或 numpy 数组，无需图片路径。
main() 直接处理图片，不使用命令行参数。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
PACKAGE_ROOT = Path(__file__).resolve().parent  # 当前父目录
sys.path.insert(0, str(PACKAGE_ROOT))

from obb_pose import OBBPosePredictor

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


class BottleDetector:
    """瓶子旋转目标检测（OBB）与 4 关键点估计检测器。
    """

    CLASS_NAMES = {0: "occluded", 1: "clear"}
    CLASS_COLORS = {0: (232, 58, 64), 1: (20, 190, 104)}
    KEYPOINT_NAMES = ("C", "N", "L", "B")
    KEYPOINT_COLORS = {
        "C": (231, 63, 214),
        "N": (255, 145, 35),
        "L": (250, 202, 37),
        "B": (28, 190, 235),
    }

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def __init__(self,
        model_path:str = None,
        save_dir:str = None,
    ) -> None:
        self.weights = Path( PACKAGE_ROOT / "model" / "bottle_detect2.pt").resolve() if model_path is None else model_path
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        self.device = _auto_device()
        self.conf = 0.528
        self.iou = 0.70
        self.imgsz = 960    # 检测的图像尺寸
        self.visibility = 0.50
        self.save_dir = Path( PACKAGE_ROOT / "outputs").resolve() if save_dir is None else save_dir
        self._predictor = OBBPosePredictor(
            overrides={
                "model": str(self.weights),
                "conf": max(0.001, min(self.conf, 0.05)),
                "iou": self.iou,
                "imgsz": self.imgsz,
                "device": self.device,
                "save": False,
                "verbose": False,
                "task": "obb",
            }
        )

    # ------------------------------------------------------------------
    # 核心检测
    # ------------------------------------------------------------------
    def detect(
        self,
        image: Image.Image | np.ndarray,
        show: bool = False,
        save: bool = False,
    ) -> list[dict]:
        """对单张图片执行检测。

        Parameters
        ----------
        image : PIL.Image.Image | np.ndarray
            输入图片，可以是 PIL Image 或 numpy 数组（H, W, 3），RGB 或 BGR 均可。
        show : bool
            为 True 时弹出系统默认图片查看器显示渲染结果。
        save : bool
            为 True 时将渲染结果保存为 PNG 文件。

        Returns
        -------
        list[dict]
            检测结果列表，按置信度降序排列。每个元素包含::
                class_id, class_name, confidence, xywhr_px,
                corners_px, keypoints_px
        """
        pil_image = self._to_pil(image)
        source = np.array(pil_image)  # predictor 接收 numpy 数组
        results = list(self._predictor(source=source, stream=True))
        if not results:
            return []
        detections = self._parse_result(results[0])

        if show or save:
            rendered = self.render(pil_image, detections)
            if save:
                save_path = self._resolve_save_path(self.save_dir)
                rendered.save(str(save_path), format="PNG", compress_level=6)
                print(f"渲染结果已保存至: {save_path}")
            if show:
                rendered.show()

        return detections

    # ------------------------------------------------------------------
    # 可视化渲染
    # ------------------------------------------------------------------
    def render(self,
               image: Image.Image,
               detections: list[dict],
    ) -> Image.Image:
        """在图片上绘制 OBB 框、关键点和标签。"""
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas, "RGBA")

        top_clear = next(
            (i for i, d in enumerate(detections) if d["class_id"] == 1), None
        )

        for index, det in enumerate(detections):
            color = self.CLASS_COLORS[det["class_id"]]
            polygon = [tuple(p) for p in det["corners_px"]]
            closed = polygon + [polygon[0]]
            if index == top_clear:
                draw.line(closed, fill=(255, 255, 255, 245), width=14, joint="curve")
            draw.line(closed, fill=(*color, 255), width=8, joint="curve")

            visible_count = sum(
                p[2] >= self.visibility for p in det["keypoints_px"]
            )
            text = f"P{index + 1:02d} {det['class_name'].upper()} {det['confidence']:.2f} {visible_count}/4"
            if index == top_clear:
                text += " TOP"
            anchor = min(polygon, key=lambda p: p[1])
            x = max(12, min(int(anchor[0]), canvas.width - 520))
            y = max(12, int(anchor[1]) - 44)
            self._draw_label(draw, x, y, text, color)

            for name, point in zip(self.KEYPOINT_NAMES, det["keypoints_px"]):
                kp_x, kp_y, vis = point
                if vis < self.visibility:
                    continue
                kp_color = self.KEYPOINT_COLORS[name]
                radius = 11
                draw.ellipse(
                    (kp_x - radius, kp_y - radius, kp_x + radius, kp_y + radius),
                    fill=(*kp_color, 255),
                    outline=(255, 255, 255, 255),
                    width=4,
                )
                draw.text(
                    (kp_x + 14, kp_y - 17),
                    f"{name} {vis:.2f}",
                    font=_font(25),
                    fill=kp_color,
                    stroke_width=3,
                    stroke_fill=(10, 16, 24),
                )
        return canvas

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    @staticmethod
    def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image.astype(np.uint8), mode="RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _resolve_save_path(image_path: str | Path | None) -> Path:
        """根据 image_path 推导渲染图的保存路径。"""
        if image_path is not None:
            src = Path(image_path)
            return src.parent / f"{src.stem}_det.png"
        save_dir = PACKAGE_ROOT / "outputs" / "visualizations"
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / "det_result.png"

    def _parse_result(self, result: Any) -> list[dict]:
        # 计算检测结果
        detections: list[dict] = []
        if result.obb is not None and len(result.obb):
            rows = result.obb.data.detach().cpu().numpy()
            corners = result.obb.xyxyxyxy.detach().cpu().numpy()
            keypoints = result.keypoints.data.detach().cpu().numpy()
            for row, polygon, points in zip(rows, corners, keypoints):
                confidence = float(row[5])
                if confidence < self.conf:
                    continue
                class_id = int(row[6])
                # xywhr_px = [cx, cy, w, h, r]，OBB 面积 = w * h（与旋转角无关）
                obb_w = float(row[2])
                obb_h = float(row[3])
                obb_area = obb_w * obb_h
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": self.CLASS_NAMES[class_id],
                        "confidence": confidence,
                        "xywhr_px": [float(v) for v in row[:5]],
                        "obb_area": obb_area,   # obb矩形框面积
                        "corners_px": [[float(v) for v in p] for p in polygon], # obb矩形框
                        "keypoints_px": [[float(v) for v in p] for p in points],    # 关键点
                    }
                )
        # 排序：先按 class_id 降序（不遮挡clear=1 在前，遮挡occluded=0 在后），
        # 组内按 OBB 面积降序（面积大的在前）
        detections.sort(
            key=lambda item: (item["class_id"], item["obb_area"]),
            reverse=True,
        )
        return detections

    @staticmethod
    def _draw_label(
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        f = _font(28)
        box = draw.textbbox((x, y), text, font=f)
        draw.rounded_rectangle(
            (box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5),
            radius=7,
            fill=(*color, 235),
            outline=(255, 255, 255, 240),
            width=2,
        )
        draw.text((x, y), text, font=f, fill=(255, 255, 255, 255))


if __name__ == "__main__":

    output_dir = Path(PACKAGE_ROOT / "outputs").resolve()
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    detector = BottleDetector()
    image_path = r"E:\py_project\wrsrobot\wrs_v2\yanjiuyuan\images\Mech\color_image_20260623-173827.jpg"
    img = cv2.imread(image_path)

    detections = detector.detect(img, show=True, save=True)
    print(f"detections: {detections}")