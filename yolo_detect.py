"""
YOLO 瓶子目标检测 —— 输出水平框（axis-aligned bounding box）信息

用法:
  python yolo_detect.py --image path/to/image.jpg
  python yolo_detect.py --image-dir yanjiuyuan/images/Mech
  python yolo_detect.py --image path/to/image.jpg --model yolov8n.pt --conf 0.35
"""

import os
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO


# COCO 数据集中 "bottle" 的类别 ID
COCO_BOTTLE_CLASS_ID = 39


class BottleDetector:
    """YOLO 瓶子检测器，输出水平框信息"""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        save_dir: Optional[str] = None,

    ):
        """
        Args:
            model_path: YOLO 模型路径 (.pt)。传入官方预训练模型 (如 yolov8n.pt) 时
                        会自动按 COCO bottle 类别过滤；传入自训练模型时需指定 bottle_class_id
            conf_threshold: 置信度阈值
            iou_threshold: NMS 的 IoU 阈值
            bottle_class_id: 自训练模型中瓶子对应的类别 ID。
                             None 时默认使用 COCO 的 bottle (39)。
            device: 推理设备，"0" 表示 GPU，"cpu" 表示 CPU
        """
        print(f"正在加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = self.model.names
        self.save_dir = save_dir
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.base_dir, "images", "Mech") if save_dir is None else save_dir
        print(f"[BottleDetector] 模型加载成功 | 类别数: {len(self.class_names)} | ")
        # 预热
        dummy = np.zeros((960, 960, 3), dtype=np.uint8)
        _ = self.model.predict(dummy, verbose=False)

    # ------------------------------------------------------------------
    #  核心检测
    # ------------------------------------------------------------------

    def detect_bottle(self,image_0, image_path, show=False, save=False):
        """
        输入图像或从文件路径加载图像并检测。

        Returns:
            (detections, image_bgr)
        """
        if image_0 is None:
            if image_path is None or image_path == "":
                print("yolo检测无图片输入")
                return None
            else:
                image = cv2.imread(str(image_path))
        else:
            image = image_0
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        # yolo检测
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        # 解析yolo输出
        detections = self._parse_results(results[0])
        if show or save:
            vis = self.draw_detections(image, detections)
            if show:
                cv2.imshow("bottle_detection", vis)
                cv2.waitKey(1000)   # 展示1s自动关闭
                cv2.destroyAllWindows()
            if save:
                save_name = f"yolo_detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                save_path = os.path.join(self.save_dir, save_name)
                cv2.imwrite(save_path, vis)
                print(f"[BottleDetector] 检测结果已保存: {save_path}")

        return detections

    # ------------------------------------------------------------------
    #  解析 YOLO 输出
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_results(result) -> list[dict]:
        """将 YOLO 原始输出解析为水平框信息列表"""
        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
            cx, cy, w, h = box.xywh[0].cpu().numpy().astype(float)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, str(cls_id))

            detections.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)], # 左上（x1,y1）， 右下(x2,y2)
                    "bbox_xywh": [float(cx), float(cy), float(w), float(h)],    # 中心点（cx,cy）-宽w, 高h
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "area": float(w * h),
                }
            )
        detections.sort(key=lambda d: d["area"], reverse=True)  # 按面积大小降序排列
        return detections

    # ------------------------------------------------------------------
    #  可视化
    # ------------------------------------------------------------------
    @staticmethod
    def draw_detections(
            image: np.ndarray,
            detections: list[dict],
            line_width: int = 2,
    ) -> np.ndarray:
        """在图像上绘制水平框，并标注面积排名（#1, #2, ...）"""
        vis = image.copy()
        for rank, det in enumerate(detections, start=1):  # rank 从1开始
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            label = f"#{rank} bottle {conf:.2f}"  # 标注排名

            # 绿色水平框
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), line_width)

            # 标签背景
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                vis,
                (x1, y1 - th - baseline - 4),
                (x1 + tw, y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                vis,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return vis



if __name__ == "__main__":

    # ====== 配置区（直接修改这里） ======
    IMAGE_DIR = str(Path(__file__).parent / "images" / "Mech")   # 图像目录
    IMAGE_PATH = None                                            # 单张图像路径，设为 None 则用 IMAGE_DIR
    MODEL_PATH = str(Path(__file__).parent / "models" / "bottle_detect.pt")  # YOLO 模型路径

    # ---------- 初始化检测器 ----------
    detector = BottleDetector(
        model_path=MODEL_PATH,
        conf_threshold=0.7,
        iou_threshold=0.5,
    )

    save_dir = str(Path(__file__).parent / "images" / "Mech")
    img_path = str(Path(__file__).parent / "images" / "Mech"/"rgb.png")
    img = cv2.imread(img_path)
    detections = detector.detect_bottle(img, "", True, True)
    print("detections:", detections)
    for d in detections:
        print(d['bbox'])    # 矩形信息

