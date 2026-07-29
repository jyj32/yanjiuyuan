#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from obb_pose import OBBPosePredictor


CLASS_NAMES = {0: "occluded", 1: "clear"}
CLASS_COLORS = {0: (232, 58, 64), 1: (20, 190, 104)}
KEYPOINT_NAMES = ("C", "N", "L", "B")
KEYPOINT_COLORS = {
    "C": (231, 63, 214),
    "N": (255, 145, 35),
    "L": (250, 202, 37),
    "B": (28, 190, 235),
}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def auto_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    f = font(28)
    box = draw.textbbox((x, y), text, font=f)
    draw.rounded_rectangle(
        (box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5),
        radius=7,
        fill=(*color, 235),
        outline=(255, 255, 255, 240),
        width=2,
    )
    draw.text((x, y), text, font=f, fill=(255, 255, 255, 255))


def render(image_path: Path, detections: list[dict], visibility_threshold: float) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    top_clear = next((index for index, item in enumerate(detections) if item["class_id"] == 1), None)

    for index, detection in enumerate(detections):
        color = CLASS_COLORS[detection["class_id"]]
        polygon = [tuple(point) for point in detection["corners_px"]]
        closed = polygon + [polygon[0]]
        if index == top_clear:
            draw.line(closed, fill=(255, 255, 255, 245), width=14, joint="curve")
        draw.line(closed, fill=(*color, 255), width=8, joint="curve")

        visible_count = sum(point[2] >= visibility_threshold for point in detection["keypoints_px"])
        text = f"P{index + 1:02d} {detection['class_name'].upper()} {detection['confidence']:.2f} {visible_count}/4"
        if index == top_clear:
            text += " TOP"
        anchor = min(polygon, key=lambda point: point[1])
        x = max(12, min(int(anchor[0]), image.width - 520))
        y = max(12, int(anchor[1]) - 44)
        label(draw, x, y, text, color)

        for name, point in zip(KEYPOINT_NAMES, detection["keypoints_px"]):
            x, y, visibility = point
            if visibility < visibility_threshold:
                continue
            kp_color = KEYPOINT_COLORS[name]
            radius = 11
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(*kp_color, 255),
                outline=(255, 255, 255, 255),
                width=4,
            )
            draw.text(
                (x + 14, y - 17),
                f"{name} {visibility:.2f}",
                font=font(25),
                fill=kp_color,
                stroke_width=3,
                stroke_fill=(10, 16, 24),
            )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Image file or image directory")
    parser.add_argument("--output", default=str(PACKAGE_ROOT / "outputs"))
    parser.add_argument("--weights", default=str(PACKAGE_ROOT / "model" / "bottle_detect2.pt"))
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, 0, 1, ...")
    parser.add_argument("--conf", type=float, default=0.528)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--visibility", type=float, default=0.50)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    output_images = output / "visualizations"
    if not source.exists():
        raise FileNotFoundError(source)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    output_images.mkdir(parents=True, exist_ok=True)

    device = auto_device() if args.device == "auto" else args.device
    predictor = OBBPosePredictor(
        overrides={
            "model": str(weights),
            "source": str(source),
            "conf": max(0.001, min(args.conf, 0.05)),
            "iou": args.iou,
            "imgsz": args.imgsz,
            "device": device,
            "save": False,
            "verbose": False,
            "task": "obb",
        }
    )
    results = list(predictor(source=str(source)))
    records: list[dict] = []

    for image_index, result in enumerate(results, start=1):
        image_path = Path(result.path)
        detections: list[dict] = []
        if result.obb is not None and len(result.obb):
            rows = result.obb.data.detach().cpu().numpy()
            corners = result.obb.xyxyxyxy.detach().cpu().numpy()
            keypoints = result.keypoints.data.detach().cpu().numpy()
            for row, polygon, points in zip(rows, corners, keypoints):
                confidence = float(row[5])
                if confidence < args.conf:
                    continue
                class_id = int(row[6])
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "confidence": confidence,
                        "xywhr_px": [float(value) for value in row[:5]],
                        "corners_px": [[float(value) for value in point] for point in polygon],
                        "keypoints_px": [[float(value) for value in point] for point in points],
                    }
                )
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        visualization = render(image_path, detections, args.visibility)
        visualization_name = f"{image_index:03d}_{image_path.stem}.png"
        visualization.save(output_images / visualization_name, format="PNG", compress_level=6)
        records.append(
            {
                "image": str(image_path),
                "width": visualization.width,
                "height": visualization.height,
                "detections": detections,
                "visualization": str(Path("visualizations") / visualization_name),
            }
        )

    payload = {
        "model": str(weights),
        "strategy": "retain visible keypoints on occluded targets",
        "device": device,
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "nms_iou_threshold": args.iou,
        "keypoint_visibility_threshold": args.visibility,
        "class_names": CLASS_NAMES,
        "keypoint_names": KEYPOINT_NAMES,
        "image_count": len(records),
        "results": records,
    }
    result_path = output / "predictions.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image_count": len(records), "output": str(output), "predictions": str(result_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
