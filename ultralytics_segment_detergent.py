"""Try Ultralytics instance segmentation on the saved detergent-bottle image.

The COCO pretrained segmentation models know the generic "bottle" class, not
the specific detergent bottle category. Use this script for a quick baseline
and for generating visual pre-labels before training a custom model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = THIS_DIR / "captures" / "20260624-130254" / "rgb.png"
DEFAULT_RUN_DIR = THIS_DIR / "runs" / "ultralytics_seg"

# Keep Ultralytics/Matplotlib settings inside the workspace instead of AppData.
os.environ.setdefault("YOLO_CONFIG_DIR", str(DEFAULT_RUN_DIR / "config"))
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_RUN_DIR / "mpl_config"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Ultralytics segmentation model on a detergent-bottle image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image. Default: {DEFAULT_IMAGE}",
    )
    parser.add_argument(
        "--model",
        default="yolo11n-seg.pt",
        help="Ultralytics segmentation model or local .pt path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Output directory. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument(
        "--classes",
        default="bottle",
        help='Comma-separated class names/ids to keep, or "all". Default: bottle',
    )
    parser.add_argument(
        "--retry-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the class filter finds no masks, retry without filtering classes.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Inference device, for example "0", "cpu", or leave empty for auto.',
    )
    return parser.parse_args()


def class_filter(model: YOLO, classes_arg: str) -> list[int] | None:
    if not classes_arg or classes_arg.strip().lower() == "all":
        return None

    names = model.names
    name_to_id = {str(name).lower(): int(idx) for idx, name in names.items()}
    class_ids: list[int] = []

    for raw_item in classes_arg.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if item.isdigit():
            class_ids.append(int(item))
            continue
        key = item.lower()
        if key not in name_to_id:
            available = ", ".join(str(v) for _, v in sorted(names.items())[:20])
            raise ValueError(
                f"Unknown class '{item}'. Use an id, 'all', or one of the model names. "
                f"First available names: {available}, ..."
            )
        class_ids.append(name_to_id[key])

    return sorted(set(class_ids)) or None


def run_predict(
    model: YOLO,
    image_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
    classes: list[int] | None,
    device: str | None,
):
    return model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=classes,
        retina_masks=True,
        device=device,
        verbose=False,
    )[0]


def ensure_uint8_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    height, width = image_shape
    if mask_u8.shape[:2] != (height, width):
        mask_u8 = cv2.resize(mask_u8, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask_u8


def safe_name(parts: Iterable[object]) -> str:
    text = "_".join(str(part) for part in parts)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def save_outputs(result, image_path: Path, out_dir: Path, model_name: str, classes_arg: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks"
    cutout_dir = out_dir / "cutouts"
    mask_dir.mkdir(exist_ok=True)
    cutout_dir.mkdir(exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    overlay_path = out_dir / "overlay.png"
    overlay = result.plot()
    cv2.imwrite(str(overlay_path), overlay)

    detections = []
    boxes = result.boxes
    masks = result.masks
    mask_arrays = [] if masks is None else masks.data.cpu().numpy()
    names = result.names

    for idx, mask in enumerate(mask_arrays):
        cls_id = int(boxes.cls[idx].item()) if boxes is not None else -1
        conf = float(boxes.conf[idx].item()) if boxes is not None else 0.0
        xyxy = boxes.xyxy[idx].cpu().numpy().round(2).tolist() if boxes is not None else None
        cls_name = names.get(cls_id, str(cls_id))

        stem = safe_name([f"{idx:02d}", cls_name, f"{conf:.2f}"])
        mask_u8 = ensure_uint8_mask(mask, image_bgr.shape[:2])
        mask_path = mask_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), mask_u8)

        cutout_bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        cutout_bgra[:, :, 3] = mask_u8
        cutout_path = cutout_dir / f"{stem}_cutout.png"
        cv2.imwrite(str(cutout_path), cutout_bgra)

        detections.append(
            {
                "index": idx,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox_xyxy": xyxy,
                "mask_path": str(mask_path),
                "cutout_path": str(cutout_path),
            }
        )

    summary = {
        "image": str(image_path),
        "model": model_name,
        "classes_filter": classes_arg,
        "overlay_path": str(overlay_path),
        "num_instances": len(detections),
        "detections": detections,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    out_dir = args.out_dir.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(args.model)
    classes = class_filter(model, args.classes)
    result = run_predict(model, image_path, args.imgsz, args.conf, args.iou, classes, args.device)

    if args.retry_all and result.masks is None and classes is not None:
        print(f"No masks found for classes={args.classes!r}; retrying with all classes.")
        result = run_predict(model, image_path, args.imgsz, args.conf, args.iou, None, args.device)

    summary = save_outputs(result, image_path, out_dir, args.model, args.classes)
    print(f"instances: {summary['num_instances']}")
    print(f"overlay:   {summary['overlay_path']}")
    print(f"summary:   {summary['summary_path']}")
    if summary["num_instances"] == 0:
        print("No instances found. Try --classes all, lower --conf, or fine-tune on detergent images.")


if __name__ == "__main__":
    main()
