"""Interactive point-prompt segmentation for the saved detergent-bottle image.

Controls in the OpenCV window:
  left click   add a foreground point
  right click  add a background point
  s            run segmentation and save outputs
  u            undo the last point
  r            reset all points
  q / Esc      quit

The script also supports non-interactive usage with --point x,y,label.
label can be 1/0, fg/bg, pos/neg, or +/-. 
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
DEFAULT_IMAGE = THIS_DIR / "captures" / "20260624-143449" / "rgb.png"
DEFAULT_RUN_DIR = THIS_DIR / "runs" / "point_hint_segment"

for config_dir in (DEFAULT_RUN_DIR / "config", DEFAULT_RUN_DIR / "mpl_config"):
    config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(DEFAULT_RUN_DIR / "config"))
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_RUN_DIR / "mpl_config"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import FastSAM, SAM  # noqa: E402


PointHint = tuple[int, int, int]
BoxHint = tuple[int, int, int, int]


def label_name(label: int) -> str:
    return "fg" if int(label) == 1 else "bg"


def format_point_hint(point: PointHint) -> str:
    x, y, label = point
    return f"{int(x)},{int(y)},{label_name(label)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment an object using manual point hints.")
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image. Default: {DEFAULT_IMAGE}",
    )
    parser.add_argument(
        "--backend",
        choices=("fastsam", "sam"),
        default="sam",
        help="Promptable segmentation backend. Default: sam",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model weights. Defaults to FastSAM-s.pt for fastsam and sam_b.pt for sam.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Output directory. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        metavar="X,Y,LABEL",
        help="Point hint in original-image pixels, e.g. --point 650,360,fg --point 900,360,bg",
    )
    parser.add_argument(
        "--box",
        default=None,
        metavar="X1,Y1,X2,Y2",
        help="Optional box hint around one target object, in original-image pixels.",
    )
    parser.add_argument(
        "--keep",
        choices=("best", "all", "largest", "smallest", "combined"),
        default="best",
        help="How to save multiple returned masks. Default: best",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="FastSAM confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.9, help="FastSAM NMS IoU threshold.")
    parser.add_argument("--device", default=None, help='Device, for example "0", "cpu", or empty for auto.')
    parser.add_argument("--max-display", type=int, default=1400, help="Largest displayed window side in pixels.")
    parser.add_argument("--no-gui", action="store_true", help="Do not open a window; require --point.")
    parser.add_argument(
        "--show-gui-with-points",
        action="store_true",
        help="When --point is provided, show those points in the interactive window instead of running immediately.",
    )
    return parser.parse_args()


def default_model(backend: str) -> str:
    if backend == "fastsam":
        local = REPO_DIR / "FastSAM-s.pt"
        return str(local if local.exists() else "FastSAM-s.pt")
    local = REPO_DIR / "sam_b.pt"
    return str(local if local.exists() else "sam_b.pt")


def parse_label(raw_label: str) -> int:
    label = raw_label.strip().lower()
    if label in {"1", "fg", "front", "pos", "positive", "+"}:
        return 1
    if label in {"0", "bg", "back", "neg", "negative", "-"}:
        return 0
    raise ValueError(f"Unknown point label '{raw_label}'. Use fg/bg or 1/0.")


def parse_point(raw_point: str) -> PointHint:
    parts = [part.strip() for part in raw_point.split(",")]
    if len(parts) not in {2, 3}:
        raise ValueError(f"Bad point '{raw_point}'. Expected X,Y or X,Y,LABEL.")
    x = int(round(float(parts[0])))
    y = int(round(float(parts[1])))
    label = parse_label(parts[2]) if len(parts) == 3 else 1
    return x, y, label


def parse_box(raw_box: str) -> BoxHint:
    parts = [part.strip() for part in raw_box.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Bad box '{raw_box}'. Expected X1,Y1,X2,Y2.")
    x1, y1, x2, y2 = [int(round(float(part))) for part in parts]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def clamp_points(points: Iterable[PointHint], width: int, height: int) -> list[PointHint]:
    clamped = []
    for x, y, label in points:
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        clamped.append((x, y, int(label)))
    return clamped


def clamp_box(box: BoxHint | None, width: int, height: int) -> BoxHint | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    x2 = max(0, min(width - 1, int(x2)))
    y1 = max(0, min(height - 1, int(y1)))
    y2 = max(0, min(height - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box is empty after clamping: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def load_model(backend: str, model_path: str):
    if backend == "fastsam":
        return FastSAM(model_path)
    return SAM(model_path)


def run_model(
    model,
    backend: str,
    image_path: Any,
    points: list[PointHint],    # 输入点
    box: BoxHint | None,    # 输入水平框
    imgsz: int,
    conf: float,
    iou: float,
    device: str | None,
):
    if not any(label == 1 for _, _, label in points):
        raise ValueError("At least one foreground point is required.")

    xy = [[x, y] for x, y, _ in points]
    labels = [label for _, _, label in points]
    source = image_path if isinstance(image_path, np.ndarray) else str(image_path)
    kwargs = dict(
        source=source,
        imgsz=imgsz,
        retina_masks=True,
        device=device,
        verbose=False,
    )

    if backend == "fastsam":
        kwargs.update(conf=conf, iou=iou, points=xy, labels=labels)
        if box is not None:
            kwargs["bboxes"] = [list(box)]
    else:
        # SAM expects multiple clicks for one object as shape (1, N, 2).
        kwargs.update(points=[xy], labels=[labels])
        if box is not None:
            kwargs["bboxes"] = [list(box)]

    return model.predict(**kwargs)[0]


def ensure_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    height, width = image_shape
    if mask_u8.shape[:2] != (height, width):
        mask_u8 = cv2.resize(mask_u8, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask_u8


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def score_mask(mask: np.ndarray, points: list[PointHint]) -> float:
    height, width = mask.shape[:2]
    image_area = height * width
    area = int((mask > 0).sum())
    if area == 0:
        return -1e9

    positive_points = [(x, y) for x, y, label in points if label == 1]
    negative_points = [(x, y) for x, y, label in points if label == 0]
    pos_hits = sum(1 for x, y in positive_points if mask[y, x] > 0)
    neg_hits = sum(1 for x, y in negative_points if mask[y, x] > 0)
    if positive_points and pos_hits == 0:
        return -1e9

    bbox = mask_bbox(mask)
    if bbox is None:
        return -1e9
    x1, y1, x2, y2 = bbox
    bbox_w = max(1, x2 - x1 + 1)
    bbox_h = max(1, y2 - y1 + 1)
    bbox_area = bbox_w * bbox_h
    area_ratio = area / image_area
    extent = area / bbox_area
    border_hits = int(x1 <= 2) + int(y1 <= 2) + int(x2 >= width - 3) + int(y2 >= height - 3)

    # FastSAM often returns a large box/bin candidate. Penalize that, but keep
    # enough tolerance for a detergent bottle lying across much of the box.
    score = 100.0 * pos_hits
    score -= 200.0 * neg_hits
    score -= 120.0 * max(0.0, area_ratio - 0.30)
    score -= 40.0 * max(0.0, area_ratio - 0.18)
    score -= 35.0 * max(0.0, extent - 0.82)
    score -= 15.0 * border_hits
    score -= 8.0 * abs(area_ratio - 0.10)
    return score


def select_masks(mask_arrays: list[np.ndarray], keep: str, points: list[PointHint]) -> list[np.ndarray]:
    if keep == "best":
        scores = [score_mask(mask, points) for mask in mask_arrays]
        best_idx = int(np.argmax(scores))
        return [mask_arrays[best_idx]] if scores[best_idx] > -1e8 else []
    if keep == "all":
        return mask_arrays
    areas = [int((mask > 0).sum()) for mask in mask_arrays]
    if keep == "largest":
        return [mask_arrays[int(np.argmax(areas))]]
    if keep == "smallest":
        positive = [(idx, area) for idx, area in enumerate(areas) if area > 0]
        if not positive:
            return []
        return [mask_arrays[min(positive, key=lambda item: item[1])[0]]]

    combined = np.zeros_like(mask_arrays[0], dtype=np.uint8)
    for mask in mask_arrays:
        combined = np.maximum(combined, mask)
    return [combined]


def draw_points(image_bgr: np.ndarray, points: list[PointHint], scale: float = 1.0) -> np.ndarray:
    canvas = image_bgr.copy()
    for x, y, label in points:
        px = int(round(x * scale))
        py = int(round(y * scale))
        color = (30, 220, 30) if label == 1 else (30, 30, 240)
        cv2.circle(canvas, (px, py), 8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
        if label == 0:
            cv2.line(canvas, (px - 6, py - 6), (px + 6, py + 6), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (px - 6, py + 6), (px + 6, py - 6), (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def make_overlay(
    image_bgr: np.ndarray,
    masks: list[np.ndarray],
    points: list[PointHint],
    box: BoxHint | None,
) -> np.ndarray:
    overlay = image_bgr.copy()
    colors = [
        (0, 180, 255),
        (80, 220, 80),
        (255, 120, 80),
        (180, 80, 255),
        (80, 255, 220),
    ]
    for idx, mask in enumerate(masks):
        color = np.array(colors[idx % len(colors)], dtype=np.uint8)
        mask_bool = mask > 0
        overlay[mask_bool] = (0.45 * overlay[mask_bool] + 0.55 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2, cv2.LINE_AA)
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 220, 40), 3, cv2.LINE_AA)
    return draw_points(overlay, points)


def safe_stem(parts: Iterable[object]) -> str:
    text = "_".join(str(part) for part in parts)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def save_outputs(
    result,
    image_bgr: np.ndarray,
    image_path: Any,
    out_dir: Path,
    points: list[PointHint],
    box: BoxHint | None,
    backend: str,
    model_path: str,
    keep: str,
) -> dict:
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = out_dir / run_name
    mask_dir = save_dir / "masks"
    cutout_dir = save_dir / "cutouts"
    mask_dir.mkdir(parents=True, exist_ok=True)
    cutout_dir.mkdir(parents=True, exist_ok=True)

    if result.masks is None:
        raw_masks: list[np.ndarray] = []
    else:
        raw_masks = [
            ensure_mask(mask, image_bgr.shape[:2])
            for mask in result.masks.data.cpu().numpy()
        ]
    selected_masks = select_masks(raw_masks, keep, points) if raw_masks else []

    detections = []
    for idx, mask in enumerate(selected_masks):
        area = int((mask > 0).sum())
        stem = safe_stem([f"{idx:02d}", f"area{area}"])
        mask_path = mask_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        cutout_bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        cutout_bgra[:, :, 3] = mask
        cutout_path = cutout_dir / f"{stem}_cutout.png"
        cv2.imwrite(str(cutout_path), cutout_bgra)

        detections.append(
            {
                "index": idx,
                "area_px": area,
                "mask_path": str(mask_path),
                "cutout_path": str(cutout_path),
            }
        )

    overlay = make_overlay(image_bgr, selected_masks, points, box)
    overlay_path = save_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    prompts = [{"x": x, "y": y, "label": int(label)} for x, y, label in points]
    summary = {
        "image": str(image_path),
        "backend": backend,
        "model": model_path,
        "keep": keep,
        "box": None if box is None else list(box),
        "points": prompts,
        "num_raw_masks": len(raw_masks),
        "num_saved_masks": len(selected_masks),
        "overlay_path": str(overlay_path),
        "detections": detections,
    }
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


class InteractiveSession:
    def __init__(self, args: argparse.Namespace, image_bgr: np.ndarray, model) -> None:
        self.args = args
        self.image_bgr = image_bgr
        self.model = model
        h, w = image_bgr.shape[:2]
        self.scale = min(1.0, args.max_display / max(w, h))
        self.display = cv2.resize(image_bgr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.points: list[PointHint] = clamp_points([parse_point(p) for p in args.point], w, h)
        self.last_overlay: np.ndarray | None = None
        self.last_summary: dict | None = None
        self.window = "point-hint segment"

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event not in {cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN}:
            return
        ox = int(round(x / self.scale))
        oy = int(round(y / self.scale))
        label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
        h, w = self.image_bgr.shape[:2]
        self.points = clamp_points([*self.points, (ox, oy, label)], w, h)
        px, py, plabel = self.points[-1]
        print(
            f"[point_hint] click #{len(self.points)}: "
            f"x={px}, y={py}, label={label_name(plabel)} "
            f"(--point {format_point_hint((px, py, plabel))})",
            flush=True,
        )
        self.refresh()

    def current_view(self) -> np.ndarray:
        if self.last_overlay is None:
            view = draw_points(self.display, self.points, scale=self.scale)
        else:
            overlay_small = cv2.resize(
                self.last_overlay,
                None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_AREA,
            )
            view = draw_points(overlay_small, self.points, scale=self.scale)

        help_lines = [
            "L-click fg | R-click bg | s segment/save | u undo | r reset | q quit",
            f"points: {len(self.points)}  backend: {self.args.backend}  keep: {self.args.keep}",
        ]
        y = 28
        for line in help_lines:
            cv2.putText(view, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            y += 30
        return view

    def refresh(self) -> None:
        cv2.imshow(self.window, self.current_view())

    def segment_and_save(self) -> None:
        box = clamp_box(parse_box(self.args.box), self.image_bgr.shape[1], self.image_bgr.shape[0]) if self.args.box else None
        _t0 = time.perf_counter()
        result = run_model(
            self.model,
            self.args.backend,
            self.args.image.resolve(),
            self.points,
            box,
            self.args.imgsz,
            self.args.conf,
            self.args.iou,
            self.args.device,
        )
        _t1 = time.perf_counter()
        summary = save_outputs(
            result,
            self.image_bgr,
            self.args.image.resolve(),
            self.args.out_dir.resolve(),
            self.points,
            box,
            self.args.backend,
            self.args.model,
            self.args.keep,
        )
        _t2 = time.perf_counter()
        print(f"[point_hint] SAM inference time: {_t1 - _t0:.3f}s")
        print(f"[point_hint] mask save time:    {_t2 - _t1:.3f}s")
        self.last_overlay = cv2.imread(summary["overlay_path"], cv2.IMREAD_COLOR)
        self.last_summary = summary
        print(f"raw masks:   {summary['num_raw_masks']}")
        print(f"saved masks: {summary['num_saved_masks']}")
        print(f"overlay:     {summary['overlay_path']}")
        print(f"summary:     {summary['summary_path']}")
        self.refresh()

    def run(self) -> dict | None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_mouse)
        if self.points:
            print(
                "[point_hint] preloaded points: "
                + " ".join(f"--point {format_point_hint(point)}" for point in self.points),
                flush=True,
            )
        self.refresh()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in {27, ord("q")}:
                break
            if key == ord("u") and self.points:
                self.points.pop()
                self.refresh()
            elif key == ord("r"):
                self.points.clear()
                self.last_overlay = None
                self.last_summary = None
                self.refresh()
            elif key == ord("s"):
                try:
                    self.segment_and_save()
                except Exception as exc:  # Keep the interactive window alive.
                    print(f"segmentation failed: {exc}")
        cv2.destroyAllWindows()
        return self.last_summary


def main() -> None:
    args = parse_args()
    args.image = args.image.resolve()
    args.out_dir = args.out_dir.resolve()
    args.model = args.model or default_model(args.backend)

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    h, w = image_bgr.shape[:2]
    points = clamp_points([parse_point(point) for point in args.point], w, h)
    box = clamp_box(parse_box(args.box), w, h) if args.box else None
    if args.no_gui and not points:
        raise ValueError("--no-gui requires at least one --point.")

    print(f"backend: {args.backend}")
    print(f"model:   {args.model}")
    print(f"image:   {args.image}")
    model = load_model(args.backend, args.model)

    if args.no_gui or (points and not args.show_gui_with_points):
        if not points:
            raise ValueError("No points were provided.")
        result = run_model(model, args.backend, args.image, points, box, args.imgsz, args.conf, args.iou, args.device)
        summary = save_outputs(
            result,
            image_bgr,
            args.image,
            args.out_dir,
            points,
            box,
            args.backend,
            args.model,
            args.keep,
        )
        print(f"raw masks:   {summary['num_raw_masks']}")
        print(f"saved masks: {summary['num_saved_masks']}")
        print(f"overlay:     {summary['overlay_path']}")
        print(f"summary:     {summary['summary_path']}")
        return

    session = InteractiveSession(args, image_bgr, model)
    session.run()


if __name__ == "__main__":
    main()





