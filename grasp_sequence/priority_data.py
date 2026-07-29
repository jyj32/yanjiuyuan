#!/usr/bin/env python3
"""RGB-D scene dataset using human-refined OBB/4KPT/ranking annotations."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


KEYPOINT_NAMES = ("C", "N", "L", "B")


def load_jsonl(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if split is None or row["split"] == split:
                rows.append(row)
    return rows


def letterbox(
    rgb: np.ndarray, depth: np.ndarray, image_size: int
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    height, width = rgb.shape[:2]
    scale = min(image_size / width, image_size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    pad_x = (image_size - new_width) / 2.0
    pad_y = (image_size - new_height) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))

    resized_rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    resized_depth = cv2.resize(
        depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST
    )
    rgb_canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    depth_canvas = np.zeros((image_size, image_size), dtype=np.float32)
    rgb_canvas[top : top + new_height, left : left + new_width] = resized_rgb
    depth_canvas[top : top + new_height, left : left + new_width] = resized_depth
    return rgb_canvas, depth_canvas, scale, float(left), float(top)


def transform_points(
    points: np.ndarray, scale: float, pad_x: float, pad_y: float
) -> np.ndarray:
    transformed = points.astype(np.float32, copy=True)
    transformed[..., 0] = transformed[..., 0] * scale + pad_x
    transformed[..., 1] = transformed[..., 1] * scale + pad_y
    return transformed


def geometry_from_quad_and_keypoints(
    quad: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    image_size: int,
) -> np.ndarray:
    center = quad.mean(axis=0)
    edge_a = quad[1] - quad[0]
    edge_b = quad[3] - quad[0]
    len_a = max(float(np.linalg.norm(edge_a)), 1e-6)
    len_b = max(float(np.linalg.norm(edge_b)), 1e-6)
    if len_a >= len_b:
        major, minor = edge_a / len_a, edge_b / len_b
        width, height = len_a, len_b
    else:
        major, minor = edge_b / len_b, edge_a / len_a
        width, height = len_b, len_a
    angle = math.atan2(float(major[1]), float(major[0]))
    raw = [
        float(center[0] / image_size),
        float(center[1] / image_size),
        float(width / image_size),
        float(height / image_size),
        math.sin(angle),
        math.cos(angle),
    ]
    for point, visible in zip(keypoints, visibility):
        if visible > 0.5:
            delta = point - center
            u = float(np.dot(delta, major) / width)
            v = float(np.dot(delta, minor) / height)
            raw.extend((u, v, 1.0))
        else:
            raw.extend((0.0, 0.0, 0.0))
    return np.asarray(raw, dtype=np.float32)


class PrioritySceneDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: int,
        depth_near_mm: float,
        depth_far_mm: float,
        augment: bool = False,
        limit: int | None = None,
        seed: int = 20260725,
    ):
        self.rows = load_jsonl(manifest_path, split)
        if limit is not None:
            rng = random.Random(seed)
            rows = list(self.rows)
            rng.shuffle(rows)
            self.rows = rows[:limit]
        self.image_size = image_size
        self.depth_near_mm = float(depth_near_mm)
        self.depth_far_mm = float(depth_far_mm)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rgb_bgr = cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)
        depth_mm = np.load(row["depth_path"]).astype(np.float32, copy=False)
        if rgb_bgr is None:
            raise RuntimeError(f"cannot read {row['rgb_path']}")
        if tuple(rgb_bgr.shape[:2]) != tuple(depth_mm.shape):
            raise RuntimeError(f"RGB/depth shape mismatch in {row['scene_id']}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        if self.augment:
            gain = random.uniform(0.90, 1.10)
            bias = random.uniform(-8.0, 8.0)
            rgb = np.clip(rgb.astype(np.float32) * gain + bias, 0, 255).astype(
                np.uint8
            )
        rgb, depth_mm, scale, pad_x, pad_y = letterbox(
            rgb, depth_mm, self.image_size
        )

        valid = np.isfinite(depth_mm) & (depth_mm > 0)
        clipped = np.clip(
            np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0),
            self.depth_near_mm,
            self.depth_far_mm,
        )
        depth_norm = np.zeros_like(depth_mm, dtype=np.float32)
        depth_norm[valid] = 0.05 + 0.95 * (
            clipped[valid] - self.depth_near_mm
        ) / (self.depth_far_mm - self.depth_near_mm)
        if self.augment and valid.any():
            noise = np.random.normal(0.0, 0.005, depth_norm.shape).astype(np.float32)
            depth_norm[valid] = np.clip(depth_norm[valid] + noise[valid], 0.05, 1.0)

        quads = []
        keypoints = []
        visibility = []
        geometry = []
        ranks = []
        instance_ids = []
        occlusion = []
        for obj in row["objects"]:
            quad = transform_points(
                np.asarray(obj["obb_corners_px"], dtype=np.float32),
                scale,
                pad_x,
                pad_y,
            )
            points = []
            visible_values = []
            for name in KEYPOINT_NAMES:
                point = obj["keypoints"][name]
                visible = float(point["point_visible"])
                if visible > 0.5:
                    xy = transform_points(
                        np.asarray(point["xy_px"], dtype=np.float32),
                        scale,
                        pad_x,
                        pad_y,
                    )
                else:
                    xy = np.zeros(2, dtype=np.float32)
                points.append(xy)
                visible_values.append(visible)
            points_array = np.asarray(points, dtype=np.float32)
            visible_array = np.asarray(visible_values, dtype=np.float32)
            quads.append(quad)
            keypoints.append(points_array)
            visibility.append(visible_array)
            geometry.append(
                geometry_from_quad_and_keypoints(
                    quad, points_array, visible_array, self.image_size
                )
            )
            ranks.append(int(obj["priority_rank"]))
            instance_ids.append(int(obj["instance_id"]))
            occlusion.append(int(obj["occlusion_state"]))

        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        rgb_tensor = rgb_tensor.float().div_(255.0)
        depth_tensor = torch.from_numpy(
            np.ascontiguousarray(
                np.stack((depth_norm, valid.astype(np.float32)), axis=0)
            )
        )
        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "quads": torch.from_numpy(np.asarray(quads, dtype=np.float32)),
            "keypoints": torch.from_numpy(np.asarray(keypoints, dtype=np.float32)),
            "visibility": torch.from_numpy(
                np.asarray(visibility, dtype=np.float32)
            ),
            "geometry": torch.from_numpy(np.asarray(geometry, dtype=np.float32)),
            "ranks": torch.tensor(ranks, dtype=torch.long),
            "instance_ids": torch.tensor(instance_ids, dtype=torch.long),
            "occlusion": torch.tensor(occlusion, dtype=torch.long),
            "scene_id": row["scene_id"],
        }


def priority_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rgb = torch.stack([sample["rgb"] for sample in samples])
    depth = torch.stack([sample["depth"] for sample in samples])
    batch_indices = []
    for batch_index, sample in enumerate(samples):
        batch_indices.append(
            torch.full(
                (sample["ranks"].numel(),), batch_index, dtype=torch.long
            )
        )
    return {
        "rgb": rgb,
        "depth": depth,
        "quads": torch.cat([sample["quads"] for sample in samples]),
        "keypoints": torch.cat([sample["keypoints"] for sample in samples]),
        "visibility": torch.cat([sample["visibility"] for sample in samples]),
        "geometry": torch.cat([sample["geometry"] for sample in samples]),
        "ranks": torch.cat([sample["ranks"] for sample in samples]),
        "instance_ids": torch.cat(
            [sample["instance_ids"] for sample in samples]
        ),
        "occlusion": torch.cat([sample["occlusion"] for sample in samples]),
        "candidate_batch_idx": torch.cat(batch_indices),
        "scene_ids": [sample["scene_id"] for sample in samples],
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return result


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)
