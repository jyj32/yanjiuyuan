#!/usr/bin/env python3
"""Inference entrypoint for the decoupled RGB-D first-grasp priority network."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import torch
import torch.nn.functional as F

from priority_data import PrioritySceneDataset, move_batch, priority_collate
from priority_model import PriorityNetwork


KEYPOINT_NAMES = ("C", "N", "L", "B")
SUPPORTED_MODES = {"full_ranking", "top1_only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score bottle candidates from aligned RGB/depth plus OBB and four "
            "keypoints. The first perception network is not loaded."
        )
    )
    bundle_dir = Path(__file__).resolve().parent
    parser.add_argument("--input", required=True, help="Scene JSON or JSONL")
    parser.add_argument("--output", required=True, help="Output result JSON")
    parser.add_argument(
        "--checkpoint",
        default=str(bundle_dir / "best.pt"),
        help="Trained priority checkpoint",
    )
    parser.add_argument(
        "--config",
        default=str(bundle_dir / "deploy_config.json"),
        help="Deployment configuration JSON",
    )
    parser.add_argument(
        "--yolo",
        default=str(bundle_dir / "yolo11s.pt"),
        help="Bundled official YOLO11s architecture initializer",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, and so on",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of input scenes for a smoke test",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def load_scene_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("scenes"), list):
            rows = payload["scenes"]
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            raise ValueError("input JSON must be one scene, a scene list, or {scenes: []}")
    if not rows:
        raise ValueError("input contains no scenes")
    return rows


def resolve_data_path(raw: Any, base_dir: Path, field: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def normalize_scene(
    raw: dict[str, Any], base_dir: Path, scene_index: int
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"scene {scene_index} is not an object")
    scene = dict(raw)
    scene["scene_id"] = str(scene.get("scene_id") or f"scene_{scene_index:04d}")
    scene["rgb_path"] = resolve_data_path(
        scene.get("rgb_path"), base_dir, f"{scene['scene_id']}.rgb_path"
    )
    scene["depth_path"] = resolve_data_path(
        scene.get("depth_path"), base_dir, f"{scene['scene_id']}.depth_path"
    )
    scene["split"] = "deploy"
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"{scene['scene_id']} must contain at least one object")

    normalized_objects = []
    seen_instance_ids: set[int] = set()
    for object_index, raw_object in enumerate(objects, start=1):
        if not isinstance(raw_object, dict):
            raise ValueError(f"{scene['scene_id']} object {object_index} is invalid")
        obj = dict(raw_object)
        instance_id = int(obj.get("instance_id", object_index))
        if instance_id in seen_instance_ids:
            raise ValueError(
                f"{scene['scene_id']} has duplicate instance_id={instance_id}"
            )
        seen_instance_ids.add(instance_id)
        obj["instance_id"] = instance_id
        obj["priority_rank"] = int(obj.get("priority_rank", object_index))
        obj["occlusion_state"] = int(obj.get("occlusion_state", 0))

        quad = obj.get("obb_corners_px")
        if (
            not isinstance(quad, list)
            or len(quad) != 4
            or any(not isinstance(point, list) or len(point) != 2 for point in quad)
        ):
            raise ValueError(
                f"{scene['scene_id']} instance {instance_id}: "
                "obb_corners_px must be [[x,y] x 4]"
            )
        keypoints = obj.get("keypoints")
        if not isinstance(keypoints, dict):
            raise ValueError(
                f"{scene['scene_id']} instance {instance_id}: keypoints missing"
            )
        for name in KEYPOINT_NAMES:
            point = keypoints.get(name)
            if not isinstance(point, dict):
                raise ValueError(
                    f"{scene['scene_id']} instance {instance_id}: keypoint {name} missing"
                )
            visible = float(point.get("point_visible", 0))
            xy = point.get("xy_px")
            if visible > 0.5 and (
                not isinstance(xy, list) or len(xy) != 2
            ):
                raise ValueError(
                    f"{scene['scene_id']} instance {instance_id}: "
                    f"visible keypoint {name} needs xy_px=[x,y]"
                )
            if not isinstance(xy, list) or len(xy) != 2:
                point["xy_px"] = [0.0, 0.0]
            point["point_visible"] = 1.0 if visible > 0.5 else 0.0
        normalized_objects.append(obj)
    scene["objects"] = normalized_objects
    return scene


def load_model(
    checkpoint_path: Path,
    config_path: Path,
    yolo_path: Path,
    device: torch.device,
) -> tuple[PriorityNetwork, dict[str, Any], torch.Tensor | None, int]:
    for path in (checkpoint_path, config_path, yolo_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    mode = config.get("mode")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported deploy mode: {mode!r}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_config = checkpoint.get("config") or {}
    checkpoint_mode = checkpoint_config.get("mode")
    if checkpoint_mode != mode:
        raise RuntimeError(
            f"checkpoint mode {checkpoint_mode!r} != config mode {mode!r}"
        )
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint does not contain a model state dictionary")

    model = PriorityNetwork(str(yolo_path), int(config["image_size"]))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    prototype = checkpoint.get("prototype")
    if prototype is not None:
        prototype = F.normalize(prototype.float().to(device), dim=0)
    if mode == "top1_only" and prototype is None:
        raise RuntimeError("top1_only checkpoint does not contain prototype")
    if mode == "full_ranking" and prototype is not None:
        raise RuntimeError("full_ranking checkpoint unexpectedly contains prototype")
    return model, config, prototype, int(checkpoint.get("epoch", -1))


def infer_scenes(
    rows: list[dict[str, Any]],
    input_path: Path,
    model: PriorityNetwork,
    config: dict[str, Any],
    prototype: torch.Tensor | None,
    device: torch.device,
) -> list[dict[str, Any]]:
    normalized = [
        normalize_scene(row, input_path.parent, index)
        for index, row in enumerate(rows)
    ]
    with tempfile.TemporaryDirectory(prefix="priority_deploy_") as temp_dir:
        manifest_path = Path(temp_dir) / "deploy.jsonl"
        manifest_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in normalized
            ),
            encoding="utf-8",
        )
        dataset = PrioritySceneDataset(
            manifest_path=manifest_path,
            split="deploy",
            image_size=int(config["image_size"]),
            depth_near_mm=float(config["depth_near_mm"]),
            depth_far_mm=float(config["depth_far_mm"]),
            augment=False,
        )
        results = []
        mode = config["mode"]
        with torch.inference_mode():
            for index, row in enumerate(normalized):
                sample = dataset[index]
                batch = move_batch(priority_collate([sample]), device)
                output = model(batch)
                if mode == "full_ranking":
                    scores = output["scores"]
                    score_type = "priority_head_raw_score"
                else:
                    assert prototype is not None
                    scores = F.normalize(output["embeddings"], dim=1) @ prototype
                    score_type = "prototype_cosine_similarity"
                score_values = scores.detach().float().cpu().tolist()

                candidates = []
                for obj, score in zip(row["objects"], score_values):
                    candidates.append(
                        {
                            "instance_id": int(obj["instance_id"]),
                            "priority_score": float(score),
                            "occlusion_state": int(obj["occlusion_state"]),
                        }
                    )
                candidates.sort(
                    key=lambda item: (-item["priority_score"], item["instance_id"])
                )
                for predicted_rank, candidate in enumerate(candidates, start=1):
                    candidate["predicted_priority_rank"] = predicted_rank
                results.append(
                    {
                        "scene_id": row["scene_id"],
                        "mode": mode,
                        "score_type": score_type,
                        "selected_instance_id": candidates[0]["instance_id"],
                        "candidate_count": len(candidates),
                        "priority_order": candidates,
                    }
                )
    return results


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    yolo_path = Path(args.yolo).resolve()
    device = resolve_device(args.device)

    rows = load_scene_rows(input_path)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        rows = rows[: args.limit]
    model, config, prototype, checkpoint_epoch = load_model(
        checkpoint_path, config_path, yolo_path, device
    )
    results = infer_scenes(
        rows, input_path, model, config, prototype, device
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": config["version_name"],
        "mode": config["mode"],
        "checkpoint_epoch": checkpoint_epoch,
        "device": str(device),
        "scene_count": len(results),
        "results": results,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": config["mode"],
                "device": str(device),
                "scene_count": len(results),
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
