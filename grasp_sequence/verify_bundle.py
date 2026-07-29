#!/usr/bin/env python3
"""Fail-closed integrity and strict-load verification for a deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from priority_model import PriorityNetwork


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Verify deployment bundle hashes and strict model loading"
    )
    parser.add_argument(
        "--manifest", default=str(root / "bundle_manifest.json")
    )
    parser.add_argument("--config", default=str(root / "deploy_config.json"))
    parser.add_argument("--checkpoint", default=str(root / "best.pt"))
    parser.add_argument("--yolo", default=str(root / "yolo11s.pt"))
    parser.add_argument(
        "--code-dir",
        default=str(root),
        help="Directory containing model/data/inference Python files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    yolo_path = Path(args.yolo).resolve()
    code_dir = Path(args.code_dir).resolve()
    for path in (
        config_path,
        checkpoint_path,
        yolo_path,
        code_dir / "priority_model.py",
        code_dir / "priority_data.py",
        code_dir / "infer_priority.py",
    ):
        require(path.is_file(), f"required file missing: {path.name}")

    checkpoint_sha = sha256(checkpoint_path)
    yolo_sha = sha256(yolo_path)
    require(
        checkpoint_sha == manifest["checkpoint_sha256"],
        "best.pt SHA256 mismatch",
    )
    require(
        yolo_sha == manifest["yolo11s_sha256"],
        "yolo11s.pt SHA256 mismatch",
    )

    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    require(config["mode"] == manifest["mode"], "deploy_config mode mismatch")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    require(
        (checkpoint.get("config") or {}).get("mode") == manifest["mode"],
        "checkpoint mode mismatch",
    )
    require(
        int(checkpoint.get("epoch", -1)) == manifest["checkpoint_epoch"],
        "checkpoint epoch mismatch",
    )
    state_dict = checkpoint.get("model")
    require(isinstance(state_dict, dict), "checkpoint model state missing")
    require(
        len(state_dict) == manifest["state_tensor_count"],
        "checkpoint tensor count mismatch",
    )
    prototype = checkpoint.get("prototype")
    if manifest["mode"] == "top1_only":
        require(prototype is not None, "top1_only prototype missing")
        require(tuple(prototype.shape) == (256,), "prototype shape mismatch")
        prototype_norm = float(prototype.float().norm())
        require(abs(prototype_norm - 1.0) < 1e-4, "prototype is not normalized")
    else:
        require(prototype is None, "full_ranking must not contain prototype")
        prototype_norm = None

    model = PriorityNetwork(str(yolo_path), int(config["image_size"]))
    model.load_state_dict(state_dict, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    require(
        parameter_count == manifest["parameter_count"],
        "model parameter count mismatch",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "bundle": manifest["bundle_name"],
                "mode": manifest["mode"],
                "checkpoint_epoch": manifest["checkpoint_epoch"],
                "checkpoint_sha256": checkpoint_sha,
                "yolo11s_sha256": yolo_sha,
                "state_tensor_count": len(state_dict),
                "parameter_count": parameter_count,
                "prototype_norm": prototype_norm,
                "strict_load": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
