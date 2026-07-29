from contextlib import contextmanager
import time
from typing import Optional
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def log(message: str) -> None:
    print(message, flush=True)

@contextmanager
def timed_step(name: str):
    start_time = time.perf_counter()
    log(f"[box_object] START {name}")
    try:
        yield
    except Exception:
        log(f"[box_object] FAILED {name} after {time.perf_counter() - start_time:.2f}s")
        raise
    log(f"[box_object] DONE {name} in {time.perf_counter() - start_time:.2f}s")

def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()

def normalize_paths(args: SimpleNamespace) -> None:
    for attr in (
        "capture_root",
        "capture_dir",
        "ply",
        "image",
        "box_transform",
        "object_output_dir",
        "object_summary",
        "mask",
        "sam_task_config",
        "grasp_pickle",
        "bottle_template_ply",
        "bottle_stl",
        "completion_template_ply",
        "completion_adapointr_script",
        "completion_adapointr_checkpoint",
        "completion_bottle_template_ply",
        "object_model",
        "rtde_plan_out",
        "yolo_model",
        "summary_out",
    ):
        setattr(args, attr, resolve_path(getattr(args, attr)))

def load_homomat(path: Path, label: str) -> np.ndarray:
    homomat = np.asarray(np.loadtxt(path), dtype=float)
    if homomat.shape != (4, 4):
        raise ValueError(f"{label} transform must be 4x4, got {homomat.shape}: {path}")
    if not np.all(np.isfinite(homomat)):
        raise ValueError(f"{label} transform contains NaN or inf: {path}")
    return homomat

def homomat_to_pose(homomat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return homomat[:3, 3].copy(), homomat[:3, :3].copy()

