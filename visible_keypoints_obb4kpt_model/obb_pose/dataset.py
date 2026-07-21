# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unified label loader for OBB corners plus four keypoints."""

from __future__ import annotations

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import (
    get_hash,
    save_dataset_cache_file,
)
from ultralytics.utils import NUM_THREADS, TQDM
from ultralytics.utils.ops import segments2boxes


def _verify_unified(args):
    im_file, lb_file, prefix, num_cls, nkpt, ndim = args
    nm = nf = ne = nc = 0
    msg = ""
    segments = []
    try:
        with Image.open(im_file) as im:
            im.verify()
        with Image.open(im_file) as im:
            width, height = ImageOps.exif_transpose(im).size
        if width < 10 or height < 10:
            raise ValueError(f"image too small: {width}x{height}")
        if Path(lb_file).is_file():
            nf = 1
            rows = [line.split() for line in Path(lb_file).read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows:
                expected = 1 + 8 + nkpt * ndim
                if any(len(row) != expected for row in rows):
                    raise ValueError(f"each label row must have {expected} columns")
                raw = np.asarray(rows, dtype=np.float32)
                if not np.isfinite(raw).all():
                    raise ValueError("label contains NaN or Inf")
                cls = raw[:, 0]
                if cls.min() < 0 or cls.max() >= num_cls or not np.all(cls == np.floor(cls)):
                    raise ValueError(f"class id must be integer in [0,{num_cls - 1}]")
                corners = raw[:, 1:9].reshape(-1, 4, 2)
                keypoints = raw[:, 9:].reshape(-1, nkpt, ndim)
                # A mathematically exact min-area rectangle of a bottle truncated by the image boundary can
                # legitimately extend slightly outside the raster. Do not clip its corners, because clipping would
                # turn it into a non-rectangular polygon and violate the reviewed OBB. The conservative guard below
                # still catches gross coordinate corruption.
                if corners.min() < -0.25 or corners.max() > 1.25:
                    raise ValueError("OBB corners exceed the allowed normalized guard range [-0.25,1.25]")
                if keypoints[..., :2].min() < 0 or keypoints[..., :2].max() > 1:
                    raise ValueError("keypoint coordinates must be normalized to [0,1]")
                if ndim == 3:
                    vis = keypoints[..., 2]
                    if not np.isin(vis, [0, 1]).all():
                        raise ValueError("keypoint visibility must be 0 or 1")
                    if not np.all(keypoints[..., :2][vis == 0] == 0):
                        raise ValueError("invisible keypoints must use 0 0 0")
                if len(np.unique(raw, axis=0)) != len(raw):
                    raise ValueError("duplicate label row")
                segments = [x.astype(np.float32) for x in corners]
                boxes = segments2boxes(segments)
                lb = np.concatenate((cls[:, None], boxes), axis=1).astype(np.float32)
            else:
                ne = 1
                lb = np.zeros((0, 5), dtype=np.float32)
                keypoints = np.zeros((0, nkpt, ndim), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 5), dtype=np.float32)
            keypoints = np.zeros((0, nkpt, ndim), dtype=np.float32)
        return im_file, lb, (height, width), segments, keypoints, nm, nf, ne, nc, msg
    except Exception as exc:
        nc = 1
        return None, None, None, None, None, nm, nf, ne, nc, f"{prefix}{im_file}: {exc}"


class OBBPoseDataset(YOLODataset):
    """YOLO dataset that keeps OBB polygon and keypoints on the same instance row."""

    def cache_labels(self, path=Path("./labels.cache")):
        x = {"labels": []}
        nm = nf = ne = nc = 0
        msgs = []
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if nkpt <= 0 or ndim != 3:
            raise ValueError("dataset.yaml must contain kpt_shape: [4, 3]")
        iterable = zip(
            self.im_files,
            self.label_files,
            repeat(self.prefix),
            repeat(len(self.data["names"])),
            repeat(nkpt),
            repeat(ndim),
        )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(_verify_unified, iterable)
            pbar = TQDM(results, desc=f"{self.prefix}Scanning unified OBB+Pose labels", total=len(self.im_files))
            for im_file, lb, shape, segments, keypoints, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],
                            "bboxes": lb[:, 1:],
                            "segments": segments,
                            "keypoints": keypoints,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
        if msgs:
            raise RuntimeError("Invalid unified labels:\n" + "\n".join(msgs[:30]))
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def build_transforms(self, hyp=None):
        # LetterBox is required model input resizing, not random geometric augmentation.
        transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=True)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_keypoint=True,
                return_obb=True,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=0.0,
            )
        )
        return transforms
