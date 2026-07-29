# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prediction result construction for OBB + four keypoints."""

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops


class OBBPosePredictor(DetectionPredictor):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        if cfg is None:
            from ultralytics.utils import DEFAULT_CFG

            cfg = DEFAULT_CFG
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.args.task = "obb"

    def construct_result(self, pred, img, orig_img, img_path):
        if pred.shape[1] < 19:
            raise RuntimeError(f"expected OBB+4KPT output with >=19 columns, got {pred.shape[1]}")
        angle = pred[:, -1:]
        rboxes = torch.cat((pred[:, :4], angle), dim=-1)
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        obb = torch.cat((rboxes, pred[:, 4:6]), dim=-1)
        keypoints = pred[:, 6:-1].reshape(-1, 4, 3)
        keypoints[..., :2] = ops.scale_coords(img.shape[2:], keypoints[..., :2], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, obb=obb, keypoints=keypoints)
