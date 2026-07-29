# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO11 OBB+Pose model construction."""

from __future__ import annotations

from ultralytics.nn import tasks
from ultralytics.nn.tasks import OBBModel, yaml_model_load

from .head import OBBPoseHead
from .loss import OBBPoseLoss


class OBBPoseModel(OBBModel):
    """Build YOLO11s-OBB with a joint four-keypoint branch in the same head."""

    def __init__(self, cfg="yolo11s-obb.yaml", ch=3, nc=2, data_kpt_shape=(4, 3), verbose=True):
        cfg_dict = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        cfg_dict["kpt_shape"] = list(data_kpt_shape)
        OBBPoseHead.default_kpt_shape = tuple(data_kpt_shape)
        original_obb = tasks.OBB
        tasks.OBB = OBBPoseHead
        try:
            super().__init__(cfg=cfg_dict, ch=ch, nc=nc, verbose=verbose)
        finally:
            tasks.OBB = original_obb
        self.kpt_shape = tuple(data_kpt_shape)

    def init_criterion(self):
        return OBBPoseLoss(self)
