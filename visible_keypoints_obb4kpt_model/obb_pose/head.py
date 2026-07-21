# Derived from Ultralytics 8.4.19 OBB and Pose heads.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A single YOLO head that binds OBB, class, and four keypoints to one instance."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import OBB, Pose


class OBBPoseHead(OBB):
    """YOLO11 OBB head with a per-anchor semantic-keypoint branch.

    Inference channel order is ``xywh + class scores + keypoints + angle``.
    Keeping angle last preserves compatibility with Ultralytics rotated NMS.
    """

    default_kpt_shape = (4, 3)

    def __init__(self, nc=2, ne=1, reg_max=16, end2end=False, ch=()):
        super().__init__(nc=nc, ne=ne, reg_max=reg_max, end2end=end2end, ch=ch)
        self.kpt_shape = tuple(self.default_kpt_shape)
        self.nk = self.kpt_shape[0] * self.kpt_shape[1]
        c5 = max(ch[0] // 4, self.nk)
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nk, 1)) for x in ch
        )
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4, pose_head=self.cv5)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            angle_head=self.one2one_cv4,
            pose_head=self.one2one_cv5,
        )

    def forward_head(self, x, box_head, cls_head, angle_head, pose_head):
        preds = OBB.forward_head(self, x, box_head, cls_head, angle_head)
        if pose_head is not None:
            bs = x[0].shape[0]
            preds["kpts"] = torch.cat(
                [pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], dim=2
            )
        return preds

    def _inference(self, x):
        obb = OBB._inference(self, x)
        angle = obb[:, -self.ne :]
        base = obb[:, : -self.ne]
        kpts = Pose.kpts_decode(self, x["kpts"])
        return torch.cat((base, kpts, angle), dim=1)

    def postprocess(self, preds):
        """Post-process end-to-end outputs while retaining instance-bound keypoints."""
        boxes, scores, kpts, angle = preds.split([4, self.nc, self.nk, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat((boxes, scores, conf, kpts, angle), dim=-1)

    def fuse(self):
        self.cv2 = self.cv3 = self.cv4 = self.cv5 = None
