# Derived from Ultralytics 8.4.19 OBB and Pose losses.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Joint rotated-box, occlusion-class, keypoint-coordinate, and visibility loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.utils.loss import KeypointLoss, v8OBBLoss
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.tal import make_anchors


class OBBPoseLoss(v8OBBLoss):
    """OBB loss plus four-keypoint location and visibility losses."""

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.kpt_shape = tuple(model.model[-1].kpt_shape)
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_coco_pose = list(self.kpt_shape) == [17, 3]
        nkpt = self.kpt_shape[0]
        sigmas = (
            torch.from_numpy(OKS_SIGMA).to(self.device)
            if is_coco_pose
            else torch.ones(nkpt, device=self.device) / nkpt
        )
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def _select_target_keypoints(self, keypoints, batch_idx, target_gt_idx, masks):
        batch_idx = batch_idx.flatten().long()
        batch_size = len(masks)
        if keypoints.shape[0] == 0:
            return torch.zeros(
                (batch_size, masks.shape[1], *self.kpt_shape), device=keypoints.device, dtype=keypoints.dtype
            )
        counts = torch.bincount(batch_idx, minlength=batch_size)
        max_kpts = int(counts.max().item())
        batched = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
            device=keypoints.device,
            dtype=keypoints.dtype,
        )
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched[i, : keypoints_i.shape[0]] = keypoints_i
        gather_idx = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
        return batched.gather(1, gather_idx.expand(-1, -1, keypoints.shape[1], keypoints.shape[2]))

    def _keypoint_losses(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        selected = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        zero = pred_kpts.sum() * 0.0
        if not masks.any():
            return zero, zero
        gt_kpt = selected[masks]
        pred_kpt = pred_kpts[masks]
        kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.ones_like(gt_kpt[..., 0]).bool()
        area = (target_bboxes[masks][:, 2] * target_bboxes[masks][:, 3]).unsqueeze(-1).clamp_min(1e-9)
        coord = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
        visible = self.bce_pose(pred_kpt[..., 2], kpt_mask.float()) if pred_kpt.shape[-1] == 3 else zero
        return coord, visible

    @staticmethod
    def _decode_keypoints(anchor_points, pred_kpts):
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def loss(self, preds, batch):
        """Return six losses: box, class, DFL, angle, keypoint coordinate, keypoint visibility."""
        loss = torch.zeros(6, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        pred_angle = preds["angle"].permute(0, 2, 1).contiguous()
        pred_kpts_raw = preds["kpts"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]
        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        batch_idx_all = batch["batch_idx"].view(-1, 1)
        targets_all = torch.cat((batch_idx_all, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
        rw = targets_all[:, 4] * float(imgsz[1])
        rh = targets_all[:, 5] * float(imgsz[0])
        valid = (rw >= 2) & (rh >= 2)
        targets = targets_all[valid]
        batch_idx = batch_idx_all[valid]
        keypoints = batch["keypoints"].to(self.device).float().clone()[valid]
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 5), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)
        bboxes_for_assigner = pred_bboxes.detach().clone()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        pred_kpts = self._decode_keypoints(
            anchor_points, pred_kpts_raw.view(batch_size, -1, *self.kpt_shape)
        )
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            loss[4], loss[5] = self._keypoint_losses(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch_idx,
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
        else:
            loss[0] += pred_angle.sum() * 0.0
            loss[4] += pred_kpts_raw.sum() * 0.0

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.hyp.angle
        loss[4] *= self.hyp.pose
        loss[5] *= self.hyp.kobj
        return loss * batch_size, loss.detach()
