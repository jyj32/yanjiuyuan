# SPDX-License-Identifier: AGPL-3.0-or-later
"""OBB validation that safely separates keypoints from the rotated angle."""

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.obb import OBBValidator


class OBBPoseValidator(OBBValidator):
    def postprocess(self, preds):
        outputs = DetectionValidator.postprocess(self, preds)
        for pred in outputs:
            extra = pred.pop("extra")
            if extra.shape[1] < 1:
                raise RuntimeError("model output is missing the OBB angle")
            pred["keypoints"] = extra[:, :-1].reshape(-1, 4, 3)
            pred["bboxes"] = __import__("torch").cat((pred["bboxes"], extra[:, -1:]), dim=-1)
        return outputs
