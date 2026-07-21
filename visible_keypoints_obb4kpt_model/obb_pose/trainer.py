# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ultralytics trainer wiring for the joint YOLO11 model."""

from copy import copy

from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.utils import RANK, colorstr

from .dataset import OBBPoseDataset
from .model import OBBPoseModel
from .validator import OBBPoseValidator


class OBBPoseTrainer(OBBTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(self.model.stride.max() if self.model else 0), 32)
        return OBBPoseDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task="obb",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = OBBPoseModel(
            cfg or "yolo11s-obb.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.kpt_shape = tuple(self.data["kpt_shape"])
        self.model.kpt_names = self.data.get("kpt_names", ["C", "N", "L", "B"])

    def get_validator(self):
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "angle_loss",
            "pose_loss",
            "kobj_loss",
        )
        return OBBPoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
