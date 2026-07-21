"""YOLO11 OBB + 4-keypoint unified training extension."""

from .model import OBBPoseModel
from .predictor import OBBPosePredictor
from .trainer import OBBPoseTrainer
from .validator import OBBPoseValidator

__all__ = ("OBBPoseModel", "OBBPosePredictor", "OBBPoseTrainer", "OBBPoseValidator")
