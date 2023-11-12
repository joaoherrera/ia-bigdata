# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Mask R-CNN object detector                                                                                          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import torch
import torchvision

from src.architectures.arch_base import ArchBase


class MaskRCNNDetector(ArchBase):
    def __init__(self, model_path: str) -> None:
        """Class constructor.

        Args:
            model_path (str): Path to the model checkpoint.
        """

        super().__init__(model_path)

        self.model = torchvision.models.detection.MaskRCNN(
            backbone="resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )

        self.model.roi_heads.box_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.model.roi_heads.box_predictor.in_features, 1)
        )
