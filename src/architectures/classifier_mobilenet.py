# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# MobileNet Classifier                                                                                                #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import torch
import torchvision

from src.architectures.arch_base import ArchBase


class MobileNetClassifier(ArchBase):
    def __init__(self, model_path: str) -> None:
        """Class constructor.

        Args:
            model_path (str): Path to the model checkpoint.
        """

        super().__init__(model_path)
        self.model = torchvision.models.mobilenet_v3_small(num_classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward pass of the model to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: Model prediction as a flattened output tensor.
        """

        x = self.model.forward(x)
        return x.flatten()
