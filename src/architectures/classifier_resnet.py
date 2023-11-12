# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ResNet Classifier                                                                                                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import torch
import torchvision

from src.architectures.arch_base import ArchBase


class ResNetClassifier(ArchBase):
    def __init__(self, model_path: str) -> None:
        """Class constructor.

        Args:
            model_path (str): Path to the model checkpoint.
        """

        super().__init__(model_path)
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = torch.nn.Sequential(torch.nn.Linear(self.model.fc.in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward pass of the model to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: Model prediction as a flattened output tensor.
        """

        probability = self.model(x)
        probability = torch.sigmoid(probability)

        if probability.dim() > 1:
            probability = torch.reshape(probability, (-1,))

        return probability
