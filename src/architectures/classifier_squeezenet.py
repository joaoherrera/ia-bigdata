# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# SqueezeNet Classifier                                                                                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import torch
import torchvision

from src.architectures.arch_base import ArchBase


class SqueezeNetClassifier(ArchBase):
    def __init__(self, model_path: str) -> None:
        """Class constructor.

        Args:
            model_path (str): Path to the model checkpoint.
        """

        super().__init__(model_path)
        self.model = torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1)

        # Transform original output layer (ImageNet1k) into binary classification problem.
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward pass of the model to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: Model prediction as a flattened output tensor.
        """

        x = self.model.forward(x)
        x = torch.nn.functional.softmax(x, dim=0)
        return x.flatten()
