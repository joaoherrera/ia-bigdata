import os

import torch
import torchvision

# Redes mais compactas -> MobileNet, SqueezeNet;
# GANS: testar umas 2 ou 3 inicialmente. MAio junho: GANS se não avançar, volta no meio de julho


class CustomClassifier(torch.nn.Module):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path
        self.model: torch.nn.Module = None

    def save(self) -> bool:
        try:
            if self.model_path is not None:
                torch.save(self.model.state_dict(), self.model_path)
                return True
            else:
                return False
        except Exception as excpt:
            print(excpt)
            return False

    def load(self) -> bool:
        try:
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path))
                return True
            else:
                return False
        except Exception as excpt:
            print(excpt)
            return False


class ResNetClassifier(CustomClassifier):
    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)
        self.model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
        self.model.fc = torch.nn.Sequential(torch.nn.Linear(self.model.fc.in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probability = self.model(x)

        if probability.dim() > 1:
            probability = torch.reshape(probability, (-1,))

        return probability


class SqueezeNetClassifier(CustomClassifier):
    def __init__(self, model_path: str) -> None:
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
        x = self.model.forward(x)

        return x.flatten()
