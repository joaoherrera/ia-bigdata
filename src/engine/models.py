import torch
import torchvision


class DummyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Sequential(torch.nn.Linear(self.model.fc.in_features, 1))

    def forward(self, x):
        return torch.squeeze(torch.sigmoid(self.model(x)))
