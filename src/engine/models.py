import torch
import torchvision


class DummyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Sequential(torch.nn.Linear(self.model.fc.in_features, 1))

    def forward(self, x):
        probability = self.model(x)

        if probability.dim() > 1:
            probability = torch.reshape(probability, (-1,))

        return probability
