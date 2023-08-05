import torch
import torchvision
from src.engine.classifiers import CustomClassifier as CustomModel


class MaskRCNNDetector(CustomModel):

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)
        self.model = torchvision.models.detection.MaskRCNN(backbone="resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.roi_heads.box_predictor = torch.nn.Sequential(torch.nn.Linear(self.model.roi_heads.box_predictor.in_features, 1))