# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Date: 2023-04-01                                                                                          #
# Author: Joao Herrera                                                                                      #
#                                                                                                           #
# Script to train a classifier model of screws.                                                             #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split

from src.dataset.augmentations import ScrewAugmentations
from src.dataset.dataset import CocoDataset
from src.dataset.preprocessing import OrderedCompose
from src.engine.models import ResNetClassifier, SqueezeNetClassifier
from src.engine.trainer import SupervisedTrainer
from src.training.tensorboard import TrainingRecorder


def main():
    dataset_root = "/home/joaoherrera/server/datasets/screws"
    dataset_images_directory = f"{dataset_root}/images"
    dataset_annotations_file = f"{dataset_root}/annotations/annotations.json"
    recorder = TrainingRecorder(f"{dataset_root}/experiments/training_{datetime.now().__str__()}")
    checkpoint_path = os.path.join(recorder.summary_filepath, "checkpoint.pth")

    model = ResNetClassifier(checkpoint_path)
    augmentations_funcs = OrderedCompose([ScrewAugmentations.augment])

    dataset = CocoDataset(
        dataset_images_directory,
        dataset_annotations_file,
        augmentations=augmentations_funcs,
        balancing_strategy="oversampling",
    )

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    loss = torch.nn.BCEWithLogitsLoss()

    train_subset, test_subset = dataset.split(0.8, 0.2, random=True)
    test_subset.augmentations = None

    train_subset = DataLoader(train_subset, 32, shuffle=True)
    test_subset = DataLoader(test_subset, 32, shuffle=True)

    trainer = SupervisedTrainer(torch.device("cuda:0"), model, recorder)
    trainer.fit(train_subset, test_subset, optimizer, loss, loss, 300, verbose=False)


if __name__ == "__main__":
    main()
