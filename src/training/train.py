# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Date: 2023-04-01                                                                                          #
# Author: Joao Herrera                                                                                      #
#                                                                                                           #
# Script to train an instance segmentation model of screws.                                           #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split

from src.dataset.augmentations import ScrewAugmentations
from src.dataset.dataset import CocoDataset
from src.dataset.preprocessing import OrderedCompose
from src.engine.models import DummyClassifier
from src.engine.trainer import SupervisedTrainer
from src.training.tensorboard import TrainingRecorder


def main():
    dataset_root = "/home/joaoherrera/server/datasets/screws"
    dataset_images_directory = f"{dataset_root}/images"
    dataset_annotations_file = f"{dataset_root}/annotations/annotations.json"
    recorder = TrainingRecorder(f"{dataset_root}/experiments/training_{datetime.now().__str__()}")

    model = DummyClassifier()
    augmentations_funcs = OrderedCompose([ScrewAugmentations.augment])

    dataset = CocoDataset(
        dataset_images_directory,
        dataset_annotations_file,
        augmentations=augmentations_funcs,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss = torch.nn.BCEWithLogitsLoss()

    train_subset, test_subset = random_split(dataset, lengths=[0.8, 0.2])
    train_subset = DataLoader(train_subset, 32, shuffle=True)
    test_subset = DataLoader(test_subset, 32, shuffle=True)

    trainer = SupervisedTrainer(torch.device("cuda:0"), model, recorder)
    trainer.fit(train_subset, train_subset, optimizer, loss, loss, 100, verbose=False)


if __name__ == "__main__":
    main()
