# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Date: 2023-04-01                                                                                          #
# Author: Joao Herrera                                                                                      #
#                                                                                                           #
# Script to train an instance segmentation model of screws.                                           #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


import torch
from torch.utils.data import DataLoader, random_split

from src.dataset.augmentations import ScrewAugmentations
from src.dataset.dataset import CocoDataset
from src.dataset.preprocessing import CocoPreprocessing, OrderedCompose
from src.engine.models import DummyClassifier
from src.engine.trainer import SupervisedTrainer


def main():
    dataset_root = "/home/joaoherrera/workspace/cv-hardware-parts/datasets/loose_screw_dataset"
    dataset_images_directory = f"{dataset_root}/images"
    dataset_annotations_file = f"{dataset_root}/annotations/annotations.json"

    model = DummyClassifier()
    augmentations_funcs = OrderedCompose([ScrewAugmentations.augment])

    dataset = CocoDataset(
        dataset_images_directory,
        dataset_annotations_file,
        augmentations=augmentations_funcs,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss = torch.nn.BCELoss()

    train_subset, test_subset = random_split(dataset, lengths=[0.8, 0.2])
    train_subset = DataLoader(train_subset, 32, shuffle=True)
    test_subset = DataLoader(test_subset, 32, shuffle=True)

    trainer = SupervisedTrainer(torch.device("cuda:0"), model)
    trainer.fit(train_subset, test_subset, optimizer, loss, loss, 10)


if __name__ == "__main__":
    main()
