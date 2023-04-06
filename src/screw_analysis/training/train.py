# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Date: 2023-04-01                                                                                          #
# Author: Joao Herrera                                                                                      #
#                                                                                                           #
# Script to train an instance segmentation model of screws.                                           #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


import torch

from src.screw_analysis.dataset.dataset import CocoDataset
from src.screw_analysis.engine.models import DummyClassifier
from src.screw_analysis.engine.trainer import SupervisedTrainer
from torch.utils.data import random_split, DataLoader
from src.screw_analysis.dataset.preprocessing import CocoPreprocessing, OrderedCompose


def main():
    dataset_root = "/home/joaoherrera/workspace/cv-hardware-parts/datasets/loose_screw_dataset"
    dataset_images_directory = f"{dataset_root}/images"
    dataset_annotations_file = f"{dataset_root}/annotations/annotations.json"

    model = DummyClassifier()
    preprocessing_funcs = OrderedCompose([CocoPreprocessing.crop])
    dataset = CocoDataset(dataset_images_directory, dataset_annotations_file, preprocessing=preprocessing_funcs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss = torch.nn.BCELoss()

    train_subset, test_subset = random_split(dataset, lengths=[0.8, 0.2])
    train_subset = DataLoader(train_subset, 32, shuffle=True)
    test_subset = DataLoader(test_subset, 32, shuffle=True)

    trainer = SupervisedTrainer(torch.device("cuda:0"), model)
    trainer.fit(train_subset, test_subset, optimizer, loss, loss)


if __name__ == "__main__":
    main()
