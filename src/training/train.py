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
from src.dataset.dataset import CocoDataset, DatasetUtils
from src.dataset.preprocessing import OrderedCompose, CocoPreprocessing
from src.engine.models import MobileNetClassifier, ResNetClassifier, SqueezeNetClassifier
from src.engine.trainer import SupervisedTrainer
from src.training.tensorboard import TrainingRecorder


def main():
    dataset_root = "/home/joaoherrera/server/datasets/lemons"
    dataset_images_directory = f"{dataset_root}/images"
    dataset_annotations_file = f"{dataset_root}/annotations/annotations.json"
    dataset_training_annotation_file = f"{dataset_root}/annotations/training_annotations.json"
    dataset_validation_annotation_file = f"{dataset_root}/annotations/validation_annotations.json"
    recorder = TrainingRecorder(f"{dataset_root}/experiments/training_{datetime.now().__str__()}")
    checkpoint_path = os.path.join(recorder.summary_filepath, "checkpoint.pth")

    model = ResNetClassifier(checkpoint_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_train = torch.nn.BCELoss()
    loss_test = torch.nn.BCELoss()

    augmentations_funcs = OrderedCompose([ScrewAugmentations.augment])
    preprocessing_funcs = OrderedCompose([CocoPreprocessing.resize], size=(224, 224))
    
    train_subset = CocoDataset(dataset_images_directory, dataset_training_annotation_file)
    test_subset = CocoDataset(dataset_images_directory, dataset_validation_annotation_file)
    
    test_subset.augmentations = None
    
    # Save annotations for both datasets
    train_subset.tree.save(output_path=os.path.join(recorder.summary_filepath, "training_annotations.json"))
    test_subset.tree.save(output_path=os.path.join(recorder.summary_filepath, "validation_annotations.json"))
 
    train_subset = DataLoader(train_subset, 16, shuffle=True)
    test_subset = DataLoader(test_subset, 16, shuffle=True)
    
    trainer = SupervisedTrainer(torch.device("cuda:0"), model, recorder)
    trainer.fit(train_subset, test_subset, optimizer, loss_train, loss_test, 50, verbose=False)


if __name__ == "__main__":
    main()
