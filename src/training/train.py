import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser

from src.dataset.augmentations import ScrewAugmentations
from src.dataset.dataset import CocoDataset, DatasetUtils
from src.dataset.preprocessing import OrderedCompose, CocoPreprocessing
from src.engine.classifiers import MobileNetClassifier, ResNetClassifier, SqueezeNetClassifier
from src.engine.trainer import SupervisedTrainer
from src.training.tensorboard import TrainingRecorder


def main(args: dict) -> None:
    recorder = TrainingRecorder(f"{args.get('output_path')}/training_{datetime.now().__str__()}")
    checkpoint_path = os.path.join(recorder.summary_filepath, "checkpoint.pth")

    # Setup augmentations and preprocessing functions to be used during training.
    augmentations_funcs = OrderedCompose([ScrewAugmentations.augment])
    preprocessing_funcs = OrderedCompose([CocoPreprocessing.resize], size=(224, 224))

    # ~~ Train subset
    train_subset = CocoDataset(
        args.get("training_images"),
        args.get("training_annotations"),
        augmentations=augmentations_funcs,
        preprocessing=preprocessing_funcs,
    )
    train_dataloader = DataLoader(
        train_subset,
        batch_size=args.get("batch_size"),
        shuffle=True,
    )

    # ~~ Validation subset
    validation_subset = CocoDataset(
        args.get("validation_images"),
        args.get("validation_annotations"),
        augmentations=None,
        preprocessing=preprocessing_funcs,
    )
    validation_dataloader = DataLoader(
        validation_subset,
        batch_size=args.get("batch_size"),
        shuffle=True,
    )

    # Training settings
    loss_validation = torch.nn.BCELoss()
    loss_train = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = args.get("epochs")

    model = ResNetClassifier(checkpoint_path)
    model.load()

    trainer = SupervisedTrainer(torch.device("cuda:0"), model, recorder)
    trainer.fit(train_subset, validation_subset, optimizer, loss_train, loss_validation, epochs, verbose=False)


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train a deep learning model")

    parser.add_argument(
        "--training-images",
        type=str,
        help="Path to the directory containing training images.",
        required=True,
    )

    parser.add_argument(
        "--training-annotations",
        type=str,
        help="Path to the annotation file of the training set.",
        required=True,
    )

    parser.add_argument(
        "--validation-images",
        type=str,
        help="Path to the directory containing validation images.",
        required=True,
    )

    parser.add_argument(
        "--validation-annotations",
        type=str,
        help="Path to the annotation file of the validation set.",
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output directory, where the model will be saved.",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
        default=16,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs",
        default=50,
    )

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(vars(args))
