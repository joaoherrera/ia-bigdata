import os
from argparse import ArgumentParser
from datetime import datetime
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.architectures.detector_maskrcnn import MaskRCNNDetector
from src.dataset.augmentations import Augmentations
from src.dataset.composer import OrderedCompose
from src.dataset.dataset_coco import CocoDataset
from src.dataset.preprocessing import CocoPreprocessing
from src.engine.trainer import SupervisedTrainer
from src.training.tensorboard import TrainingRecorder


def create_training_report(args: dict) -> None:
    """Creates a training report and save it as a text file.

    Args:
        args (dict): A dictionary containing the training arguments provided by the user.
    """

    output_path = args.get("output_path")
    report_path = os.path.join(output_path, "training_report.txt")

    os.makedirs(output_path, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("Training report: \n")
        f.write(f"Training images: {args.get('training_images')}\n")
        f.write(f"Training annotations: {args.get('training_annotations')}\n")
        f.write(f"Validation images: {args.get('validation_images')}\n")
        f.write(f"Validation annotations: {args.get('validation_annotations')}\n")
        f.write(f"Batch size: {args.get('batch_size')}\n")
        f.write(f"Epochs: {args.get('epochs')}\n")
        f.write(f"Learning rate: {args.get('learning_rate')}\n")
        f.write(f"GPU: {args.get('gpu')}\n")


def load_model(checkpoint_path: str, num_classes: int) -> torch.nn.Module:
    """Load a pre-trained model.

    Args:
        checkpoint_path (str): The path to the checkpoint file.
        num_classes (int): The number of classes in the dataset.

    Returns:
        torch.nn.Module: The loaded model.
    """

    model = MaskRCNNDetector(checkpoint_path, num_classes)
    model.load()

    return model


def load_dataset(
    images_path: str,
    annotations_path: str,
    preprocessing_funcs: list,
    augmentations_funcs: list,
    batch_size: int,
    shuffle: bool,
    seed: str,
) -> Tuple[Dataset, DataLoader]:
    """Load a dataset for training or evaluation.

    Args:
        images_path (str): The path to the directory containing the images.
        annotations_path (str): The path to the file containing the annotations.
        preprocessing_funcs (list): A list of preprocessing functions to apply to the images.
        augmentations_funcs (list): A list of augmentation functions to apply to the images.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the samples before each epoch.

    Returns:
        Tuple[Dataset, DataLoader]: A tuple containing the dataset and the dataloader.
    """

    dataset = CocoDataset(images_path, annotations_path, augmentations_funcs, preprocessing_funcs, seed)
    return dataset, dataset.dataloader(batch_size, shuffle)


def train(args: dict) -> None:
    """Trains a model using PyTorch.

    Args:
    - args (dict): A dictionary containing the arguments for training. It should have the following keys:
        - training_images (str): The path to the training images.
        - training_annotations (str): The path to the training annotations.
        - validation_images (str): The path to the validation images.
        - validation_annotations (str): The path to the validation annotations.
        - batch_size (int): The batch size for training.
        - output_path (str): The path where the training outputs will be saved.
        - learning_rate (float): The learning rate for training.
        - epochs (int): The number of training epochs.
        - gpu (bool): Whether to use GPU for training.
    """

    # Load datasets for training and validation
    seed = args.get("seed")
    batch_size = args.get("batch_size")
    augmentations_funcs = OrderedCompose([Augmentations.augment])
    preprocessing_funcs = OrderedCompose([CocoPreprocessing.resize], size=(224, 224))

    train_set, train_loader = load_dataset(
        images_path=args.get("training_images"),
        annotations_path=args.get("training_annotations"),
        preprocessing_funcs=preprocessing_funcs,
        augmentations_funcs=augmentations_funcs,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    __, validation_loader = load_dataset(
        images_path=args.get("validation_images"),
        annotations_path=args.get("validation_annotations"),
        preprocessing_funcs=preprocessing_funcs,
        augmentations_funcs=None,  # No augmentation in validation!
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    # ~ Training settings
    output_path = args.get("output_path")
    learning_rate = args.get("learning_rate")
    epochs = args.get("epochs")
    loss = {"train": torch.nn.BCELoss(), "validation": torch.nn.BCELoss()}
    device = torch.device("cuda") if args.get("gpu") and torch.cuda.is_available() else torch.device("cpu")
    recorder = TrainingRecorder(f"{output_path}/training_{datetime.now().__str__()}")

    # Load model. Weights will be saved in output_path.
    checkpoint_path = os.path.join(output_path, "checkpoint.pth")
    model = load_model(checkpoint_path, num_classes=len(train_set.categories))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    create_training_report(args)
    trainer = SupervisedTrainer(device, model, recorder, seed)
    trainer.fit(train_loader, validation_loader, optimizer, loss["train"], loss["validation"], epochs, verbose=False)


def build_arg_parser() -> ArgumentParser:
    """Builds and returns an argument parser for training a deep learning model.

    Returns:
        ArgumentParser: The argument parser object.
    """

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

    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="A seed for reproducibility",
        default=2183648025,
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Whether to preprocess the dataset or not",
        default=False,
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Whether to augment the dataset or not",
        default=False,
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Whether to use GPU or not",
        default=True,
    )

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(vars(args))
