# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Core module for training a deep learning model for computer vision tasks                                            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import torch
from torch.nn.modules import loss
from torch.optim import Optimizer

from src.dataset.dataset_base import BaseDataset
from src.training.tensorboard import TrainingRecorder


class SupervisedTrainer:
    def __init__(self, device: str, model: torch.nn.Module, recorder: TrainingRecorder = None):
        """Class constructor. Initializes the trainer module for supervised learning.

        Args:
            device (str): The device to use for training (e.g., 'cuda' or 'cpu').
            model (torch.nn.Module): The model to be trained.
            recorder (TrainingRecorder, optional): A training recorder to track training progress. Defaults to None.
        """

        self.device = device
        self.model = model
        self.recorder = recorder  # Tensorboard recorder to track training progress.
        self.best_loss = 1e20  # Set to a large value, so that the first validation loss is always better.

        self.model.to(self.device)  # Load model in the GPU

    def train(self, dataset: BaseDataset, optimizer: Optimizer, loss_func: loss, verbose: bool = False) -> float:
        """Trains the model on a given dataset.

        Args:
            dataset (BaseDataset): The dataset to train the model on.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            loss_func (torch.nn.modules.loss): The loss function used for training.
            verbose (bool, optional): Whether to print the training loss for each batch. Defaults to False.

        Returns:
            float: The average training loss over all batches.
        """

        loss_training = 0

        # Set module status to training. Implemented in torch.nn.Module
        self.model.train()

        with torch.set_grad_enabled(True):
            for n, batch in enumerate(dataset):
                x_pred, y_true = batch
                x_pred, y_true = x_pred.to(self.device), y_true.to(self.device)

                # Zero gradients for each batch
                optimizer.zero_grad()

                # Predict
                y_pred = self.model(x_pred)

                # Loss computation and weights correction
                loss = loss_func(y_pred, y_true)
                loss.backward()  # backpropagation
                optimizer.step()

                loss_value = loss.item()
                loss_training += loss_value

                if verbose:
                    print(f"Training loss: {loss_value}")

        return loss_training / n

    def evaluate(self, dataset: BaseDataset, loss_func: loss, verbose: bool = False) -> float:
        """Calculate the evaluation loss on the given dataset.

        Args:
            dataset (BaseDataset): The dataset to evaluate the model on.
            loss_func (torch.nn.modules.loss): The loss function to calculate the loss.
            verbose (bool, optional): Whether to print the validation loss. Defaults to False.

        Returns:
            float: The average validation loss.
        """

        loss_validation = 0

        # Set module status to evalutation. Implemented in torch.nn.Module
        self.model.eval()

        with torch.no_grad():
            for n, batch in enumerate(dataset):
                x_pred, y_true = batch
                x_pred, y_true = x_pred.to(self.device), y_true.to(self.device)

                # Predict
                y_pred = self.model(x_pred)

                loss = loss_func(y_pred, y_true)
                loss_value = loss.item()
                loss_validation += loss_value

                if verbose:
                    print(f"Validation loss {loss_value}")

        return loss_validation / n

    def fit(
        self,
        training_dataset: BaseDataset,
        validation_dataset: BaseDataset,
        optimizer: Optimizer,
        train_loss: loss,
        validation_loss: loss,
        epochs: int,
        verbose: bool = False,
    ):
        """Fits the model to the training dataset and validates it on the validation dataset for a
        specified number of epochs.

        Parameters:
            training_dataset (BaseDataset): The dataset used for training.
            validation_dataset (BaseDataset): The dataset used for validation.
            optimizer (Optimizer): The optimizer used for training.
            train_loss (loss): The loss function used for training.
            validation_loss (loss): The loss function used for validation.
            epochs (int): The number of epochs to train the model.
            verbose (bool, optional): Whether to print training progress. Defaults to False.
        """

        for epoch in range(epochs):
            print(f"Epoch {epoch}")

            loss_training = self.train(training_dataset, optimizer, train_loss, verbose)
            loss_validation = self.evaluate(validation_dataset, validation_loss, verbose)

            print(f"Loss training: {loss_training}")
            print(f"Loss validation: {loss_validation}")

            if self.recorder:
                self.recorder.record_scalar("training loss", loss_training, epoch)
                self.recorder.record_scalar("validation loss", loss_validation, epoch)

            # Save checkpoint.
            if loss_validation < self.best_loss:
                self.best_loss = loss_validation
                self.model.save()

        if self.recorder:
            self.recorder.close()
