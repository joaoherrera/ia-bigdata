""" A basic model for machine learning.
"""

import numpy as np
import torch


class SupervisedTrainer:
    def __init__(self, device, model):
        self.device = device
        self.model = model

        self.model.to(self.device)  # Load model in the GPU

    def train(self, dataset, optimizer, loss_func, verbose=False):
        loss_training = []

        # Set module status to training. Implemented in torch.nn.Module
        self.model.train()

        with torch.set_grad_enabled(True):
            for batch in dataset:
                x_pred, y_true = batch
                x_pred, y_true = x_pred.to(self.device), y_true.to(self.device)

                # Predict
                y_pred = self.model(x_pred)

                # Loss computation and weights correction
                loss = loss_func(y_pred, y_true)
                loss.backward()  # backpropagation
                optimizer.step()

                loss_value = loss.item()
                loss_training.append(loss_value)

                if verbose:
                    print(f"Training loss: {loss_value}")

        return np.mean(loss_training)

    def evaluate(self, dataset, coef_func, verbose=False):
        coef_validation = []

        # Set module status to evalutation. Implemented in torch.nn.Module
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(dataset):
                x_pred, y_true = batch
                x_pred, y_true = x_pred.to(self.device), y_true.to(self.device)

                # Predict
                y_pred = self.model(x_pred)

                coef = coef_func(y_pred, y_true)
                coef_value = coef.item()
                coef_validation.append(coef_value)

                if verbose:
                    print(f"Validation loss {coef_value}")

        return np.mean(coef_validation)

    def fit(self, training_dataset, validation_dataset, optimizer, loss_func, coef_func, epochs, verbose=False):
        for epoch in range(epochs):
            print(f"Epoch {epoch}")

            loss_training = self.train(training_dataset, optimizer, loss_func, verbose)
            coef_evalutation = self.evaluate(validation_dataset, coef_func, verbose)

            print(f"Loss training: {loss_training}")
            print(f"Loss validation: {coef_evalutation}")
