""" A basic model for machine learning.
"""

import math

import numpy as np
import torch


class SupervisedTrainer:
    def __init__(self, device, model, recorder=None):
        self.device = device
        self.model = model
        self.recorder = recorder
        self.best_loss = 1e20

        self.model.to(self.device)  # Load model in the GPU

    def train(self, dataset, optimizer, loss_func, verbose=False):
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

    def evaluate(self, dataset, loss_func, verbose=False):
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

    def fit(self, training_dataset, validation_dataset, optimizer, train_loss, validation_loss, epochs, verbose=False):       
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

