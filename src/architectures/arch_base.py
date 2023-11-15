# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Base routines for all deep learning models                                                                          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
from abc import ABC

import torch


class ArchBase(torch.nn.Module, ABC):
    def __init__(self, model_path: str) -> None:
        """Class constructor.

        Args:
            model_path (str): The path to the model checkpoint file.
        """

        super().__init__()
        self.model_path = model_path
        self.model: torch.nn.Module = None

    def save(self) -> bool:
        """Saves the model checkpoint to a file.

        Returns:
            bool: True if the model is successfully saved, False otherwise.
        """

        try:
            if self.model_path is not None:
                torch.save(self.model.state_dict(), self.model_path)
                return True
            else:
                return False
        except Exception as excpt:
            print(excpt)
            return False

    def load(self) -> bool:
        """Loads the model from the checkpoint file (weights).

        Returns:
            bool: True if the model was successfully loaded, False otherwise.
        """

        try:
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path))
                return True
            else:
                return False
        except Exception as excpt:
            print(f"Error while loading model checkpoints: {excpt}")
            return False
