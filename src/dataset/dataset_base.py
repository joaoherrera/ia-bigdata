# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Base classes for datasets.                                                                                          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from abc import ABC, abstractmethod
from typing import Any, Tuple

from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset, ABC):
    def __init__(self) -> None:
        """Initializes a new instance of the class."""

        self.data = None

    def is_empty(self) -> bool:
        """Check if the data structure is empty.

        Returns:
            bool: True if the data structure is empty, False otherwise.
        """

        return self.data is None or len(self.data) == 0

    def set_data(self, data) -> None:
        """Sets the data attribute of the object.

        Args:
            data (any): The data to be set.
        """

        self.data = data

    def get_data(self) -> Any:
        """Return the data stored in the object."""

        return self.data

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the object.

        Args:
            self: The object itself.
        Returns:
            An integer representing the length of the object.
        """

        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Get the item at the specified index.

        Args:
            idx (int): The index of the item to get.
        """

        raise NotImplementedError()

    @abstractmethod
    def dataloader(self, batch_size: int, shuffle: bool) -> DataLoader:
        """Class method that returns a DataLoader object.

        Args:
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data or not.

        Returns:
            DataLoader: The DataLoader object.
        """

        raise NotImplementedError()


class MutableDataset(BaseDataset, ABC):
    @abstractmethod
    def split(self) -> Tuple[Any, ...]:
        """An abstract method that splits a dataset.

        Returns:
            Tuple[Any, ...]: A tuple containing the split objects.
        """

        raise NotImplementedError()
