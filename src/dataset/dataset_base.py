# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Base classes for datasets.                                                                                          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseDataset(ABC):
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


class MutableDataset(BaseDataset, ABC):
    @abstractmethod
    def split(self) -> Tuple[Any, ...]:
        """An abstract method that splits a dataset.

        Returns:
            Tuple[Any, ...]: A tuple containing the split objects.
        """

        raise NotImplementedError()
