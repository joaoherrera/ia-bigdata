# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Hook for looging training progress on TensorBoard.                                                                  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import torch
from torch.utils.tensorboard.writer import SummaryWriter


class TrainingRecorder:
    def __init__(self, summary_filepath: str) -> None:
        """Class constructor.

        Args:
            summary_filepath (str): The filepath to the summary file.
        """

        self.summary_filepath = summary_filepath
        self.writer = SummaryWriter(self.summary_filepath)

    def record_scalar(self, tag: str, value: float, step=None) -> None:
        """Record a scalar value for the given tag at the specified step.

        Args:
            tag (str): The tag to identify the scalar value.
            value (float): The scalar value to record.
            step (Optional[int]): The step at which to record the scalar value. Defaults to None.
        """

        self.writer.add_scalar(tag, value, step)

    def record_image(self, tag: str, image: torch.Tensor, step=None) -> None:
        """Records an image with a given tag and adds it to the writer.

        Args:
            tag (str): The tag for the image.
            image (torch.Tensor): The image to be recorded.
            step (Optional): The step number for the image. Default is None.
        """

        self.writer.add_image(tag, image, step)

    def close(self) -> None:
        """Closes the writer."""

        self.writer.close()
