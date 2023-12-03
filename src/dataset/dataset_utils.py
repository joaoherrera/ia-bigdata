# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Common dataset utility functions                                                                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
from typing import List, Tuple

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided


def read_image(image_path: str, read_mode: int = cv2.IMREAD_COLOR, channel_first: bool = False) -> np.ndarray:
    """Reads an image from the given path.

    Args:
        image_path (str): The path of the image file.
        read_mode (int, optional): The mode used to read the image. Defaults to cv2.IMREAD_COLOR (BGR image).
        channel_first (bool, optional): Whether to return the image with channel first format (C, H, W).
        Otherwise return the image with channel last format (H, W, C). Defaults to False.
    Returns:
        np.ndarray: The image read from the file.
    Raises:
        ValueError: If the image path does not exist.
    """

    if not os.path.exists(image_path):
        raise ValueError(f"Path {image_path} does not exist.")

    image = cv2.imread(image_path, read_mode)

    if channel_first:
        return image.transpose(2, 0, 1)
    else:
        return image


def read_paths(directory_path: str) -> List[str]:
    """Read the list of paths in a given directory.

    Args:
        directory_path (str): The path of the directory to read.
    Returns:
        List[str]: A list of paths in the directory.
    Raises:
        ValueError: If the given path is not a directory.
    """

    if not os.path.isdir(directory_path):
        raise ValueError(f"Path {directory_path} is not a directory.")

    return os.listdir(directory_path)


def to_patches(image: np.ndarray, patch_size: int, stride: int) -> Tuple[List[np.ndarray], List[str]]:
    # Shape of the output array of patches.
    # o o x x x      x o o x x      x x o o x     x x x o o      x x x x x      x x x x x      x x x x x      x x x x x
    # o o x x x  ->  x o o x x  ->  x x o o x  -> x x x o o  ->  o o x x x  ->  x o o x x  ->  x x o o x  ->  x x x o o
    # x x x x x      x x x x x      x x x x x     x x x x x      o o x x x      x o o x x      x x o o x      x x x o o

    # In this case, we have 8 matrices of 2 rows by 2 columns as result.
    # And the resulting shape should be (2 (image rows), 4 (image columns), 2 (patch rows), 2 (patch columns))

    rows, cols = image.shape[:2]
    patch_rows = (rows - patch_size) // stride + 1
    patch_cols = (cols - patch_size) // stride + 1
    shape = (patch_rows * patch_cols + patch_rows, patch_size, patch_size)

    # ndarray.strides gives us the number of bytes to move in the image to get to the next row or column.
    bytes_patch_cols = image.strides[1]
    bytes_patch_rows = image.strides[0]
    bytes_shift = bytes_patch_cols * stride

    patches = as_strided(x=image, shape=shape, strides=(bytes_shift, bytes_patch_rows, bytes_patch_cols))

    # Remove patches that are not fully in the image, e.g.
    # x x x x o
    # o x x x o
    # o x x x x

    patches = np.delete(patches, obj=np.arange(patch_cols, patches.shape[0], patch_cols), axis=0)
