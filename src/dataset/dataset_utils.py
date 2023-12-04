# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Common dataset utility functions                                                                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
from itertools import product
from typing import Dict, List, Tuple

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from pycocotools import mask as coco_mask


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


def generate_instance_mask(image: np.ndarray, annotation: Dict) -> np.ndarray:
    """Generate an instance mask for the given annotation.

    Args:
        image (np.ndarray): The image array.
        annotation (Dict): The annotation dictionary.print(annotation["segmentation"].shape)

    Returns:
        np.ndarray: The instance mask.
    """

    # There are some annotations that correspond to a crowd of objects (e.g. crowd of people).
    # In such cases, the segmentation is not a polygon, but a Run-length Encoding (RLE). A RLE is basicaly composed by
    # a list of values followed by the number of occurences that this value appears sequentially.
    # Pycocotools package provides functions to encode and decode RLEs.

    if annotation["iscrowd"] == 1:
        mask = coco_mask.decode(annotation["segmentation"])
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        polygons = np.array(annotation["segmentation"], dtype=object)

        # Sometimes an object can be composed by multiple polygons.
        if polygons.shape[0] > 1:
            for polygon in polygons:
                polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
                mask = cv2.fillPoly(mask, [polygon], color=1)
        else:
            polygon = polygons.reshape((-1, 2)).astype(np.int32)
            mask = cv2.fillPoly(mask, [polygon], color=1)

    return mask


def to_patches(image: np.ndarray, patch_size: int, stride: int) -> Tuple[List[np.ndarray], List[Tuple, Tuple]]:
    """Split an image into patches according to a given patch size and stride.

    IMPORTANT: `as_strided` returns a view, so modifying the returned array will modify the original array.
    To avoid this, copy the returned array before modifying it.

    Args:
        image (np.ndarray): The image to split into patches.
        patch_size (int): The size of the patches. Only square patches are supported.
        stride (int): The stride between the patches.
    Returns:
        Tuple[List[np.ndarray], List[Tuple, Tuple]]: A tuple containing the list of patches and the list of
        their coordinates.
    Raises:
        AssertionError: If the patch size or stride are not greater than 0 or if the patch size is greater than
        the image size.
    """

    assert patch_size > 0, "Patch size must be greater than 0"
    assert stride > 0, "Stride must be greater than 0"
    assert patch_size < image.shape[0], "Patch size must be smaller than the image height"
    assert patch_size < image.shape[1], "Patch size must be smaller than the image width"

    # ~~ Defining the shape of the output array of patches
    # Consider the following image:

    # o o x x x      x o o x x      x x o o x     x x x o o      x x x x x      x x x x x      x x x x x      x x x x x
    # o o x x x  ->  x o o x x  ->  x x o o x  -> x x x o o  ->  o o x x x  ->  x o o x x  ->  x x o o x  ->  x x x o o
    # x x x x x      x x x x x      x x x x x     x x x x x      o o x x x      x o o x x      x x o o x      x x x o o

    # In this case, we will end up with an array of shape (8, 2, 2),
    # which corresponds to (image rows * image columns, patch rows, patch columns)
    # We add patch_rows to `shape` because `as_strided` includes patches that are not fully in the image, e.g.

    # x x x x o
    # o x x x o
    # o x x x x

    rows, cols = image.shape[:2]
    patch_rows = (rows - patch_size) // stride + 1
    patch_cols = (cols - patch_size) // stride + 1
    shape = (patch_rows * patch_cols + patch_rows, patch_size, patch_size)

    # ndarray.strides gives us the number of bytes to move in the image to get to the next row or column.
    bytes_patch_cols = image.strides[1]
    bytes_patch_rows = image.strides[0]
    bytes_shift = bytes_patch_cols * stride

    patches = as_strided(x=image, shape=shape, strides=(bytes_shift, bytes_patch_rows, bytes_patch_cols))
    coordinates = list(product(np.arange(2), np.arange(4)))  # [(0, 0), (0, 1), (1, 0), (1, 1), ... ]

    # Remove patches that are not fully in the image.
    patches = np.delete(patches, obj=np.arange(patch_cols, patches.shape[0], patch_cols + 1), axis=0)

    return patches, coordinates
