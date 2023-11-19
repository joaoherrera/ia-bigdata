# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Common dataset utility functions                                                                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
from typing import Dict, List

import cv2
import numpy as np
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
