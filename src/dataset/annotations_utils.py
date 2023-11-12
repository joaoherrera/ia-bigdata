# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Extra functions useful for handling annotations.                                                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import itertools
import os
from typing import Dict, List

import pandas as pd

from src.dataset.annotations_coco import COCOAnnotations


def to_dataframe(dictionary: Dict) -> pd.DataFrame:
    """Convert a dictionary into a Pandas DataFrame.

    Args:
        dictionary (Dict): The dictionary to be converted.
    Returns:
        pd.DataFrame: The resulting DataFrame.
    """

    return pd.DataFrame(data=dictionary)


def to_dict(data: List[Dict], key_type: str) -> Dict:
    """Convert a list of dictionaries into a dictionary of lists using a specified key type.

    Args:
        data (List[Dict]): The list of dictionaries to be converted.
        key_type (str): The key type to be used for grouping the dictionaries.
    Returns:
        Dict: A dictionary of lists, where the key is the specified key type and the value is a list
        of dictionaries with the same key type.
    Raises:
        ValueError: If the specified key type is not present in the dictionaries.
    Example:
        data = [
            {"name": "John", "age": 25},
            {"name": "Jane", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        key_type = "age"
        result = to_dict(data, key_type)
        # Result: {25: [{"name": "John", "age": 25}, {"name": "Bob", "age": 25}], 30: [{"name": "Jane", "age": 30}]}
    """

    if key_type not in data[0].keys():
        raise ValueError("Invalid key")

    data_dictionary = {}
    data = sorted(data.copy(), key=lambda x: x[key_type])

    for key, group in itertools.groupby(data, lambda x: x[key_type]):
        data_dictionary[key] = list(group)

    return data_dictionary


def to_patches(annotations: COCOAnnotations) -> COCOAnnotations:
    """Given annotations in the COCO format, generate patches based on the annotation bounding boxes.
    For each bounding box, a new image and annotation is created.

    Args:
        annotations (COCOAnnotations): The annotations object to have patches generated.
    Returns:
        COCOAnnotations: A new annotations object with patches as images.
    """

    # Group images and annotations by image_id. This will work as a hash table where the hash is the image ID.
    # Consequently, the algorithm will perform much faster, since only one iteration over the data is needed.
    # All the other access to the data has a constant time complexity.
    images_group = annotations.to_dict(annotations.data["images"], "id")
    annotations_group = annotations.to_dict(annotations.data["annotations"], "image_id")

    patches_annotations = COCOAnnotations.from_dict(dict())
    patches_data = patches_annotations.data  # Copy by reference
    patches_data["categories"] = annotations.data["categories"]  # Categories don't change

    annotations_index = 1  # TODO: work with enumerate() instead.
    for image_id, annotation_list in annotations_group.items():
        image_basename = os.path.basename(images_group[image_id][0]["file_name"])
        image_basename, image_extension = os.path.splitext(image_basename)

        # For each bounding box in an image, a new image and annotation is created.
        for index, annotation in enumerate(annotation_list):
            image_name = f"{image_basename}_{index + 1}{image_extension}"
            image_dimensions = (annotation["bbox"][2], annotation["bbox"][3])

            segmentation = annotation["segmentation"]
            if len(annotation["segmentation"]) > 0:
                segmentation[::2] -= annotation["bbox"][0]
                segmentation[1::2] -= annotation["bbox"][1]

            patches_data["images"].append(
                COCOAnnotations.create_image_instance(
                    id=annotations_index,
                    file_name=image_name,
                    width=image_dimensions[0],
                    height=image_dimensions[1],
                )
            )
            patches_data["annotations"].append(
                COCOAnnotations.create_annotation_instance(
                    id=annotations_index,
                    image_id=annotations_index,
                    cateogory_id=annotation["category_id"],
                    bbox=[[0, 0, image_dimensions[0], image_dimensions[1]]],
                    segmentation=segmentation,
                    history=annotation["bbox"],
                )
            )
            annotations_index += 1
    return patches_annotations
