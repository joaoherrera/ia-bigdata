# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Handle coco-like annotations.                                                                                       #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
import random
from ctypes import ArgumentError
from typing import Dict, List

from src.dataset.annotations_base import JSONAnnotations


class COCOAnnotations(JSONAnnotations):
    def __init__(self, filepath: str = None, balancing_strategy: str = None) -> None:
        """Initializes an instance of the class.

        Args:
            filepath (str): The file path to load data from.
            balancing_strategy (str): The balancing strategy to apply.
        """

        self.filepath = filepath

        if self.filepath is not None and os.path.isfile(self.filepath):
            self.load()

        if balancing_strategy is not None:
            if balancing_strategy == "oversampling":
                self.apply_oversampling(inplace=True)
            elif balancing_strategy == "undersampling":
                self.apply_undersampling(inplace=True)
            else:
                raise ArgumentError(f"Invalid balancing strategy: {balancing_strategy}")

    @staticmethod
    def from_dict(annotations: Dict) -> "COCOAnnotations":
        """Create a new COCOAnnotations object from the given annotations.

        Args:
            annotations (Dict): A dictionary containing COCO annotations.
        Returns:
            COCOAnnotations: A new COCOAnnotations object.
        """

        coco_annotations = COCOAnnotations()

        if len(annotations) == 0:
            coco_annotations.data = {"categories": [], "images": [], "annotations": []}
        else:
            coco_annotations.data = annotations

        return coco_annotations

    def load(self, inplace: bool = True) -> None | Dict:
        """Load the data from the specified filepath.

        Args:
            inplace (bool, optional): Whether to load the data inplace. Defaults to True.
        Returns:
            None or Dict: If `inplace` is True, returns None. If `inplace` is False, returns the loaded data as
            a dictionary.
        """

        if inplace:
            self.data = super(COCOAnnotations, self).load_file(self.filepath)
        else:
            return super(COCOAnnotations, self).load_file(self.filepath)

    def save(self, output_path: str = None) -> None:
        """Saves the object to a JSON file.

        Args:
            output_path (str, optional): The path where the JSON file will be saved. If not provided,
            the object will be saved to the same path as the input file. Defaults to None.
        Raises:
            ArgumentError: If output_path is not provided and the input file does not exist.
        """

        if output_path is None:
            if os.path.isfile(self.filepath):
                output_path = self.filepath
            else:
                raise ArgumentError("Need a path for a JSON file as output.")
        self.save_file(self, output_path)

    def apply_oversampling(self, inplace=False, seed=None) -> List[Dict] | None:
        """Apply oversampling to the data. Oversampling is a data augmentation technique that balances
        the number of annotations in each category by randomly selecting annotations from the
        same category.

        Parameters:
            inplace (bool, optional): If True, the oversampled data is added to the existing data.
                If False, the oversampled data is returned. Defaults to False.
            seed (int, optional): Seed for the random number generator. Defaults to None.

        Returns:
            List[Dict] | None: The oversampled data if inplace is False, None otherwise.
        """

        if seed is not None:
            random.seed(seed)

        oversampled = []

        annotations_category = self.to_dict(self.data["annotations"], "category_id")
        categories_summary = {id: len(anns) for id, anns in annotations_category.items()}
        categories_max = max(categories_summary.values())

        for category_id, n_annotations in categories_summary.items():
            if n_annotations < categories_max:
                n_random = random.choices(annotations_category[category_id], k=(categories_max - n_annotations))
                oversampled.extend(n_random)

        if inplace:
            self.data["annotations"].extend(oversampled)
        else:
            return oversampled

    def apply_undersampling(self, inplace=False, seed=None) -> List[Dict] | None:
        """Apply undersampling to the dataset. Undersampling is a data augmentation technique that balances the
        number of annotations in each category by randomly selecting annotations from the same category
        and removing them.

        Args:
            inplace (bool, optional): If True, the undersampled annotations will be added to the existing dataset.
                If False, the undersampled annotations will be returned. Defaults to False.
            seed (int, optional): The seed value for the random number generator. Defaults to None.

        Returns:
            List[Dict] | None: The undersampled annotations if inplace is False, otherwise None.
        """

        if seed is not None:
            random.seed(seed)

        undersampled = []

        annotations_category = self.to_dict(self.data["annotations"], "category_id")
        categories_summary = {id: len(anns) for id, anns in annotations_category.items()}
        categories_min = min(categories_summary.values())

        for category_id, n_annotations in categories_summary.items():
            if n_annotations > categories_min:
                n_random = random.choices(annotations_category[category_id], k=(n_annotations - categories_min))
                undersampled.extend(n_random)

        if inplace:
            self.data["annotations"].extend(undersampled)
        else:
            return undersampled

    @staticmethod
    def create_image_instance(id: int, file_name: str, width: int, height: int, **kwargs) -> Dict:
        """Creates an image annotation instance with the given parameters and optional keyword arguments.

        Args:
            id (int): The ID of the image.
            file_name (str): The file name of the image.
            width (int): The width of the imagee.
            height (int): The height of the image.
            **kwargs: Additional keyword arguments to be included in the image annotation.

        Returns:
            Dict: A dictionary representing the image with the provided parameters and keyword arguments.
        """

        instance = {"id": id, "file_name": file_name, "width": width, "height": height}

        for key, value in kwargs.items():
            instance[key] = value

        return instance

    @staticmethod
    def create_annotation_instance(id: int, image_id: int, category_id: int, bbox: List, **kwargs) -> Dict:
        """Creates an annotation instance with the given parameters.

        Parameters:
            id (int): The ID of the annotation.
            image_id (int): The ID of the image.
            category_id (int): The ID of the category.
            bbox (List): The bounding box coordinates of the annotation.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict: The created annotation instance.
        """

        instance = {"id": id, "image_id": image_id, "category_id": category_id, "bbox": bbox}

        for key, value in kwargs.items():
            instance[key] = value

        return instance

    @staticmethod
    def create_category_instance(id: int, category_name: str, **kwargs) -> Dict:
        """Create a category instance with the given id and category name.

        Parameters:
            id (int): The id of the category.
            category_name (str): The name of the category.
            **kwargs: Additional key-value pairs to include in the instance.

        Returns:
            Dict: The category instance with the id, name, and additional key-value pairs.
        """

        instance = {"id": id, "name": category_name}

        for key, value in kwargs.items():
            instance[key] = value

        return instance
