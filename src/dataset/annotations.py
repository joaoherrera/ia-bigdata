# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Date: 2023-04-01                                                                                          #
# Author: Joao Herrera                                                                                      #
#                                                                                                           #
# A class representation of an annotation file.                                                             #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import itertools
import json
import os
import random
from ctypes import ArgumentError
from typing import Dict, List

import numpy as np
import pandas as pd


class JSONAnnotations:
    @staticmethod
    def dict_to_dataframe(dictionary: Dict) -> pd.DataFrame:
        return pd.DataFrame(data=dictionary)

    @staticmethod
    def dataframe_to_dict(data_frame: pd.DataFrame) -> Dict:
        raise NotImplementedError

    @staticmethod
    def to_dict(data: list, key_type: str) -> dict:
        if not key_type in data[0].keys():
            raise ValueError("Invalid key")

        data_dictionary = {}
        data = sorted(data.copy(), key=lambda x: x[key_type])

        for key, group in itertools.groupby(data, lambda x: x[key_type]):
            data_dictionary[key] = list(group)

        return data_dictionary

    @staticmethod
    def load_file(file_path: str) -> Dict | None:
        try:
            with open(file_path, "r") as annotation_file:
                annotations = json.load(annotation_file)
            return annotations
        except OSError:
            return None

    @staticmethod
    def save_file(annotations: "COCOAnnotations", file_path: str) -> None:
        with open(file_path, "w") as annotation_file:
            json.dump(annotations.data, annotation_file)


class COCOAnnotations(JSONAnnotations):
    """Handle coco-like annotations."""

    def __init__(self, filepath: str = None, balancing_strategy: str = None) -> None:
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
    def from_data(annotations: Dict) -> "COCOAnnotations":
        coco_annotations = COCOAnnotations()

        if len(annotations) == 0:
            coco_annotations.data = {"categories": [], "images": [], "annotations": []}
        else:
            coco_annotations.data = annotations

        return coco_annotations

    def load(self, inplace: bool = True) -> None | Dict:
        if inplace:
            self.data = super(COCOAnnotations, self).load_file(self.filepath)
        else:
            return super(COCOAnnotations, self).load_file(self.filepath)

    def save(self, output_path: str = None) -> None:
        if output_path is None:
            if os.path.isfile(self.filepath):
                output_path = self.filepath
            else:
                raise ArgumentError("Need a path for a JSON file as output.")
        self.save_file(self, output_path)

    def apply_oversampling(self, inplace=False, seed=None) -> List[dict] | None:
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

    def apply_undersampling(self, inplace=False, seed=None) -> List[dict] | None:
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
        instance = {"id": id, "file_name": file_name, "width": width, "height": height}

        for key, value in kwargs.items():
            instance[key] = value

        return instance

    @staticmethod
    def create_annotation_instance(id: int, image_id: int, cateogory_id: int, bbox: List, **kwargs) -> Dict:
        instance = {"id": id, "image_id": image_id, "category_id": cateogory_id, "bbox": bbox}

        for key, value in kwargs.items():
            instance[key] = value

        return instance

    @staticmethod
    def create_category_instance(id: int, category_name: str, **kwargs) -> Dict:
        instance = {"id": id, "category_name": category_name}

        for key, value in kwargs.items():
            instance[key] = value

        return instance
