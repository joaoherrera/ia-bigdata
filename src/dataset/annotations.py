# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Date: 2023-04-01                                                                                          #
# Author: Joao Herrera                                                                                      #
#                                                                                                           #
# A class representation of an annotation file.                                                             #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from ctypes import ArgumentError
from typing import Dict, List
import json
import pandas as pd
import itertools
import os


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

    def __init__(self, filepath: str = None) -> None:
        self.filepath = filepath

        if self.filepath is not None and os.path.isfile(self.filepath):
            self.load()

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
