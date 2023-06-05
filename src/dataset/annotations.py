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
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import xmltodict


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


class XMLAnnotations:
    @staticmethod
    def load_file(file_path: str) -> Dict | None:
        try:
            with open(file_path, "r") as annotation_file:
                annotations = annotation_file.read()
                annotations = xmltodict.parse(annotations, attr_prefix="")
            return annotations

        except Exception as exc:
            print(exc)
            return None

    @staticmethod
    def save_file(annotations: Any, file_path: str) -> None:
        with open(file_path, "w") as annotation_file:
            annotation_file.write(xmltodict.unparse(annotations))


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
    def create_annotation_instance(id: int, image_id: int, category_id: int, bbox: List, **kwargs) -> Dict:
        instance = {"id": id, "image_id": image_id, "category_id": category_id, "bbox": bbox}

        for key, value in kwargs.items():
            instance[key] = value

        return instance

    @staticmethod
    def create_category_instance(id: int, category_name: str, **kwargs) -> Dict:
        instance = {"id": id, "name": category_name}

        for key, value in kwargs.items():
            instance[key] = value

        return instance


class CVATAnnotations(XMLAnnotations):
    def __init__(self, filepath: str = None, balancing_strategy: str = None) -> None:
        self.filepath = filepath
        self.balancing = balancing_strategy
        
        if self.filepath is not None and os.path.isfile(self.filepath):
            self.load()
    
    def load(self, inplace=True) -> None | Dict:
        # "annotations" is the root field in the xml file. There is nothing else in this level.

        if inplace:
            self.data = super(CVATAnnotations, self).load_file(self.filepath)["annotations"]
            
            if "meta" not in self.data.keys() or "image" not in self.data.keys():
                raise TypeError("XML file does not contain CVAT annotations file structure.")
        else:
            return super(CVATAnnotations, self).load_file(self.filepath)["annotations"]
    
    def convert_to_coco(self) -> COCOAnnotations:
        export_type = "task" if "task" in self.data["meta"].keys() else "job"
        cvat_categories = self.data["meta"][export_type]["labels"]["label"]
        cvat_images = self.data["image"]
        
        coco_annotations = COCOAnnotations(None, None)
        coco_annotations.filepath = self.filepath        
        coco_annotations.data = {"categories": [], "images": [], "annotations": []}
        
        # Convert CVAT categories to COCO categories
        for i, category in enumerate(cvat_categories):
            coco_category = COCOAnnotations.create_category_instance(i + 1, category["name"])
            coco_annotations.data["categories"].append(coco_category)
        
        # --> Just to facilitate CVAT annotations categories to COCO annotations categories.
        categories_by_name = COCOAnnotations.to_dict(coco_annotations.data["categories"], "name")

        # Convert CVAT annotations to COCO annotations
        for i, image in enumerate(cvat_images):
            coco_image = COCOAnnotations.create_image_instance(i + 1, image["name"], image["width"], image["height"])
            coco_annotations.data["images"].append(coco_image)
            
            category_id = categories_by_name[image["tag"]["label"]][0]["id"]
            bbox = [0, 0, coco_image["width"], coco_image["height"]]
            coco_annotation = COCOAnnotations.create_annotation_instance(i + 1, coco_image["id"], category_id, bbox)
            coco_annotations.data["annotations"].append(coco_annotation)
                
        return coco_annotations
