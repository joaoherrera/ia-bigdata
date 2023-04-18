# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Date: 2023-04-01                                                                                          #
# Author: Joao Herrera                                                                                      #
#                                                                                                           #
# A class representation of an annotation file.                                                             #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from typing import  Dict
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
            data_dictionary[key] = list(group)[0]

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
            json.dump(annotations, annotation_file)


class COCOAnnotations:
    """Handle coco-like annotations."""
    
    def __init__(self, filepath: str) -> None:
        if not os.path.isfile(filepath):
            raise ValueError(f"Path {filepath} is not a file.")

        self.filepath = filepath
        self.load()
        
    def load(self, inplace: bool = True) -> None | dict:
        with open(self.filepath) as coco_file:
            data = json.load(coco_file)

            if inplace:
                self.data = data
            else: 
                return data
    
    @staticmethod
    def to_dict(data: list, key_type: str) -> dict:
        if not key_type in data[0].keys():
            raise ValueError("Invalid key")
        
        data_dictionary = {}
        for key, group in itertools.groupby(data, lambda x: x[key_type]):
            data_dictionary[key] = list(group)[0]
        
        return data_dictionary