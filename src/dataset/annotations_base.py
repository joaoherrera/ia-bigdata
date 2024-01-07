# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Base routines for dealing with annotation files                                                                     #
# Annotation files contain the ground truth, which can be generated manually by an annotator or by an IA model.       #
# Both COCO 1.0 and XML formats are supported. We recommend using CVAT to generate them                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import json
from abc import ABC
from typing import Any, Dict

import xmltodict


class AnnotationsBase(ABC):
    @staticmethod
    def load_file(file_path: str) -> Dict | None:
        """Load a file and parse it as XML.

        Args:
            file_path (str): The path to the file to be loaded.
        Returns:
            Dict | None: The parsed XML data as a dictionary, or None if there was an error.
        """

        raise NotImplementedError

    @staticmethod
    def save_file(data: Dict, file_path: str) -> None:
        """
        Save a file.

        Args:
            data (Dict): The data to be saved.
            file_path (str): The path of the file to be saved.
        """

        raise NotImplementedError


class JSONAnnotations(AnnotationsBase):
    @staticmethod
    def load_file(file_path: str) -> Dict | None:
        """Load a JSON annotation file and return its contents as a dictionary.

        Args:
            file_path (str): The path to the file to be loaded.
        Returns:
            Dict | None: The contents of the file as a dictionary, or None if an OSError occurs.
        """

        try:
            with open(file_path, "r") as annotation_file:
                annotations = json.load(annotation_file)
            return annotations

        except OSError:
            return None

    @staticmethod
    def save_file(annotations: Any, file_path: str) -> None:
        """Save the annotations data to a JSON file.

        Args:
            annotations (Any): The annotations to be saved.
            file_path (str): The path of the file to save the annotations to.
        """

        with open(file_path, "w") as annotation_file:
            json.dump(annotations.data, annotation_file)


class XMLAnnotations(AnnotationsBase):
    @staticmethod
    def load_file(file_path: str) -> Dict | None:
        """Load a file and parse it as XML.

        Args:
            file_path (str): The path to the file to be loaded.
        Returns:
            Dict | None: The parsed XML data as a dictionary, or None if there was an error.
        """

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
        """Save the given annotations to the specified file.

        Args:
            annotations (Any): The annotations to be saved.
            file_path (str): The path of the file to save the annotations to.
        """

        with open(file_path, "w") as annotation_file:
            annotation_file.write(xmltodict.unparse(annotations))
