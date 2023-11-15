# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# This file contains a class for handling CVAT annotations exported as XML or CVAT for images v1.1                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
from typing import Dict

from src.dataset.annotations_base import XMLAnnotations
from src.dataset.annotations_coco import COCOAnnotations
from src.dataset.annotations_utils import to_dict


class CVATAnnotations(XMLAnnotations):
    def __init__(self, filepath: str = None, balancing_strategy: str = None) -> None:
        """Initializes the class object.

        Args:
            filepath (str): The path to the file to be loaded.
            balancing_strategy (str): The balancing strategy to be used.
        """

        self.filepath = filepath
        self.balancing = balancing_strategy

        if self.filepath is not None and os.path.isfile(self.filepath):
            self.load()

    def load(self, inplace: bool = True) -> None | Dict:
        """Loads the CVAT annotations from the XML file.

        Args:
            inplace (bool): If True, the annotations are loaded in-place. If False, the annotations are returned.
        Returns:
            Dict | None: If inplace is True, returns None. If inplace is False, returns the loaded annotations.
        """

        if inplace:
            self.data = super(CVATAnnotations, self).load_file(self.filepath)["annotations"]

            # Look for mandatory fields in a CVAT annotations file.
            if "meta" not in self.data.keys() or "image" not in self.data.keys():
                raise TypeError("XML file does not contain CVAT annotations file structure.")
        else:
            return super(CVATAnnotations, self).load_file(self.filepath)["annotations"]

    def convert_to_coco(self) -> COCOAnnotations:
        """Converts the current data format to the COCOAnnotations format.

        Returns:
            COCOAnnotations: The converted data in COCOAnnotations format.
        """

        export_type = "task" if "task" in self.data["meta"].keys() else "job"
        cvat_categories = self.data["meta"][export_type]["labels"]["label"]
        cvat_images = self.data["image"]

        coco_annotations = COCOAnnotations(None, None)
        coco_annotations.filepath = self.filepath
        coco_annotations.data = {"categories": [], "images": [], "annotations": []}

        # Convert CVAT categories into COCO categories.
        for i, category in enumerate(cvat_categories):
            coco_category = COCOAnnotations.create_category_instance(i + 1, category["name"])
            coco_annotations.data["categories"].append(coco_category)

        categories_by_name = to_dict(coco_annotations.data["categories"], "name")

        # Finally, Convert CVAT annotations to COCO annotations
        for i, image in enumerate(cvat_images):
            coco_image = COCOAnnotations.create_image_instance(i + 1, image["name"], image["width"], image["height"])
            coco_annotations.data["images"].append(coco_image)

            category_id = categories_by_name[image["tag"]["label"]][0]["id"]
            bbox = [0, 0, coco_image["width"], coco_image["height"]]
            coco_annotation = COCOAnnotations.create_annotation_instance(i + 1, coco_image["id"], category_id, bbox)
            coco_annotations.data["annotations"].append(coco_annotation)

        return coco_annotations
