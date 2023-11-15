# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# A COCO dataset class that should be used with PyTorch.                                                              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import numpy as np
import torch

from src.dataset.annotations_coco import COCOAnnotations
from src.dataset.dataset_base import MutableDataset
from src.dataset.dataset_utils import read_image


@dataclass
class CocoDataset(MutableDataset):
    data_directory_path: str
    data_annotation_path: str
    augmentations: Callable = None
    preprocessing: Callable = None
    seed: Any | None = None
    balancing_strategy: str | None = None

    def __post_init__(self) -> None:
        """Initialize important attributes of the object. This function is automatically called after the
        object has been created."""

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        super().__init__()

        self.tree = COCOAnnotations(self.data_annotation_path, self.balancing_strategy)
        self.images = COCOAnnotations.to_dict(self.tree.data["images"], "id")
        self.categories = COCOAnnotations.to_dict(self.tree.data["categories"], "id")
        self.annotations = self.tree.data.get("annotations")

        self.preview_dataset()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the image and category data for a given index.

        Args:
            idx (int): The index of the data to retrieve.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image tensor and the category tensor.
        """

        annotation = self.annotations[idx]
        image_data = self.images[annotation["image_id"]]
        image_path = os.path.join(self.data_directory_path, image_data[0]["file_name"])
        image = read_image(image_path)

        # apply preprocessing
        if self.preprocessing is not None:
            image, annotation = self.preprocessing(image, annotation)

        # apply augmentations
        if self.augmentations is not None:
            image, annotation = self.augmentations(image, annotation)

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # (H, W, C) -> (C, H, W)
        category = torch.tensor(self.categories[annotation["category_id"]][0]["id"] - 1, dtype=torch.float32)

        return image, category

    def __len__(self) -> int:
        """Returns the length of the object.

        Returns:
            int: The length of the object.
        """

        return len(self.annotations)

    def split(self, *percentages: float, random: bool) -> Tuple[Any, ...]:
        """Splits the dataset into subsets based on the given percentages.

        Args:
            *percentages (float): The percentages to split the dataset into subsets. The sum of percentages must be
            equal to 1.
            random (bool): Determines whether to shuffle the images before splitting.
        Returns:
            Tuple: A tuple of subsets, each containing a portion of the dataset.
        """

        assert np.sum([*percentages]) == 1, "Summation of percentages must be equal to 1."

        subsets = []
        all_images = list(self.images.keys())
        total_images = len(all_images)
        subset_sizes = [int(total_images * perc) for perc in percentages]

        if random:
            np.random.shuffle(all_images)

        for ss in subset_sizes:
            subset = deepcopy(self)
            indexes = all_images[:ss]

            subset.tree.data["images"] = [self.images[i][0] for i in indexes]
            subset.images = COCOAnnotations.to_dict(subset.tree.data["images"], "id")

            image_annotations = COCOAnnotations.to_dict(subset.tree.data["annotations"], "image_id")
            subset.tree.data["annotations"] = [image_annotations[image_id][0] for image_id in subset.images.keys()]
            subset.tree.data["annotations"] = np.array(subset.tree.data["annotations"]).flatten().tolist()
            subset.annotations = subset.tree.data.get("annotations")

            subsets.append(subset)
            all_images = all_images[ss:]

        return tuple(subsets)

    def preview_dataset(self) -> None:
        """Prints a preview of the dataset by displaying some dataset statistics."""

        horizontal_bar_length = 100
        categories_str = [f"{c['id']}: {c['name']}" for c in self.tree.data["categories"]]

        print("=" * horizontal_bar_length)
        print(f"Dataset categories: {categories_str}")
        print(f"Number of images: {len(self.tree.data['images'])}")
        print(f"Number of Annotations: {len(self.tree.data['annotations'])}")
        print("=" * horizontal_bar_length)
        print("Per-category info:")

        images_per_category = COCOAnnotations.to_dict(self.tree.data["annotations"], "category_id")

        for c in self.tree.data["categories"]:
            print(f"Category Label: {c['name']} \t Category ID: {c['id']}")
            print(f"Instances: {len(images_per_category[c['id']])}")
        print("=" * horizontal_bar_length)
