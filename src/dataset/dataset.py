import copy
import os
from abc import ABC, abstractmethod
from ctypes import ArgumentError
from dataclasses import dataclass
from operator import index
from typing import Any, Callable, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset.annotations import COCOAnnotations


class DatasetUtils:
    """Base operations for datasets."""

    @staticmethod
    def read_image(image_path: str, read_mode: int = cv2.IMREAD_COLOR, channel_first: bool = False) -> np.ndarray:
        if not os.path.exists(image_path):
            raise ValueError(f"Path {image_path} does not exist.")

        image = cv2.imread(image_path, read_mode)

        if channel_first:
            return image.transpose(2, 0, 1)
        else:
            return image

    @staticmethod
    def read_paths(directory_path: str) -> list[str]:
        if not os.path.isdir(directory_path):
            raise ValueError(f"Path {directory_path} is not a directory.")

        return os.listdir(directory_path)


class BaseDataset(ABC):
    def __init__(self) -> None:
        self.data = None

    def is_empty(self) -> bool:
        return self.data is None or len(self.data) == 0

    def set_data(self, data) -> None:
        self.data = data

    def get_data(self) -> Any:
        return self.data


class MutableDataset(BaseDataset, ABC):
    @abstractmethod
    def split(self) -> Tuple[Any, Any]:
        raise NotImplementedError()


@dataclass
class CocoDataset(Dataset, MutableDataset):
    data_directory_path: str
    data_annotation_path: str
    augmentations: Callable = None
    preprocessing: Callable = None
    seed: Any | None = None
    balancing_strategy: str | None = None

    def __post_init__(self) -> None:
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        super().__init__()

        self.tree = COCOAnnotations(self.data_annotation_path, self.balancing_strategy)
        self.images = COCOAnnotations.to_dict(self.tree.data["images"], "id")
        self.categories = COCOAnnotations.to_dict(self.tree.data["categories"], "id")
        self.annotations = self.tree.data.get("annotations")

        self.preview_dataset()

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        annotation = self.annotations[idx]
        image_data = self.images[annotation["image_id"]]
        image_path = os.path.join(self.data_directory_path, image_data[0]["file_name"])
        image = DatasetUtils.read_image(image_path)

        # apply preprocessing
        if self.preprocessing is not None:
            image, annotation = self.preprocessing(image, annotation)

        # apply augmentations
        if self.augmentations is not None:
            image, annotation = self.augmentations(image, annotation)

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        category = torch.tensor(self.categories[annotation["category_id"]][0]["id"] - 1, dtype=torch.float32)

        return image, category

    def __len__(self) -> int:
        return len(self.annotations)

    def split(self, *percentages, random: bool) -> Tuple[Any, ...]:
        assert np.sum([*percentages]) == 1, "Summation of percentages must be equal to 1."

        subsets = []
        all_images = list(self.images.keys())
        total_images = len(all_images)
        subset_sizes = [int(total_images * perc) for perc in percentages]

        if random:
            np.random.shuffle(all_images)

        for ss in subset_sizes:
            subset = copy.deepcopy(self)
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
        horizontal_bar_length = 100
        categories_str = [f"{c['id']}: {c['name']}" for c in self.tree.data['categories']]

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
        

    @classmethod
    def dataloader(cls, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(cls, batch_size=batch_size, shuffle=shuffle)
