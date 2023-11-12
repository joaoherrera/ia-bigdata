# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ALbumentations is a Python library for image augmentation. The Compose function is used to apply                    #
# multiple augmentations at once. To visualize the effects of each algumentation algorithm on an image, please visit: #
# Albumentations Demo: https://github.com/albumentations-team/albumentations-demo                                     #
# Official Albumentations repo: https://github.com/albumentations-team/albumentations                                 #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from typing import Dict, Tuple

import albumentations
import numpy as np


class Augmentations:
    transformer = albumentations.Compose(
        [
            albumentations.RandomRotate90(
                always_apply=False,
                p=0.5,
            ),
            albumentations.Downscale(
                always_apply=False,
                p=0.5,
                scale_min=0.87,
                scale_max=0.99,
            ),
            albumentations.GridDistortion(
                always_apply=False,
                p=0.5,
                num_steps=1,
                distort_limit=(-0.12, 0.07),
                interpolation=0,
                border_mode=4,
                value=(0, 0, 0),
                mask_value=None,
                normalized=False,
            ),
            albumentations.RandomBrightnessContrast(
                always_apply=False,
                p=0.5,
                brightness_limit=(-0.16, 0.52),
                contrast_limit=(-0.3, 0.2),
                brightness_by_max=True,
            ),
            albumentations.ShiftScaleRotate(
                always_apply=False,
                p=0.5,
                shift_limit_x=(-0.01, -0.01),
                shift_limit_y=(-0.01, -0.01),
                scale_limit=(-0.07999999999999996, 0.8599999999999999),
                rotate_limit=(5, 0),
                interpolation=4,
                border_mode=4,
                value=(0, 0, 0),
                mask_value=None,
                rotate_method="largest_box",
            ),
            albumentations.HueSaturationValue(
                always_apply=False,
                p=0.5,
                hue_shift_limit=(-30, 30),
                sat_shift_limit=(-30, 30),
                val_shift_limit=(-30, 30),
            ),
        ]
    )

    @classmethod
    def augment(cls, image: np.ndarray, annotations: Dict, **kwargs: Dict) -> Tuple[np.ndarray, Dict]:
        """Augments an image and its annotations.

        Args:
            image (np.ndarray): The input image.
            annotations (Dict): The annotations associated with the image.
            **kwargs (Dict): Additional keyword arguments. Currently not used. Created for matching with other methods
            called by src.dataset.preprocessing.OrderedCompose class.

        Returns:
            Tuple[np.ndarray, Dict]: A tuple containing the augmented image and the updated annotations.
        """

        return cls.transformer(image=image)["image"], annotations
