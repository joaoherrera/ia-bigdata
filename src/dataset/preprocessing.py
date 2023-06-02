from ctypes import ArgumentError
from typing import Any, Callable, List, Tuple

import numpy as np
import cv2

class OrderedCompose:
    def __init__(self, funcs: List[Callable], **kwargs: dict) -> None:
        self.funcs = funcs
        self.kwargs = kwargs

    def __call__(self, image: np.ndarray, annotations: dict) -> Any:
        for func in self.funcs:
            image, annotations = func(image, annotations, **self.kwargs)

        return image, annotations


class CocoPreprocessing:
    @staticmethod
    def crop(image: np.ndarray, annotations: dict, **kwargs: dict) -> Tuple[np.ndarray, dict]:
        bbox = np.array(annotations["bbox"], dtype=np.int32)
        
        if "format" in kwargs.keys():
            format = kwargs["format"]

            if format == "channel_last":
                return image[bbox[1] : (bbox[1] + bbox[3]), bbox[0] : (bbox[0] + bbox[2]), :], annotations

        return image[:, bbox[1] : (bbox[1] + bbox[3]), bbox[0] : (bbox[0] + bbox[2])], annotations
        
    @staticmethod
    def resize(image: np.ndarray, annotations: dict, **kwargs: dict) -> Tuple[np.ndarray, dict]:
        if "size" in kwargs.keys():
            size = kwargs["size"]
            return cv2.resize(image, size), annotations
        else:
            raise ArgumentError("Size is not specified.")
                