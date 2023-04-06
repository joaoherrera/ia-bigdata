from typing import Any, Callable, List, Tuple

import numpy as np


class OrderedCompose:
    def __init__(self, funcs: List[Callable]) -> None:
        self.funcs = funcs

    def __call__(self, image: np.ndarray, annotations: dict) -> Any:
        for func in self.funcs:
            image, annotations = func(image, annotations)
        
        return image, annotations


class CocoPreprocessing:
    @staticmethod
    def crop(image: np.ndarray, annotations: dict) -> Tuple[np.ndarray, dict]:
        bbox = np.array(annotations["bbox"], dtype=np.int32)
        return image[:, bbox[1]: (bbox[1] + bbox[3]), bbox[0]: (bbox[0] + bbox[2])], annotations