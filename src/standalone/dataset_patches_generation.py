import os
from argparse import ArgumentParser
from copy import copy
from typing import Dict

import cv2
from torch.utils import data

from src.dataset.annotations import COCOAnnotations
from src.dataset.preprocessing import CocoPreprocessing
from src.dataset.utils import DatasetCustomizer


def main(args: Dict) -> None:
    input_directory = args.get("input_directory")
    input_annotations = args.get("input_annotations")
    output_directory = args.get("output_directory")
    output_file = f"{os.path.join(output_directory, os.path.splitext(os.path.basename(input_annotations))[0])}_new.json"

    if not os.path.isdir(input_directory):
        raise NotADirectoryError()

    if not os.path.isfile(input_annotations):
        raise FileNotFoundError()

    os.makedirs(output_directory, exist_ok=True)

    print("Creating annotation file...")
    dataset = COCOAnnotations(input_annotations)
    dataset_patches = DatasetCustomizer.to_patches(copy(dataset))
    dataset_patches.save(output_file)

    annotations_group = dataset.to_dict(dataset.data["annotations"], "image_id")

    print(f"Saving images to {output_directory}...")
    for image in dataset.data["images"]:
        im = cv2.imread(os.path.join(input_directory, image["file_name"]))
        image_name = os.path.splitext(os.path.basename(image["file_name"]))[0]

        if im is None:
            print(f"Image {image['file_name']} not found. Skipping...")
            continue

        for i, annotation in enumerate(annotations_group[image["id"]]):
            patch_name = f"{image_name}_{i + 1}.jpg"
            patch_image, __ = CocoPreprocessing.crop(im, annotation, format="channel_last")
            output_patch_path = os.path.join(output_directory, patch_name)

            try:
                cv2.imwrite(output_patch_path, patch_image)
            except Exception:
                print(f"Error when saving the patch {patch_name}. Skipping...")
                continue

    print("Done.")


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Create a new dataset where each image is a bounding box annotation.")

    parser.add_argument(
        "--input-directory",
        type=str,
        help="Path to the directory containing images with annotations.",
    )
    parser.add_argument(
        "--input-annotations",
        type=str,
        help="Path to the annotation file.",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        help="Path to the output directory.",
    )

    return parser


if __name__ == "__main__":
    main(vars(build_arg_parser().parse_args()))
