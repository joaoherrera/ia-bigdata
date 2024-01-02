"""Extract patches from a set of images and save them on disk.
"""
from argparse import ArgumentParser
from typing import Dict

from src.dataset.dataset_coco import CocoDatasetInstanceSegmentation


def extract_patches(args: Dict) -> None:
    dataset = CocoDatasetInstanceSegmentation(args.get("images_path"), args.get("annotations_path"))
    patch_size = args.get("patch_size")
    stride = args.get("stride")
    min_area_ratio = args.get("min_area_ratio")
    output_path = args.get("output_path")

    dataset.extract_patches(output_path, patch_size, stride, min_area_ratio)


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser("Extract patches from a set of images and save them on disk.")

    parser.add_argument(
        "--images-path",
        type=str,
        help="Path to a directory containing the images to extract patches.",
        required=True,
    )

    parser.add_argument(
        "--annotations-path",
        type=str,
        help="Path to the annotation file.",
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output directory.",
        required=True,
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        help="Patch size.",
        default=512,
        required=False,
    )

    parser.add_argument(
        "--stride",
        type=int,
        help="Stride.",
        default=1,
        required=False,
    )

    parser.add_argument(
        "--min-area-ratio",
        type=float,
        help="Patches with an area smaller than this ratio will be discarded.",
        default=0.05,
        required=False,
    )

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    extract_patches(vars(args))
