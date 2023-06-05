from argparse import ArgumentParser
from src.dataset.annotations import CVATAnnotations
from typing import Dict


def main(args: Dict) -> None:
    cvat_annotations = CVATAnnotations(args.get("input_annotations"))
    coco_annotations = cvat_annotations.convert_to_coco()
    coco_annotations.save(args.get("output_annotations"))


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Create a new dataset where each image is a bounding box annotation.")

    parser.add_argument(
        "--input-annotations",
        type=str,
        required=True,
        help="Path to the directory containing images with annotations.",
    )
    parser.add_argument(
        "--output-annotations",
        type=str,
        help="Path to the output directory.",
    )

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(vars(args))