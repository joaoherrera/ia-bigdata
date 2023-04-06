# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Date: 2023-04-01                                                                                          #
# Author: Joao Herrera                                                                                      #
#                                                                                                           #
# Script to merge multiples annotations files into a single one.                                            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from argparse import ArgumentParser
from annotations import COCOAnnotations


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Merge multiples annotation files.")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("annotation_file", metavar="ANNOTATION FILES", type=str, nargs="+")

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    coco_annotations = []

    for annotation_file_path in args.annotation_file:
        coco_annotation = COCOAnnotations(annotation_file_path)
        coco_annotation.load()
        coco_annotations.append(coco_annotation)

    merged_annotations = COCOAnnotations.merge(args.output_path, *coco_annotations)
    print(merged_annotations.table.sample(15))
