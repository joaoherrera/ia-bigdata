""" Given an annotation file, split the dataset into train and test subsets.
"""

from argparse import ArgumentParser
from typing import Dict
from src.dataset.annotations import COCOAnnotations
from src.dataset.dataset import CocoDataset
import os

def main(args: Dict) -> None:
    annotations_path = args.get("annotations_path")
    output_path = args.get("output_path")
    splits = tuple(args.get("split"))
    split_labels = []
        
    if sum(splits) != 1:
        raise ValueError("Sum of splits must be 1.")
    
    match len(splits):
        case 2:
            split_labels = ["training", "validation"]
        case 3:
            split_labels = ["training", "validation", "test"]
        case _:
            split_labels = [str(n) for n in len(splits)]
    
    # Load annotations
    annotations = COCOAnnotations(annotations_path)
    datasets = CocoDataset(data_directory_path=None, data_annotation_path=annotations_path).split(*splits, random=True)
    
    for dataset, label in zip(datasets, split_labels):
        dataset.tree.save(os.path.join(output_path, f"{label}_annotations.json")) 
    

def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Split dataset into subsets for training and testing.")

    parser.add_argument(
        "--annotations-path",
        type=str,
        help="Path to the annotation file.",
        required=True
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output file.",
        required=True
    )
    
    parser.add_argument(
        "--split",
        type=float,
        help="Split percentage.",
        nargs='+',
        required=True
    )

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(vars(args))
    