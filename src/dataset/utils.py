import os

from src.dataset.annotations import COCOAnnotations


class DatasetCustomizer:
    @staticmethod
    def to_patches(annotations: COCOAnnotations) -> COCOAnnotations:
        images_group = annotations.to_dict(annotations.data["images"], "id")
        annotations_group = annotations.to_dict(annotations.data["annotations"], "image_id")

        cc_ann = COCOAnnotations.from_data(dict())
        data = cc_ann.data
        data["categories"] = annotations.data["categories"]
        annotations_index = 1

        for image_id, annotation_list in annotations_group.items():
            image_basename = os.path.basename(images_group[image_id][0]["file_name"])
            image_basename, image_extension = os.path.splitext(image_basename)

            for index, annotation in enumerate(annotation_list):
                image_name = f"{image_basename}_{index + 1}{image_extension}"
                image_dimensions = (annotation["bbox"][2], annotation["bbox"][3])

                segmentation = annotation["segmentation"]
                if len(annotation["segmentation"]) > 0:
                    segmentation[::2] -= annotation["bbox"][0]
                    segmentation[1::2] -= annotation["bbox"][1]

                data["images"].append(
                    COCOAnnotations.create_image_instance(
                        id=annotations_index,
                        file_name=image_name,
                        width=image_dimensions[0],
                        height=image_dimensions[1],
                    )
                )
                data["annotations"].append(
                    COCOAnnotations.create_annotation_instance(
                        id=annotations_index,
                        image_id=annotations_index,
                        cateogory_id=annotation["category_id"],
                        bbox=[[0, 0, image_dimensions[0], image_dimensions[1]]],
                        segmentation=segmentation,
                        history=annotation["bbox"],
                    )
                )

                annotations_index += 1

        return cc_ann
