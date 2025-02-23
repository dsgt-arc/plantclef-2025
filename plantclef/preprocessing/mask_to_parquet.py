import os
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, BinaryType
from pyspark.sql.functions import udf
from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import Model
from typing import List
from plantclef.spark import get_spark

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_home_dir():
    return Path(os.path.expanduser("~"))


def segment(
    sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [f"all {class_name}s" for class_name in class_names]


def image_to_bytes(image_array, ext=".png"):
    success, buffer = cv2.imencode(ext, image_array)
    if not success:
        raise ValueError("Image encoding failed!")
    return buffer.tobytes()


def process_image(binary_data):
    try:
        if not hasattr(process_image, "initialized"):
            HOME = get_home_dir()
            SAM_ENCODER_VERSION = "vit_h"
            SAM_CHECKPOINT_PATH = os.path.join(
                HOME, "scratch/weights", "sam_vit_h_4b8939.pth"
            )
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sam = sam_model_registry[SAM_ENCODER_VERSION](
                checkpoint=SAM_CHECKPOINT_PATH
            ).to(device=DEVICE)
            process_image.sam_predictor = SamPredictor(sam)

            GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
                HOME, "scratch/weights", "groundingdino_swint_ogc.pth"
            )
            GROUNDING_DINO_CONFIG_PATH = os.path.join(
                HOME,
                "scratch/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            )

            process_image.grounding_dino_model = Model(
                model_config_path=GROUNDING_DINO_CONFIG_PATH,
                model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            )

            process_image.CLASSES = [
                "leaf",
                "flower",
                "sand",
                "wood",
                "stone",
                "tape",
                "plant",
                "tree",
                "rock",
                "vegetation",
            ]
            process_image.BOX_TRESHOLD = 0.15
            process_image.TEXT_TRESHOLD = 0.1
            process_image.include_class_ids = {0, 1, 6}
            process_image.initialized = True

        sam_predictor = process_image.sam_predictor
        grounding_dino_model = process_image.grounding_dino_model
        CLASSES = process_image.CLASSES
        BOX_TRESHOLD = process_image.BOX_TRESHOLD
        TEXT_TRESHOLD = process_image.TEXT_TRESHOLD
        include_class_ids = process_image.include_class_ids

        nparr = np.frombuffer(binary_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=CLASSES),
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
        )

        # Generate masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )

        merged_masks = []
        class_mask_results = {"leaf": None, "flower": None, "plant": None}
        grouped = {}
        for mask, class_id in zip(detections.mask, detections.class_id):
            grouped.setdefault(class_id, []).append(mask)

        for class_id, masks in grouped.items():
            if class_id in include_class_ids and len(masks) > 0:
                merged_mask = np.any(np.stack(masks, axis=0), axis=0)
                merged_masks.append(merged_mask)
                mask_uint8 = (merged_mask.astype(np.uint8)) * 255
                if CLASSES[class_id] in class_mask_results:
                    class_mask_results[CLASSES[class_id]] = image_to_bytes(
                        mask_uint8, ext=".png"
                    )

        if merged_masks:
            final_mask = np.any(np.stack(merged_masks, axis=0), axis=0)
            final_mask_uint8 = (final_mask.astype(np.uint8)) * 255
            final_mask_bytes = image_to_bytes(final_mask_uint8, ext=".png")
        else:
            final_mask_bytes = None

        return (
            final_mask_bytes,
            class_mask_results["leaf"],
            class_mask_results["flower"],
            class_mask_results["plant"],
        )
    except Exception as e:
        sys.stderr.write(f"Error processing image: {e}\n")
        return (None, None, None, None)


def parse_args():
    home_dir = get_home_dir()
    dataset_base_path = f"{home_dir}/shared/plantclef/data"

    parser = argparse.ArgumentParser(
        description="Process test image dataset stored on PACE."
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=os.cpu_count(),
        help="Number of cores used in Spark driver",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="16g",
        help="Amount of memory to use in Spark driver",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=f"{dataset_base_path}/parquet/test_2024",
        help="Path to the existing Parquet file.",
    )
    parser.add_argument(
        "--image-root-path",
        type=str,
        default=f"{dataset_base_path}/test/",
        help="Base directory path for image data",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=f"{home_dir}/scratch/masks_2024",
        help="PACE path for output Parquet files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_schema = StructType(
        [
            StructField("final_mask", BinaryType(), True),
            StructField("leaf", BinaryType(), True),
            StructField("flower", BinaryType(), True),
            StructField("plant", BinaryType(), True),
        ]
    )

    process_image_udf = udf(process_image, output_schema)

    spark = get_spark(cores=args.cores, memory=args.memory)
    existing_df = spark.read.parquet(args.input_path)
    updated_df = (
        existing_df.withColumn("mask_struct", process_image_udf(F.col("data")))
        .withColumn("final_mask", F.col("mask_struct").getField("final_mask"))
        .withColumn("leaf", F.col("mask_struct").getField("leaf"))
        .withColumn("flower", F.col("mask_struct").getField("flower"))
        .withColumn("plant", F.col("mask_struct").getField("plant"))
        .drop("mask_struct")
    )

    updated_df = updated_df.repartition(2000)
    updated_df.write.mode("overwrite").parquet(args.output_path)
    print("Updated Parquet file written to:", args.output_path)


if __name__ == "__main__":
    main()
