import io
import numpy as np
import torch
from PIL import Image
from transformers import (
    SamModel,
    SamProcessor,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
from typing import List

from .params import (
    HasCheckpointPathSAM,
    HasCheckpointPathGroundingDINO,
    HasBatchSize,
)

from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, BinaryType


class WrappedMasking(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasCheckpointPathSAM,
    HasCheckpointPathGroundingDINO,
    HasBatchSize,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for fine-tuned DINOv2 to add it to the pipeline.
    """

    def __init__(
        self,
        input_col: str = "data",
        output_col: str = "masks",
        checkpoint_path_sam: str = "facebook/sam-vit-huge",
        checkpoint_path_groundingdino: str = "IDEA-Research/grounding-dino-base",
        batch_size: int = 32,
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            # NOTE: these are more ids than they are checkpoint paths
            checkpointPathSAM=checkpoint_path_sam,
            checkpointPathGroundingDINO=checkpoint_path_groundingdino,
            batchSize=batch_size,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # https://huggingface.co/docs/transformers/main/en/model_doc/sam
        self.sam_model = SamModel.from_pretrained(checkpoint_path_sam).to(self.device)
        self.sam_processor = SamProcessor.from_pretrained(checkpoint_path_sam)
        # https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino#transformers.GroundingDinoForObjectDetection
        self.groundingdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            checkpoint_path_groundingdino
        ).to(self.device)
        self.groundingdino_processor = AutoProcessor.from_pretrained(
            checkpoint_path_groundingdino
        )
        # params for groundingdino
        self.CLASSES = [
            "leaf",
            "flower",
            "plant",
            "sand",
            "wood",
            "stone",
            "tape",
            "tree",
            "rock",
            "vegetation",
        ]
        self.BOX_THRESHOLD = 0.15
        self.TEXT_THRESHOLD = 0.1
        self.INCLUDE_CLASS_IDS = {0, 1, 2}
        self.INITIALIZED = True

    def _nvidia_smi(self):
        from subprocess import run, PIPE

        try:
            result = run(
                ["nvidia-smi"], check=True, stdout=PIPE, stderr=PIPE, text=True
            )
            print("=== GPU Utilization (before/after prediction) ===")
            print(result.stdout)
        except Exception as e:
            print(f"nvidia-smi failed: {e}")

    def enhance_class_name(self, class_names: List[str]) -> List[str]:
        return [f"all {class_name}" for class_name in class_names]

    def detect(self, image: Image) -> dict:
        # predict with groundingdino
        inputs = self.groundingdino_processor(
            images=image,
            text=self.enhance_class_name(class_names=self.CLASSES),
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.groundingdino_model(**inputs)

        # dictionary with boxes, scores, text_labels
        return self.groundingdino_processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
            target_sizes=[(image.height, image.width)],
        )[0]  # return the dictionary inside the list

    def convert_boxes_to_tensor(self, detections: dict) -> torch.tensor:
        input_boxes = torch.tensor(
            detections["boxes"].cpu().numpy(), dtype=torch.float32
        ).unsqueeze(0)
        return input_boxes

    def segment(self, image: Image, input_boxes: torch.tensor) -> np.ndarray:
        inputs = self.sam_processor(
            image, input_boxes=input_boxes, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # convert to numpy
        return np.array(masks)

    def mask_to_bytes(self, image_array: np.ndarray) -> bytes:
        """Encode the numpy mask array as raw bytes using np.save()."""
        buffer = io.BytesIO()
        np.save(buffer, image_array)  # save array to buffer
        return buffer.getvalue()  # convert buffer to bytes

    def group_masks_by_class(self, masks: np.ndarray, scores: torch.Tensor) -> dict:
        """Groups segmentation masks by class ID using the highest confidence score."""
        grouped = {}
        class_ids = torch.argmax(scores, dim=-1)

        # ensure class_ids is iterable
        if class_ids.dim() == 0:
            class_ids = class_ids.unsqueeze(0)

        for mask, class_id in zip(masks, class_ids.tolist()):
            grouped.setdefault(class_id, []).append(mask)

        return grouped

    def empty_array(self, shape) -> bytes:
        """Create an empty numpy mask (H, W) and returns it as bytes."""
        empty_mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        return self.mask_to_bytes(empty_mask)  # store as raw bytes

    def merge_masks(self, grouped_masks: dict, empty_shape) -> tuple:
        """Merges masks for each class and prepares the output dictionary."""

        class_masks = {
            key: np.zeros(empty_shape, dtype=np.uint8)
            for key in ["leaf", "flower", "plant"]
        }

        for class_id, masks in grouped_masks.items():
            if not masks or class_id not in self.INCLUDE_CLASS_IDS:
                continue
            class_name = self.CLASSES[class_id]
            for mask in masks:
                class_masks[class_name] |= mask

        final_mask = np.zeros(empty_shape, dtype=np.uint8)
        for _, mask in class_masks.items():
            final_mask |= mask

        return final_mask, class_masks

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        # check on the nvidia stats when generating the predict function
        self._nvidia_smi()

        def predict(input_image: np.ndarray) -> np.ndarray:
            # convert binary to RGB
            image = Image.open(io.BytesIO(input_image)).convert("RGB")
            detections = self.detect(image)  # returns dictionary of detections
            input_boxes = self.convert_boxes_to_tensor(detections)
            masks = self.segment(image, input_boxes=input_boxes)

            grouped_masks = self.group_masks_by_class(masks, detections["scores"])
            final_mask, class_masks = self.merge_masks(
                grouped_masks, (image.height, image.width)
            )

            # print size of the final mask and mask results
            # TODO: remove this later
            if True:
                print(f"Final mask size: {final_mask.shape}", flush=True)
                for k, v in class_masks.items():
                    print(f"{k} mask size: {v.shape}", flush=True)

            return {
                "combined_mask": self.mask_to_bytes(final_mask),
                **{f"{k}_mask": self.mask_to_bytes(v) for k, v in class_masks.items()},
            }

        return predict

    def _transform(self, df: DataFrame):
        predict_udf = F.udf(
            self._make_predict_fn(),
            returnType=StructType(
                [
                    StructField("combined_mask", BinaryType(), False),
                    StructField("leaf_mask", BinaryType(), False),
                    StructField("flower_mask", BinaryType(), False),
                    StructField("plant_mask", BinaryType(), False),
                ]
            ),
        )
        return df.withColumn(
            self.getOutputCol(), predict_udf(F.col(self.getInputCol()))
        )
