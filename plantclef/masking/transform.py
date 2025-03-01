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

from ..serde import serialize_mask
from .params import HasCheckpointPathSAM, HasCheckpointPathGroundingDINO

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
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            # NOTE: these are more ids than they are checkpoint paths
            checkpointPathSAM=checkpoint_path_sam,
            checkpointPathGroundingDINO=checkpoint_path_groundingdino,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # https://huggingface.co/docs/transformers/main/en/model_doc/sam
        self.sam_model = SamModel.from_pretrained(checkpoint_path_sam).to(self.device)
        self.sam_processor = SamProcessor.from_pretrained(checkpoint_path_sam)
        # https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino
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
            "tape",
            "tree",
            "rock",
            "vegetation",
        ]
        self.BOX_THRESHOLD = 0.15
        self.TEXT_THRESHOLD = 0.1

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

    def _refine_masks(self, masks: torch.BoolTensor) -> List[np.ndarray]:
        # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        return masks

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
        return self._refine_masks(masks[0])

    def merge_class_masks(
        self, masks: np.ndarray, text_labels: list[str], empty_shape: tuple
    ) -> tuple:
        """Merges masks for each class and prepares the output dictionary."""
        # create empty np arrays for empty masks
        class_masks = {
            key: np.zeros(empty_shape, dtype=np.uint8) for key in self.CLASSES
        }

        # return all masks
        for text_label, mask in zip(text_labels, masks):
            for key in class_masks.keys():
                if key not in text_label:
                    continue
                # merge multiple masks into a single mask
                class_masks[key] |= mask

        return class_masks

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        # check on the nvidia stats when generating the predict function
        self._nvidia_smi()

        def predict(input_image: np.ndarray) -> np.ndarray:
            # convert binary to RGB
            image = Image.open(io.BytesIO(input_image)).convert("RGB")
            detections = self.detect(image)  # dictionary of the detections
            input_boxes = self.convert_boxes_to_tensor(detections)
            masks = self.segment(image, input_boxes=input_boxes)

            class_masks = self.merge_class_masks(
                masks, detections["text_labels"], (image.height, image.width)
            )

            return {
                **{f"{k}_mask": serialize_mask(v) for k, v in class_masks.items()},
            }

        return predict

    def _transform(self, df: DataFrame):
        predict_udf = F.udf(
            self._make_predict_fn(),
            returnType=StructType(
                [
                    StructField("leaf_mask", BinaryType(), False),
                    StructField("flower_mask", BinaryType(), False),
                    StructField("plant_mask", BinaryType(), False),
                    StructField("sand_mask", BinaryType(), False),
                    StructField("wood_mask", BinaryType(), False),
                    StructField("tape_mask", BinaryType(), False),
                    StructField("tree_mask", BinaryType(), False),
                    StructField("rock_mask", BinaryType(), False),
                    StructField("vegetation_mask", BinaryType(), False),
                ]
            ),
        )
        return df.withColumn(
            self.getOutputCol(), predict_udf(F.col(self.getInputCol()))
        )
