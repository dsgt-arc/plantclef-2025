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
        # scores = outputs.iou_scores
        # convert to numpy
        return np.array(masks)

    def enhance_class_name(self, class_names: List[str]) -> List[str]:
        return [f"all {class_name}" for class_name in class_names]

    def mask_to_bytes(self, image_array):
        # ensure the input is uint8 (0-255)
        mask_uint8 = np.clip(image_array, 0, 255).astype(np.uint8)
        # convert grayscale mask to 3-channel (RGB)
        if mask_uint8.ndim == 2:  # if single-channel, expand to 3 channels
            mask_rgb = np.stack([mask_uint8] * 3, axis=-1)  # Convert to (H, W, 3)
        else:
            mask_rgb = mask_uint8  # Assume it's already (H, W, 3)
        # convert to RGB image
        img = Image.fromarray(mask_rgb, mode="RGB")
        # save as PNG to a bytes buffer
        img_bytes_io = io.BytesIO()
        img.save(img_bytes_io, format="PNG")
        # retrieve bytes and return
        return img_bytes_io.getvalue()

    def empty_png(self, shape):
        # Create an empty RGB image (height, width, 3)
        empty_array = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        return self.mask_to_bytes(empty_array)

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        # check on the nvidia stats when generating the predict function
        self._nvidia_smi()

        def predict(input_image: np.ndarray) -> np.ndarray:
            # convert binary to RGB
            image = Image.open(io.BytesIO(input_image)).convert("RGB")

            detections = self.detect(image)  # returns list of dictionaries
            # # move tensors to CPU
            # detections = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in detections.items()}
            # get input_boxes as tensors for segmentation
            input_boxes = self.convert_boxes_to_tensor(detections)
            masks = self.segment(image, input_boxes=input_boxes)

            empty_image = self.empty_png(image.size[::-1])  # extract (H, W)
            class_mask_results = {
                "leaf": empty_image,
                "flower": empty_image,
                "plant": empty_image,
            }
            grouped = {}
            for mask, class_id in zip(masks, torch.argmax(detections["scores"], dim=0)):
                grouped.setdefault(class_id, []).append(mask)

            # get the masks for the classes we are interested in
            merged_masks = []
            for class_id, masks in grouped.items():
                if class_id in self.INCLUDE_CLASS_IDS and len(masks) > 0:
                    merged_mask = torch.any(torch.stack(masks, dim=0), dim=0)
                    merged_masks.append(merged_mask.cput().numpy())
                    class_name = self.CLASSES[class_id]
                    class_mask_results[class_name] = self.mask_to_bytes(
                        merged_mask.cpu().numpy()
                    )

            # merge the masks
            final_mask = torch.any(torch.stack(merged_masks, dim=0), dim=0)
            final_mask_bytes = self.mask_to_bytes(final_mask.cpu().numpy())

            # make sure masks are in binary
            combined = final_mask_bytes or empty_image

            return {
                "combined_mask": combined,
                **{f"{k}_mask": v for k, v in class_mask_results.items()},
            }

        return predict

    def _transform(self, df: DataFrame):
        predict_fn = self._make_predict_fn()
        predict_udf = F.udf(
            predict_fn,
            StructType(
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
