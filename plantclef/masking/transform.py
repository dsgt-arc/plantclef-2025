import io
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import Model
from typing import List
from plantclef.model_setup import (
    setup_segment_anything_checkpoint_path,
    setup_groundingdino_checkpoint_path,
    setup_groundingdino_config_path,
)

from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, BinaryType


class HasCheckpointPathSAM(Param):
    """
    Mixin for param checkpoint_path: str
    """

    checkpointPathSAM = Param(
        Params._dummy(),
        "checkpointPathSAM",
        "The path to the segment-anything checkpoint weights",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__(
            default=setup_segment_anything_checkpoint_path(),
            doc="The path to the segment-anything checkpoint weights",
        )

    def getCheckpointPathSAM(self) -> str:
        return self.getOrDefault(self.checkpointPathSAM)


class HasEncoderVersion(Param):
    """
    Mixin for param encoder_version: str
    """

    encoderVersion = Param(
        Params._dummy(),
        "encoderVersion",
        "The name of the SAM encoder version to use",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__(
            default="vit_h",
            doc="The name of the SAM encoder version to use",
        )

    def getEncoderVersion(self) -> str:
        return self.getOrDefault(self.encoderVersion)


class HasCheckpointPathGroundingDINO(Param):
    """
    Mixin for param checkpoint_path_groundingdino: str
    """

    checkpointPathGroundingDINO = Param(
        Params._dummy(),
        "checkpointPathGroundingDINO",
        "The path to the GroundingDINO checkpoint weights",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__(
            default=setup_groundingdino_checkpoint_path(),
            doc="The path to the GroundingDINO checkpoint weights",
        )

    def getCheckpointPathGroundingDINO(self) -> str:
        return self.getOrDefault(self.checkpointPathGroundingDINO)


class HasConfigPathGroundingDINO(Param):
    """
    Mixin for param config_path_groundingdino: str
    """

    configPathGroundingDINO = Param(
        Params._dummy(),
        "configPathGroundingDINO",
        "The path to the GroundingDINO config path",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__(
            default=setup_groundingdino_config_path(),
            doc="The path to the GroundingDINO config path",
        )

    def getConfigPathGroundingDINO(self) -> str:
        return self.getOrDefault(self.configPathGroundingDINO)


class HasBatchSize(Param):
    """
    Mixin for param batch_size: int
    """

    batchSize = Param(
        Params._dummy(),
        "batchSize",
        "The batch size to use for embedding extraction",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self):
        super().__init__(
            default=32,
            doc="The batch size to use for embedding extraction",
        )

    def getBatchSize(self) -> int:
        return self.getOrDefault(self.batchSize)


class WrappedMasking(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasCheckpointPathSAM,
    HasEncoderVersion,
    HasCheckpointPathGroundingDINO,
    HasConfigPathGroundingDINO,
    HasBatchSize,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for fine-tuned DINOv2 to add it to the pipeline.
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "masks",
        checkpoint_path_sam: str = setup_segment_anything_checkpoint_path(),
        checkpoint_path_groundingdino: str = setup_groundingdino_checkpoint_path(),
        config_path_groundingdino: str = setup_groundingdino_config_path(),
        encoder_version: str = "vit_h",
        batch_size: int = 32,
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            checkpointPathSAM=checkpoint_path_sam,
            encoderVersion=encoder_version,
            checkpointPathGroundingDINO=checkpoint_path_groundingdino,
            configPathGroundingDINO=config_path_groundingdino,
            batchSize=batch_size,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Segment Anything Model
        sam = sam_model_registry[self.getEncoderVersion()](
            checkpoint=self.getCheckpointPathSAM()
        ).to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        # GroundingDINO Model
        self.groundingdino_model = Model(
            model_config_path=self.getConfigPathGroundingDINO(),
            model_checkpoint_path=self.getCheckpointPathGroundingDINO(),
        )
        # params for groundingdino
        self.CLASSES = [
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
        self.BOX_THRESHOLD = 0.15
        self.TEXT_THRESHOLD = 0.1
        self.INCLUDE_CLASS_IDS = {0, 1, 6}
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

    def segment(
        self,
        sam_predictor: SamPredictor,
        image: np.ndarray,
        xyxy: np.ndarray,
    ) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def enhance_class_name(self, class_names: List[str]) -> List[str]:
        return [f"all {class_name}s" for class_name in class_names]

    def mask_to_bytes(self, image_array):
        # convert ndarray to image
        mask_uint8 = image_array.astype("uint8")
        img = Image.fromarray(mask_uint8, mode="L")
        img_bytes = io.BytesIO()
        # save image as PNG
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()
        return img_bytes

    def empty_png(self, shape):
        # Create an empty image (all zeros) of the given shape.
        empty_array = np.zeros(shape, dtype=np.uint8)
        return self.mask_to_bytes(empty_array)

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        # check on the nvidia stats when generating the predict function
        self._nvidia_smi()

        def predict(input_image: np.ndarray) -> np.ndarray:
            # convert binary to RGB
            pil_image = Image.open(io.BytesIO(input_image)).convert("RGB")
            image_np = np.array(pil_image)

            detections = self.groundingdino_model.predict_with_classes(
                image=image_np,
                classes=self.enhance_class_name(class_names=self.CLASSES),
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD,
            )

            # Generate masks
            detections.mask = self.segment(
                sam_predictor=self.sam_predictor,
                image=image_np,
                xyxy=detections.xyxy,
            )

            merged_masks = []
            class_mask_results = {"leaf": None, "flower": None, "plant": None}
            grouped = {}
            for mask, class_id in zip(detections.mask, detections.class_id):
                grouped.setdefault(class_id, []).append(mask)

            for class_id, masks in grouped.items():
                if class_id in self.INCLUDE_CLASS_IDS and len(masks) > 0:
                    merged_mask = np.any(np.stack(masks, axis=0), axis=0)
                    merged_masks.append(merged_mask)
                    for image_mask in merged_masks:
                        # mask_uint8 = (merged_mask.astype(np.uint8)) * 255
                        if self.CLASSES[class_id] in class_mask_results:
                            class_mask_results[self.CLASSES[class_id]] = (
                                self.mask_to_bytes(image_mask)
                            )

            final_mask = np.any(np.stack(merged_masks, axis=0), axis=0)
            # final_mask_uint8 = (final_mask.astype(np.uint8)) * 255
            final_mask_bytes = self.mask_to_bytes(
                final_mask,
            )

            return {
                "combined_mask": final_mask_bytes
                if final_mask_bytes is not None
                else self.empty_png(image_np.shape[:2]),
                "leaf_mask": class_mask_results.get("leaf")
                or self.empty_png(image_np.shape[:2]),
                "flower_mask": class_mask_results.get("flower")
                or self.empty_png(image_np.shape[:2]),
                "plant_mask": class_mask_results.get("plant")
                or self.empty_png(image_np.shape[:2]),
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

    # def _transform(self, df: DataFrame):
    #     return df.withColumn(
    #         self.getOutputCol(),  # a single column name
    #         predict_batch_udf(
    #             make_predict_fn=self._make_predict_fn,
    #             return_type=StructType(
    #                 [
    #                     StructField("leaf_mask", ArrayType(FloatType()), False),
    #                     StructField("flower_mask", ArrayType(FloatType()), False),
    #                     StructField("plant_mask", ArrayType(FloatType()), False),
    #                     StructField("final_mask", ArrayType(FloatType()), False),
    #                 ]
    #             ),
    #             batch_size=self.getBatchSize(),
    #         )(self.getInputCol()),
    #     )
