import cv2
import torch
import numpy as np
from PIL import Image
from plantclef.serde import deserialize_image, serialize_image

from .params import HasBatchSize, HasCheckpointPathGroundingDINO
from pyspark.sql import DataFrame
from pyspark.ml import Transformer
from pyspark.sql import functions as F
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    BinaryType,
    StructType,
    StructField,
)


class WrappedGroundingDINO(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasBatchSize,
    HasCheckpointPathGroundingDINO,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for using GroundingDINO to extract bounding boxes detections.
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        batch_size: int = 32,
        checkpoint_path_groundingdino: str = "IDEA-Research/grounding-dino-base",
    ):
        # NOTE(anthony): move the import outside of top-level module scope
        # because this import is doing a surprising amount of work making the cli
        # unusable. I suppose this was the reason why the models were typically
        # loaded in the function generator.
        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
        )

        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            batchSize=batch_size,
            # NOTE: this is more of an ID than a checkpoint path
            checkpointPathGroundingDINO=checkpoint_path_groundingdino,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino
        self.groundingdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            checkpoint_path_groundingdino
        ).to(self.device)
        self.groundingdino_processor = AutoProcessor.from_pretrained(
            checkpoint_path_groundingdino
        )
        self.BOX_THRESHOLD = 0.15
        self.TEXT_THRESHOLD = 0.1
        self.CLASSES = [
            "all leaves",
            "single flower",
            "single fruit",
            "sand",
            "gravel",
            "wood",
            "tape",
            "single plant",
            "tree",
            "rock",
            "vegetation",
        ]
        self.positive_classes = [
            "leaves",
            "flower",
            "fruit",
            "plant",
            "vegetation",
        ]

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

    def detect(self, image) -> dict:
        # predict with groundingdino
        inputs = self.groundingdino_processor(
            images=image,
            text=self.CLASSES,
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

    def non_max_suppression(
        self,
        detections,
        nms_threshold: float = 0.25,
        box_threshold: float = 0.15,
    ):
        # convert tensors to lists
        boxes = detections["boxes"].tolist()
        scores = detections["scores"].tolist()
        labels = detections["text_labels"]

        # convert boxes to [x, y, width, height] format
        boxes_coco = [
            [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            for x_min, y_min, x_max, y_max in boxes
        ]

        # apply NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_coco,
            scores=scores,
            score_threshold=box_threshold,
            nms_threshold=nms_threshold,
        )

        # flatten indices and filter detections
        if len(indices) > 0:
            kept_indices = indices.flatten()
        else:
            kept_indices = []

        # filter detections
        filtered_boxes = [boxes[i] for i in kept_indices]
        filtered_scores = [scores[i] for i in kept_indices]
        filtered_labels = [labels[i] for i in kept_indices]

        # prepare filtered detections dictionary
        filtered_detections = {
            "boxes": filtered_boxes,
            "scores": filtered_scores,
            "text_labels": filtered_labels,
        }

        return filtered_detections

    def get_positive_detections(self, filtered_detections):
        from collections import defaultdict

        # dictionary of lists to store positive detections
        positive_detections = defaultdict(list)
        for label in range(len(filtered_detections["text_labels"])):
            for positive_class in self.positive_classes:
                if positive_class in filtered_detections["text_labels"][label]:
                    positive_detections["text_labels"].append(
                        filtered_detections["text_labels"][label]
                    )
                    positive_detections["boxes"].append(
                        filtered_detections["boxes"][label]
                    )
                    positive_detections["scores"].append(
                        filtered_detections["scores"][label]
                    )

        positive_detections["text_labels"] = positive_detections["text_labels"]
        positive_detections["boxes"] = positive_detections["boxes"]
        positive_detections["scores"] = positive_detections["scores"]

        return positive_detections

    def extract_and_serialize_bounding_boxes(
        self,
        image: Image.Image,
        positive_detections: dict,
    ):
        serialized_boxes = []
        box_coordinates = []

        for box in positive_detections["boxes"]:
            # Ensure box is a list of integers
            x_min, y_min, x_max, y_max = map(int, box)
            box_coordinates.append([x_min, y_min, x_max, y_max])
            cropped = image.crop((x_min, y_min, x_max, y_max))
            serialized_box = serialize_image(cropped)
            serialized_boxes.append(serialized_box)

        return {
            "extracted_bbox": serialized_boxes,
            "boxes": box_coordinates,
            "scores": positive_detections["scores"],
            "text_labels": positive_detections["text_labels"],
        }

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        # check on the nvidia stats when generating the predict function
        # self._nvidia_smi()

        def predict(image_binary: np.ndarray) -> np.ndarray:
            print("=== Predicting ===")
            print(f"image_binary type: {type(image_binary)}")
            image = deserialize_image(image_binary)
            # model_inputs = torch.stack(images).to(self.device)
            detections = self.detect(image)
            filtered_detections = self.non_max_suppression(detections)
            positive_detections = self.get_positive_detections(filtered_detections)
            final_detections = self.extract_and_serialize_bounding_boxes(
                image, positive_detections
            )

            return {
                "extracted_bbox": final_detections["extracted_bbox"],
                "boxes": final_detections["boxes"],
                "scores": final_detections["scores"],
                "text_labels": final_detections["text_labels"],
            }

        return predict

    def _transform(self, df: DataFrame):
        schema = StructType(
            [
                StructField("extracted_bbox", ArrayType(BinaryType())),
                StructField("boxes", ArrayType(ArrayType(IntegerType()))),
                StructField("scores", ArrayType(FloatType())),
                StructField("text_labels", ArrayType(StringType())),
            ]
        )
        predict_udf = F.udf(
            self._make_predict_fn(),
            returnType=schema,
        )
        return df.withColumn(
            self.getOutputCol(), predict_udf(F.col(self.getInputCol()))
        )
