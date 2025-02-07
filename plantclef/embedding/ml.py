import io

import numpy as np
import timm
import torch
from PIL import Image
from pyspark.ml import Transformer
from pyspark.ml.functions import predict_batch_udf
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, MapType, StringType


class WrappedFineTunedDINOv2(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for fine-tuned DINOv2 to add it to the pipeline
    """

    def __init__(
        self,
        model_path: str,
        input_col: str = "input",
        output_col: str = "output",
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size: int = 8,
    ):
        super().__init__()
        self._setDefault(inputCol=input_col, outputCol=output_col)
        self.model_path = model_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_classes = 7806  # total number of plant species
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=self.num_classes,
            checkpoint_path=self.model_path,
        )
        # Data transform
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False
        )
        # Move model to GPU if available
        self.model.to(self.device)

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        def predict(inputs: np.ndarray) -> np.ndarray:
            images = [Image.open(io.BytesIO(input)) for input in inputs]
            model_inputs = torch.stack(
                [self.transforms(img).to(self.device) for img in images]
            )

            with torch.no_grad():
                features = self.model.forward_features(model_inputs)
                cls_token = features[:, 0, :]

            numpy_array = cls_token.cpu().numpy()
            return numpy_array

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=ArrayType(FloatType()),
                batch_size=self.batch_size,
            )(self.getInputCol()),
        )


class PretrainedDinoV2(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    def __init__(
        self,
        model_path: str,
        input_col: str = "input",
        output_col: str = "output",
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size: int = 8,
        grid_size: int = 3,
        use_grid: bool = False,
    ):
        super().__init__()
        self._setDefault(inputCol=input_col, outputCol=output_col)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_path = model_path
        self.num_classes = 7806  # total number of plant species
        self.local_directory = "/mnt/data/models/pretrained_models"
        self.class_mapping_file = f"{self.local_directory}/class_mapping.txt"
        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=self.num_classes,
            checkpoint_path=self.model_path,
        )
        self.model.to(self.device)
        self.model.eval()
        # Data transform
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False
        )
        self.cid_to_spid = self._load_class_mapping()
        self.use_grid = use_grid
        self.grid_size = grid_size

    def _load_class_mapping(self):
        with open(self.class_mapping_file) as f:
            class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
        return class_index_to_class_name

    def _split_into_grid(self, image):
        w, h = image.size
        grid_w, grid_h = w // self.grid_size, h // self.grid_size
        images = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                left = i * grid_w
                upper = j * grid_h
                right = left + grid_w
                lower = upper + grid_h
                crop_image = image.crop((left, upper, right, lower))
                images.append(crop_image)
        return images

    def _make_predict_fn(self):
        def predict(input_data):
            img = Image.open(io.BytesIO(input_data))
            top_k_proba = 20
            limit_logits = 20
            images = [img]
            # Use grid to get logits
            if self.use_grid:
                images = self._split_into_grid(img)
                top_k_proba = 10
                limit_logits = 5
            results = []
            for img in images:
                processed_image = self.transforms(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(processed_image)
                    probabilities = torch.softmax(outputs, dim=1) * 100
                    top_probs, top_indices = torch.topk(probabilities, k=top_k_proba)
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]
                result = [
                    {self.cid_to_spid.get(index, "Unknown"): float(prob)}
                    for index, prob in zip(top_indices, top_probs)
                ]
                results.append(result)
            # Flatten the results from all grids, get top 5 probabilities
            flattened_results = [
                item for grid in results for item in grid[:limit_logits]
            ]
            # Sort by score in descending order
            sorted_results = sorted(
                flattened_results, key=lambda x: -list(x.values())[0]
            )
            return sorted_results

        return predict

    def _transform(self, df: DataFrame):
        predict_fn = self._make_predict_fn()
        predict_udf = F.udf(predict_fn, ArrayType(MapType(StringType(), FloatType())))
        return df.withColumn(
            self.getOutputCol(), predict_udf(F.col(self.getInputCol()))
        )
