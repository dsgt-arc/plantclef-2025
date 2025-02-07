import io

import numpy as np
import timm
import torch
from PIL import Image
from plantclef.model_setup import setup_fine_tuned_model
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.functions import predict_batch_udf
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, FloatType


class HasModelPath(Param):
    """
    Mixin for param model_path: str
    """

    modelPath = Param(
        Params._dummy(),
        "modelPath",
        "The path to the fine-tuned DINOv2 model",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__(
            default=setup_fine_tuned_model(),
            doc="The path to the fine-tuned DINOv2 model",
        )

    def getModelPath(self) -> str:
        return self.getOrDefault(self.modelPath)


class HasModelName(Param):
    """
    Mixin for param model_name: str
    """

    modelName = Param(
        Params._dummy(),
        "modelName",
        "The name of the DINOv2 model to use",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__(
            default="vit_base_patch14_reg4_dinov2.lvd142m",
            doc="The name of the DINOv2 model to use",
        )

    def getModelName(self) -> str:
        return self.getOrDefault(self.modelName)


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
            default=8,
            doc="The batch size to use for embedding extraction",
        )

    def getBatchSize(self) -> int:
        return self.getOrDefault(self.batchSize)


class WrappedFineTunedDINOv2(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasModelPath,
    HasModelName,
    HasBatchSize,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for fine-tuned DINOv2 to add it to the pipeline
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        model_path: str = setup_fine_tuned_model(),
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size: int = 8,
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            modelPath=model_path,
            modelName=model_name,
            batchSize=batch_size,
        )
        self.num_classes = 7806  # total number of plant species
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(
            self.getModelName(),
            pretrained=False,
            num_classes=self.num_classes,
            checkpoint_path=self.getModelPath(),
        )
        # Data transform
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False
        )
        # Move model to GPU if available
        self.model.to(self.device)

    def _nvidia_smi(self):
        from subprocess import run

        try:
            run(["nvidia-smi"], check=True)
        except Exception:
            pass

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        def predict(inputs: np.ndarray) -> np.ndarray:
            # check GPU status before processing
            self._nvidia_smi()

            images = [Image.open(io.BytesIO(input)) for input in inputs]
            model_inputs = torch.stack(
                [self.transforms(img).to(self.device) for img in images]
            )

            with torch.no_grad():
                features = self.model.forward_features(model_inputs)
                cls_token = features[:, 0, :]

            # check GPU status after processing
            self._nvidia_smi()

            # return the computed embeddings as numpy array
            numpy_array = cls_token.cpu().numpy()
            return numpy_array

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=ArrayType(FloatType()),
                batch_size=self.getBatchSize(),
            )(self.getInputCol()),
        )
