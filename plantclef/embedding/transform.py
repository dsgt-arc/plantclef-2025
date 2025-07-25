import timm
import torch
import numpy as np
from plantclef.serde import deserialize_image
from plantclef.model_setup import setup_fine_tuned_model
from pyspark.sql import DataFrame
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.functions import predict_batch_udf
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField


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
            default=32,
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
    Wrapper for fine-tuned DINOv2 to add it to the pipeline.
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        model_path: str = setup_fine_tuned_model(),
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size: int = 32,
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
        self.model = (
            timm.create_model(
                self.getModelName(),
                pretrained=False,
                num_classes=self.num_classes,
                checkpoint_path=self.getModelPath(),
            )
            .to(self.device)
            .eval()
        )
        # data transform
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False
        )

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

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        # check on the nvidia stats when generating the predict function
        # self._nvidia_smi()

        def predict(image_batch: np.ndarray) -> np.ndarray:
            images = [self.transforms(deserialize_image(img)) for img in image_batch]
            model_inputs = torch.stack(images).to(self.device)

            with torch.no_grad():
                outputs = self.model(model_inputs).cpu().numpy()
                features = self.model.forward_features(model_inputs).cpu().numpy()
                cls_token = features[:, 0, :]

            return [
                {
                    "cls_token": cls_token[i],
                    "logits": outputs[i],
                }
                for i in range(len(image_batch))
            ]

        return predict

    def _transform(self, df: DataFrame):
        schema = StructType(
            [
                StructField("cls_token", ArrayType(FloatType())),
                StructField("logits", ArrayType(FloatType())),
            ]
        )

        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=schema,
                batch_size=self.getBatchSize(),
            )(self.getInputCol()),
        )
