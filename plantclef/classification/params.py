from pyspark.ml.param import Param, Params, TypeConverters
from plantclef.model_setup import setup_fine_tuned_model


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
