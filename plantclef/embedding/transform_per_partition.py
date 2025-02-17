import io
import pandas as pd
import timm
import torch
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame

from plantclef.model_setup import setup_fine_tuned_model

# ------------------------------------------------------------------------------
# Global configuration and broadcast variable
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model configuration used on both driver and workers.
MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
NUM_CLASSES = 7806  # Total number of plant species
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable to hold the broadcasted model state (using a new name to avoid conflicts).
MODEL_STATE_BROADCAST = None


def load_and_broadcast_model(spark):
    """
    Load the model on the driver and broadcast its CPU state_dict to all executors.
    """
    global MODEL_STATE_BROADCAST
    if MODEL_STATE_BROADCAST is None:
        model_path = setup_fine_tuned_model()
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=NUM_CLASSES,
            checkpoint_path=model_path,
        )
        model.to(DEVICE)
        model.eval()
        # Move model to CPU before broadcasting to avoid GPU state issues.
        state = model.cpu().state_dict()
        MODEL_STATE_BROADCAST = spark.sparkContext.broadcast(state)
        logger.info("Model state_dict broadcasted to executors.")
    return MODEL_STATE_BROADCAST


def get_image_transform():
    """
    Return the transformation pipeline for images.
    We use a temporary model to resolve the correct configuration.
    """
    tmp_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    data_config = timm.data.resolve_model_data_config(tmp_model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return transform


# ------------------------------------------------------------------------------
# PyTorch Dataset for Image Loading
# ------------------------------------------------------------------------------


class ImageDataset(Dataset):
    """
    A PyTorch Dataset that converts a Pandas Series of binary image data
    into transformed tensors.
    """

    def __init__(self, image_series: pd.Series, transform):
        self.image_series = image_series.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_series)

    def __getitem__(self, idx):
        img_bytes = self.image_series.iloc[idx]
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Error processing image at index {idx}: {e}")
            # Create a dummy image if reading fails.
            img = Image.new("RGB", (224, 224))
        return self.transform(img)


# ------------------------------------------------------------------------------
# Pandas UDF for Batched Inference
# ------------------------------------------------------------------------------


@pandas_udf(ArrayType(FloatType()))
def predict_batch_udf(image_series: pd.Series) -> pd.Series:
    """
    Given a Pandas Series of binary image data, this UDF:
      1. Re-creates the model on the worker using the broadcasted state.
      2. Creates the image transformation pipeline.
      3. Uses a PyTorch DataLoader (with our ImageDataset) for efficient batching.
      4. Returns a Pandas Series of embeddings (one per image).
    """
    # Re-create the model on the worker and load the broadcasted state.
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(MODEL_STATE_BROADCAST.value)
    model.to(DEVICE)
    model.eval()

    transform = get_image_transform()

    # Create our PyTorch dataset and DataLoader.
    dataset = ImageDataset(image_series, transform)
    # Using num_workers=0 inside the UDF to avoid multi-threading issues.
    dataloader = DataLoader(dataset, batch_size=128, num_workers=0)

    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            features = model.forward_features(batch)
            # Extract the [CLS] token (assumed to be the first token).
            cls_tokens = features[:, 0, :].cpu().numpy()
            embeddings.extend(cls_tokens)

    return pd.Series([emb.tolist() for emb in embeddings])


# ------------------------------------------------------------------------------
# Parameter Mixins for Model Settings
# ------------------------------------------------------------------------------


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
            default=MODEL_NAME,
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
            default=128,
            doc="The batch size to use for embedding extraction",
        )

    def getBatchSize(self) -> int:
        return self.getOrDefault(self.batchSize)


# ------------------------------------------------------------------------------
# WrappedFineTunedDINOv2 Transformer
# ------------------------------------------------------------------------------


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
    A Spark ML Transformer that extracts embeddings from images using a
    fine-tuned DINOv2 model. The model is loaded on the driver and its state is
    broadcast to workers. Inference is performed via a Pandas UDF that uses a
    PyTorch Dataset for efficient batched processing.
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "cls_embedding",
        model_path: str = setup_fine_tuned_model(),
        model_name: str = MODEL_NAME,
        batch_size: int = 128,
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            modelPath=model_path,
            modelName=model_name,
            batchSize=batch_size,
        )
        self.num_classes = NUM_CLASSES
        self.device = DEVICE  # Set the device for this transformer.

    def _transform(self, df: DataFrame) -> DataFrame:
        # Ensure the model is loaded on the driver and its state is broadcast.
        load_and_broadcast_model(df.sql_ctx.sparkSession)
        # Apply the Pandas UDF to the designated input column.
        return df.withColumn(
            self.getOutputCol(), predict_batch_udf(df[self.getInputCol()])
        )
