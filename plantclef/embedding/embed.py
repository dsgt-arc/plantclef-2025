from argparse import ArgumentParser

import io
import timm
import logging
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType

from plantclef.spark import spark_resource
from plantclef.config import get_data_dir
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
    dataset = ImageDataset(image_series, transform=transform)
    # Using num_workers=0 inside the UDF to avoid multi-threading issues.
    dataloader = DataLoader(dataset, batch_size=128, num_workers=6)

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
# Parse command-line arguments
# ------------------------------------------------------------------------------


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Luigi pipeline")
    parser.add_argument(
        "--input-dataset-name",
        type=str,
        default="subset_top10_train",
        help="Input dataset name in parquet format.",
    )
    parser.add_argument(
        "--cpu-count",
        type=int,
        default=6,
        help="The number of CPUs to use for the Spark job",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=10,
        help="Number of partitions for the output DataFrame",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size to use for embedding extraction",
    )
    parser.add_argument(
        "--num-sample-id",
        type=int,
        default=20,
        help="The number of sample IDs to use for embedding extraction",
    )
    return parser.parse_args()


# ------------------------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------------------------


def main():
    # parse args
    args = parse_args()

    # Get the base path for the PACE parquet files
    dataset_base_path = get_data_dir()

    # Input and output paths for training workflow
    # "~/p-dsgt_clef2025-0/shared/plantclef/data"
    input_path = f"{dataset_base_path}/parquet/{args.input_dataset_name}"
    output_path = f"{dataset_base_path}/embeddings/{args.input_dataset_name}_embeddings"

    # Initialize Spark with settings for using the big-disk-dev VM
    kwargs = {
        "cores": args.cpu_count,
        "spark.sql.shuffle.partitions": args.num_partitions,
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.execution.arrow.maxRecordsPerBatch": "64",
    }
    print(f"Running task with kwargs: {kwargs}")
    with spark_resource(**kwargs) as spark:
        # Load the input data
        # number of partitions should be a small multiple of total number of nodes
        df = spark.read.parquet(input_path).repartition(args.num_partitions)

        # Load the model and broadcast it to the executors
        load_and_broadcast_model(spark)

        embeddings_df = df.withColumn("cls_embeddings", predict_batch_udf(df["data"]))
        embeddings_df.write.mode("overwrite").parquet(output_path)


if __name__ == "__main__":
    main()
