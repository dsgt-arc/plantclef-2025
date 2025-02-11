import io
import pytest
import torch
import luigi
import platform
import subprocess
from PIL import Image
from pyspark.sql import Row

from plantclef.embedding.workflow import ProcessEmbeddings
from plantclef.spark import get_spark
from plantclef.model_setup import setup_fine_tuned_model


def print_nvidia_smi():
    """Run nvidia-smi and print its output."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("=== NVIDIA-SMI Output ===")
        print(result.stdout)
    except Exception as e:
        print(f"nvidia-smi call failed: {e}")


@pytest.fixture
def spark_df(spark):
    # generate a small dummy image (RGB, 32X32) for testing
    img = Image.new("RGB", (32, 32), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    df = spark.createDataFrame(
        [
            Row(image_name="123.jpg", species_id=1, data=img_bytes),
            Row(image_name="456.jpg", species_id=2, data=img_bytes),
        ]
    )
    print("Initial number of partitions:", df.rdd.getNumPartitions())
    # Coalesce to 1 partition to force serialization of GPU tasks.
    df = df.coalesce(1)
    print("Number of partitions after coalesce:", df.rdd.getNumPartitions())
    return df


@pytest.fixture
def temp_parquet(spark_df, tmp_path):
    """Write the Spark DataFrame to a temporary parquet directory."""
    path = tmp_path / "data"
    spark_df.write.parquet(path.as_posix())
    return path


def test_gpu_process_embeddings(spark, spark_df, temp_parquet, tmp_path):
    # Check if CUDA is available
    assert (
        torch.cuda.is_available()
    ), "CUDA is not available, GPU utilization cannot be verified."

    # ----------------------------
    # 1. Environment Checks
    # ----------------------------
    print("=== ENVIRONMENT CHECKS ===")
    print("Python version:", platform.python_version())
    print("Torch version:", torch.__version__)
    print("Torch CUDA version:", torch.version.cuda)
    print("Is CUDA available?", torch.cuda.is_available())

    # ----------------------------
    # 2: Pre-Initialization Debugging
    # ----------------------------
    print("=== BEFORE MODEL INITIALIZATION ===")
    print("GPU Memory Summary (before model init):")
    print(torch.cuda.memory_summary())
    print_nvidia_smi()

    print("Spark DataFrame Number of Partitions:", spark_df.rdd.getNumPartitions())

    # ----------------------------
    # 3: Build and Run Luigi Task for Embeddings
    # ----------------------------
    output = tmp_path / "output"
    task = ProcessEmbeddings(
        input_path=temp_parquet.as_posix(),
        output_path=output.as_posix(),
        model_path=setup_fine_tuned_model(),
        sample_col="species_id",
        num_partitions=1,
        sample_id=0,
        num_sample_id=1,
        cpu_count=4,
        sql_statement="SELECT image_name, species_id, cls_embedding FROM __THIS__",
    )
    luigi.build([task], local_scheduler=True)

    # ----------------------------
    # 4: Restart Spark Session
    # ----------------------------
    # restart spark since luigi kills the spark session
    spark = get_spark(app_name="pytest")

    print("=== AFTER MODEL INITIALIZATION (Post-Luigi) ===")
    print("GPU Memory Summary (after model init):")
    print(torch.cuda.memory_summary())
    print_nvidia_smi()

    # ----------------------------
    # 5: Read and Debug Transformed Data
    # ----------------------------
    transformed = spark.read.parquet(f"{output}/data")

    print("=== AFTER TRANSFORMATION CALL ===")
    print("GPU Memory Summary (after transform call):")
    print(torch.cuda.memory_summary())
    print_nvidia_smi()

    # Ensure Spark execution is working correctly
    try:
        count = transformed.count()
        print("Transformed DataFrame count:", count)
    except Exception as e:
        print("Error during transformed.count():", e)
        raise

    # ----------------------------
    # 6: Verify Transformed DataFrame
    # ----------------------------
    assert count == 2, "Expected 2 rows in transformed DataFrame."
    expected_columns = ["image_name", "species_id", "cls_embedding", "sample_id"]
    assert (
        transformed.columns == expected_columns
    ), f"Unexpected columns in transformed DataFrame. Expected {expected_columns}, got {transformed.columns}"

    row = transformed.select("cls_embedding").first()
    assert len(row.cls_embedding) == 768, "Expected embedding length 768."
    assert all(
        isinstance(x, float) for x in row.cls_embedding
    ), "Not all elements in embedding are floats."
