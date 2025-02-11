import io
import pytest
import torch
import platform
import subprocess
from PIL import Image
from pyspark.sql import Row

from plantclef.embedding.transform import WrappedFineTunedDINOv2
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

    return spark.createDataFrame(
        [
            Row(img=img_bytes),
            Row(img=img_bytes),
        ]
    )


@pytest.mark.parametrize(
    "model_name,expected_dim",
    [
        ("vit_base_patch14_reg4_dinov2.lvd142m", 768),  # Adjust output dim if needed
    ],
)
def test_wrapped_finetuned_dinov2_gpu(spark_df, model_name, expected_dim):
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
    # 3: Model Initialization
    # ----------------------------
    model = WrappedFineTunedDINOv2(
        input_col="img",
        output_col="transformed",
        model_path=setup_fine_tuned_model(),
        model_name=model_name,
        batch_size=2,
    )

    print("=== AFTER MODEL INITIALIZATION ===")
    print("GPU Memory Summary (after model init):")
    print(torch.cuda.memory_summary())
    print_nvidia_smi()

    # ----------------------------
    # 4: Run Transformation and Debug
    # ----------------------------
    transformed = model.transform(spark_df).cache()

    print("=== AFTER TRANSFORMATION CALL ===")
    print("GPU Memory Summary (after transform call):")
    print(torch.cuda.memory_summary())
    print_nvidia_smi()

    # Ensure Spark execution is working correctly
    assert transformed.count() == 2
    assert transformed.columns == ["img", "transformed"]

    row = transformed.select("transformed").first()
    assert isinstance(row.transformed, list)
    assert len(row.transformed) == expected_dim
    assert all(isinstance(x, float) for x in row.transformed)

    # ----------------------------
    # 5: Check Model GPU Placement
    # ----------------------------
    for param in model.model.parameters():
        assert param.device.type == "cuda", "Model parameters are not on GPU."

    print("GPU utilization confirmed.")
