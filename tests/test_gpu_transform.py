import io
import pytest
import torch
from PIL import Image
from pyspark.sql import Row

from plantclef.embedding.transform import WrappedFineTunedDINOv2
from plantclef.model_setup import setup_fine_tuned_model


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

    model = WrappedFineTunedDINOv2(
        input_col="img",
        output_col="transformed",
        model_path=setup_fine_tuned_model(),
        model_name=model_name,
        batch_size=2,
    )

    transformed = model.transform(spark_df).cache()

    # Ensure Spark execution is working correctly
    assert transformed.count() == 2
    assert transformed.columns == ["img", "transformed"]

    row = transformed.select("transformed").first()
    assert isinstance(row.transformed, list)
    assert len(row.transformed) == expected_dim
    assert all(isinstance(x, float) for x in row.transformed)

    # Check if model parameters are on the GPU
    for param in model.model.parameters():
        assert param.device.type == "cuda", "Model parameters are not on GPU."

    print("GPU utilization confirmed.")
