import io
import os
import psutil
import pytest
import torch
from PIL import Image
from pyspark.sql import Row

from plantclef.embedding.transform import WrappedFineTunedDINOv2
from plantclef.model_setup import setup_fine_tuned_model


@pytest.fixture
def spark_df(spark):
    # generate a small dummy image(RGB, 32X32) for testing
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


def test_model_memory_usage():
    """
    Measure how much memory one copy of the model uses:
    - CPU memory usage is estimated by checking the process RSS before and after instantiation.
    - GPU memory usage is estimated using torch.cuda.memory_allocated().
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    # instantiate the model (CPU-side)
    _ = WrappedFineTunedDINOv2(
        input_col="dummy",
        output_col="dummy_out",
        model_path=setup_fine_tuned_model(),
        model_name="vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size=32,
    )
    mem_after = process.memory_info().rss
    cpu_usage = mem_after - mem_before
    print(
        "Estimated CPU memory usage for model instance: {:.2f} MB".format(
            cpu_usage / (1024 * 1024)
        )
    )

    # measure GPU memory usage
    # clear any cached memory first
    torch.cuda.empty_cache()
    base_gpu = torch.cuda.memory_allocated()

    # instantiate another model instance so that its weights get loaded to the GPU
    _ = WrappedFineTunedDINOv2(
        input_col="dummy",
        output_col="dummy_out",
        model_path=setup_fine_tuned_model(),
        model_name="vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size=32,
    )
    # the model __init__ calls model.to(self.device) so it moves to GPU
    gpu_usage = torch.cuda.memory_allocated() - base_gpu
    print(
        "Estimated GPU memory usage for model instance: {:.2f} MB".format(
            gpu_usage / (1024 * 1024)
        )
    )

    # Basic assertions: both should be greater than zero
    assert cpu_usage > 0, "CPU memory usage should be greater than zero."
    assert gpu_usage > 0, "GPU memory usage should be greater than zero."


def test_image_memory_usage(dummy_image, spark_df):
    """
    Measure the memory footprint of one image when it is processed and sent to the GPU.
    This test:
    - Uses the model's transformation pipeline to convert the dummy image.
    - Computes the size of the resulting tensor in bytes.
    - Also estimates the GPU memory increase when transferring that tensor.
    """
    # Instantiate the model to access its transform and device.
    model = WrappedFineTunedDINOv2(
        input_col="dummy",
        output_col="dummy_out",
        model_path=setup_fine_tuned_model(),
        model_name="vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size=32,
    )

    # Apply the transformation to the dummy image.
    transformed = model.transforms(spark_df).cache()

    # Calculate the tensor size manually.
    tensor_size = transformed.element_size() * transformed.numel()
    print(
        "Transformed image tensor size (calculated): {:.2f} MB".format(
            tensor_size / (1024 * 1024)
        )
    )

    # Alternatively, measure GPU memory before and after transferring the image.
    torch.cuda.empty_cache()
    base_gpu = torch.cuda.memory_allocated()
    dummy_tensor = model.transforms(spark_df).cache()
    gpu_usage = torch.cuda.memory_allocated() - base_gpu
    print(
        "Estimated additional GPU memory usage for image tensor: {:.2f} MB".format(
            gpu_usage / (1024 * 1024)
        )
    )

    print("Transformed tensor shape:", dummy_tensor.shape)

    # A common configuration for ViT-like models is to resize images to (3, 224, 224).
    # However, your transform might be different; adjust assertions as needed.
    assert (
        dummy_tensor.shape[0] == 3
    ), "Expected 3 channels in transformed image tensor."
