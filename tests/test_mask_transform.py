import io
import pytest
from PIL import Image
from pyspark.sql import Row

from plantclef.masking.transform import WrappedMasking
from plantclef.model_setup import (
    setup_segment_anything_checkpoint_path,
    setup_groundingdino_checkpoint_path,
    setup_groundingdino_config_path,
)


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


@pytest.mark.parametrize(
    "encoder_version,expected_dim",
    [
        ("vit_h", 768),  # Adjust output dim if needed
    ],
)
def test_wrapped_finetuned_dinov2(spark_df, encoder_version, expected_dim):
    model = WrappedMasking(
        input_col="img",
        output_col=["leaf_mask", "flower_mask", "plant_mask", "final_mask"],
        checkpoint_path_sam=setup_segment_anything_checkpoint_path(),
        checkpoint_path_groundingdino=setup_groundingdino_checkpoint_path(),
        config_path_groundingding=setup_groundingdino_config_path(),
        encoder_version="vit_h",
        batch_size=2,
    )
    transformed = model.transform(spark_df).cache()
    transformed.printSchema()
    transformed.show()

    assert transformed.count() == 2
    assert transformed.columns == [
        "img",
        "leaf_mask",
        "flower_mask",
        "plant_mask",
        "final_mask",
    ]

    row = transformed.select("plant_mask").first()
    assert isinstance(row.transformed, list)
    assert len(row.transformed) == expected_dim
    assert all(isinstance(x, float) for x in row.transformed)
