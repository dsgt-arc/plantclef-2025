import io
import pytest
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


@pytest.mark.parametrize(
    "model_name, expected_dim, num_classes",
    [
        ("vit_base_patch14_reg4_dinov2.lvd142m", 768, 7806),
    ],
)
def test_wrapped_finetuned_dinov2(spark_df, model_name, expected_dim, num_classes):
    model = WrappedFineTunedDINOv2(
        input_col="img",
        output_col="output",
        model_path=setup_fine_tuned_model(),
        model_name=model_name,
        batch_size=2,
    )
    transformed = model.transform(spark_df).cache()
    transformed.printSchema()
    # transformed.show()

    transformed = transformed.select("img", "output.*")

    assert transformed.count() == 2
    assert transformed.columns == ["img", "cls_token", "logits"]

    row = transformed.select("cls_token").first()
    assert isinstance(row.cls_token, list)
    assert len(row.cls_token) == expected_dim
    assert all(isinstance(x, float) for x in row.cls_token)
