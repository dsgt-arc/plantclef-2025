import io
import pytest
from PIL import Image
from pyspark.sql import Row

from plantclef.retrieval.embed.transform import EmbedderFineTunedDINOv2
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
    "model_name,expected_dim",
    [
        ("vit_base_patch14_reg4_dinov2.lvd142m", 768),  # Adjust output dim if needed
    ],
)
def test_wrapped_finetuned_dinov2(spark_df, model_name, expected_dim):
    model = EmbedderFineTunedDINOv2(
        input_col="img",
        output_col="transformed",
        model_path=setup_fine_tuned_model(),
        model_name=model_name,
        batch_size=2,
        use_grid=True,
        grid_size=3,
    )
    transformed = model.transform(spark_df).cache()
    transformed.printSchema()
    transformed.show()

    assert transformed.count() == 2 * 3 * 3
    assert transformed.columns == ["tile", "transformed"]
    
    tile_values = set([row.tile for row in transformed.select("tile").collect()])
    assert tile_values == set(range(3 * 3))

    row = transformed.select("transformed").first()
    assert isinstance(row.transformed, list)
    assert len(row.transformed) == expected_dim
    assert all(isinstance(x, float) for x in row.transformed)
