import pytest
from pathlib import Path
from plantclef.spark import get_spark

from plantclef.masking.transform import WrappedMasking
from plantclef.model_setup import (
    setup_segment_anything_checkpoint_path,
    setup_groundingdino_checkpoint_path,
    setup_groundingdino_config_path,
)


@pytest.fixture
def spark_df():
    spark = get_spark(cores=6, memory="16g", app_name="pytest")
    # image path
    image_path = Path(__file__).parent / "images/CBN-can-A1-20230705.jpg"
    # dataframe with a single image column
    image_df = (
        spark.read.format("binaryFile")
        .load(image_path.as_posix())
        .withColumnRenamed("content", "img")
    )
    image_df.printSchema()
    return image_df


@pytest.mark.parametrize(
    "encoder_version",
    [
        "vit_h",  # Adjust output dim if needed
    ],
)
def test_wrapped_mask(spark_df, encoder_version):
    model = WrappedMasking(
        input_col="img",
        output_col="masks",
        checkpoint_path_sam=setup_segment_anything_checkpoint_path(),
        checkpoint_path_groundingdino=setup_groundingdino_checkpoint_path(),
        config_path_groundingdino=setup_groundingdino_config_path(),
        encoder_version=encoder_version,
        batch_size=2,
    )
    transformed = model.transform(spark_df).cache()
    transformed.printSchema()
    transformed.show()

    transformed.count()
    assert transformed.count() == 1
    assert transformed.columns == "masks"

    row = transformed.select("plant_mask").first()
    assert isinstance(row.transformed, list)
    # assert len(row.transformed) == expected_dim
    assert all(isinstance(x, float) for x in row.transformed)
