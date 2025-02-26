import pytest
import luigi
from pathlib import Path

from plantclef.masking.workflow import ProcessMasking
from plantclef.spark import get_spark
from plantclef.masking.transform import WrappedMasking
from plantclef.model_setup import (
    setup_segment_anything_checkpoint_path,
    setup_groundingdino_checkpoint_path,
    setup_groundingdino_config_path,
)


@pytest.fixture
def spark_df():
    spark = get_spark(cores=2, memory="16g", app_name="pytest")
    # image path
    image_path = Path(__file__).parent / "images/CBN-can-A1-20230705.jpg"
    # dataframe with a single image column
    image_df = (
        spark.read.format("binaryFile")
        .load(image_path.as_posix())
        .withColumnRenamed("content", "data")
        .withColumnRenamed("path", "image_name")
    )
    image_df.printSchema()
    image_df = image_df.select("image_name", "data")
    return image_df


@pytest.fixture
def transformed_df(spark_df):
    # transform image and return masks
    model = WrappedMasking(
        input_col="data",
        output_col="masks",
        checkpoint_path_sam=setup_segment_anything_checkpoint_path(),
        checkpoint_path_groundingdino=setup_groundingdino_checkpoint_path(),
        config_path_groundingdino=setup_groundingdino_config_path(),
        encoder_version="vit_h",
        batch_size=1,
    )
    transformed = model.transform(spark_df).cache()
    return transformed


@pytest.fixture
def temp_parquet(transformed_df, tmp_path):
    path = tmp_path / "data"
    transformed_df.write.parquet(path.as_posix())
    return path


def test_process_masking(spark, temp_parquet, tmp_path):
    output = tmp_path / "output"
    # process image masks from transformed DF
    task = ProcessMasking(
        input_path=temp_parquet.as_posix(),
        output_path=output.as_posix(),
        cpu_count=4,
        batch_size=1,
        num_partitions=1,
        sample_id=0,
        num_sample_ids=1,
    )
    luigi.build([task], local_scheduler=True)

    # restart spark since luigi kills the spark session
    spark = get_spark(app_name="pytest")
    transformed = spark.read.parquet(f"{output}/data")

    assert transformed.count() == 1
    assert transformed.columns == [
        "image_name",
        "combined_mask",
        "leaf_mask",
        "flower_mask",
        "plant_mask",
        "sample_id",
    ]
    row = transformed.select("leaf_mask").first()
    assert isinstance(row.leaf_mask, (bytes, bytearray))
    transformed.show()
    transformed.printSchema()
