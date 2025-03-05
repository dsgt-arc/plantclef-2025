import pytest
import luigi

from plantclef.spark import get_spark
from plantclef.masking.workflow import ProcessMasking
from plantclef.masking.transform import WrappedMasking


@pytest.fixture
def transformed_df(spark_df):
    # transform image and return masks
    model = WrappedMasking(input_col="data", output_col="masks", batch_size=1)
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
        "leaf_mask",
        "flower_mask",
        "plant_mask",
        "sand_mask",
        "wood_mask",
        "tape_mask",
        "tree_mask",
        "rock_mask",
        "vegetation_mask",
        "sample_id",
    ]
    row = transformed.select("leaf_mask").first()
    assert isinstance(row.leaf_mask, (bytes, bytearray))
    transformed.show()
    transformed.printSchema()
