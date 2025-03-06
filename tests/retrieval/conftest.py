import pytest
from pathlib import Path
from plantclef.spark import get_spark


@pytest.fixture
def test_mask_path() -> Path:
    return Path(
        "/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data/masking/test_2024_v2"
    )


@pytest.fixture
def test_data_path() -> Path:
    return Path(
        "/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data/parquet/test_2024"
    )


@pytest.fixture
def spark_df(test_mask_path):
    spark = get_spark(cores=4, memory="20g", executor_memory="20g", app_name="pytest")
    # dataframe with masked images
    image_df = spark.read.parquet(test_mask_path.as_posix())
    image_df.printSchema()
    image_df = image_df.limit(1)
    return image_df


@pytest.fixture
def temp_parquet(spark_df, tmp_path):
    path = tmp_path / "data"
    spark_df.write.parquet(path.as_posix())
    return path


@pytest.fixture
def joined_df(temp_parquet, test_data_path):
    spark = get_spark(cores=1, memory="10g", app_name="pytest")
    # join the test data with the mask data
    mask_df = spark.read.parquet(temp_parquet.as_posix())
    test_df = spark.read.parquet(test_data_path.as_posix())
    df = mask_df.join(test_df, on="image_name", how="inner")
    return df


@pytest.fixture
def temp_joined_parquet(joined_df, tmp_path):
    path = tmp_path / "data"
    joined_df.write.mode("overwrite").parquet(path.as_posix())
    return path
