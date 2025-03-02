import pytest
from pathlib import Path
from plantclef.spark import get_spark


@pytest.fixture
def test_mask_path() -> Path:
    return Path(
        "/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data/masking/test_2024_v2"
    )


@pytest.fixture
def spark_df(test_mask_path):
    spark = get_spark(cores=1, memory="16g", app_name="pytest")
    # dataframe with masked images
    image_df = spark.read.parquet(test_mask_path.as_posix())
    image_df.printSchema()
    image_df = image_df.limit(1)
    return image_df
