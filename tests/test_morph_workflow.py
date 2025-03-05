import pytest
from plantclef.morph.workflow import mask_stats_workflow
from plantclef.morph import workflow
from plantclef.serde import serialize_mask
from pyspark.sql import functions as F
import numpy as np


@pytest.fixture(autouse=True)
def mock_get_spark(spark, monkeypatch):
    def mock_get_spark():
        return spark

    monkeypatch.setattr(workflow, "get_spark", mock_get_spark)


def random_mask():
    return np.random.randint(0, 2, (100, 100)).astype(np.uint8)


@pytest.fixture
def mask_parquet(spark, tmp_path):
    dataset_path = tmp_path / "input_data"
    spark.createDataFrame(
        [
            ("image1", serialize_mask(random_mask()), serialize_mask(random_mask())),
            ("image2", serialize_mask(random_mask()), serialize_mask(random_mask())),
        ],
        ["image_name", "plant_mask", "rock_mask"],
    ).write.parquet(dataset_path.as_posix(), mode="overwrite")
    return dataset_path


def test_mask_stats_workflow(spark, tmp_path, mask_parquet):
    input_path = mask_parquet
    output_path = tmp_path / "output_data"

    mask_stats_workflow(
        input_path=input_path.as_posix(),
        output_path=output_path.as_posix(),
        iterations_max=6,
        iterations_step=2,
        num_partitions=1,
    )
    assert output_path.is_dir()

    df = spark.read.parquet(output_path.as_posix())
    df.printSchema()
    assert df.count() == 2

    # check the number of columns if we explode everything.
    # there should be 3 iteratiions per row
    assert df.select(F.explode("plant_mask_stats")).count() == 6
