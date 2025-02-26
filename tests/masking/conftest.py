import pytest
from pathlib import Path
from plantclef.spark import get_spark
from PIL import Image


@pytest.fixture
def test_image_path() -> Path:
    return Path(__file__).parent.parent / "images/CBN-can-A1-20230705.jpg"


@pytest.fixture
def test_image(test_image_path) -> Image:
    return Image.open(test_image_path)


@pytest.fixture
def spark_df(test_image_path):
    spark = get_spark(cores=1, memory="16g", app_name="pytest")
    # dataframe with a single image column
    image_df = (
        spark.read.format("binaryFile")
        .load(test_image_path.as_posix())
        .withColumnRenamed("content", "img")
    )
    image_df.printSchema()
    image_df = image_df.select("img")
    return image_df
