import pytest
from pathlib import Path
from plantclef.spark import get_spark


@pytest.fixture
def spark_df():
    spark = get_spark(cores=2, memory="16g", app_name="pytest")
    # image path
    image_path = Path(__file__).parent.parent / "images/CBN-can-A1-20230705.jpg"
    # dataframe with a single image column
    image_df = (
        spark.read.format("binaryFile")
        .load(image_path.as_posix())
        .withColumnRenamed("content", "img")
    )
    image_df.printSchema()
    image_df = image_df.select("img")
    return image_df
