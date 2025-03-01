import io
import pytest
import luigi
from PIL import Image
from pyspark.sql import Row

from plantclef.embedding.workflow import ProcessEmbeddings
from plantclef.spark import get_spark
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
            Row(image_name="123.jpg", species_id=1, data=img_bytes),
            Row(image_name="456.jpg", species_id=2, data=img_bytes),
        ]
    )


@pytest.fixture
def temp_parquet(spark_df, tmp_path):
    path = tmp_path / "data"
    spark_df.write.parquet(path.as_posix())
    return path


def test_process_embeddings(spark, temp_parquet, tmp_path):
    output = tmp_path / "output"
    task = ProcessEmbeddings(
        input_path=temp_parquet.as_posix(),
        output_path=output.as_posix(),
        model_path=setup_fine_tuned_model(),
        sample_col="species_id",
        num_partitions=1,
        sample_id=0,
        num_sample_ids=1,
        cpu_count=4,
        sql_statement="SELECT image_name, species_id, cls_embedding FROM __THIS__",
    )
    luigi.build([task], local_scheduler=True)

    # restart spark since luigi kills the spark session
    spark = get_spark(app_name="pytest")
    transformed = spark.read.parquet(f"{output}/data")

    assert transformed.count() == 2
    assert transformed.columns == [
        "image_name",
        "species_id",
        "cls_embedding",
        "sample_id",
    ]
    row = transformed.select("cls_embedding").first()
    assert len(row.cls_embedding) == 768
    assert all(isinstance(x, float) for x in row.cls_embedding)
    transformed.show()
