import pytest
import luigi

from plantclef.spark import get_spark
from plantclef.retrieval.embed.workflow import ProcessEmbeddings
from plantclef.retrieval.embed.transform import EmbedderFineTunedDINOv2
from plantclef.model_setup import setup_fine_tuned_model


@pytest.fixture
def transformed_df(spark_df):
    # transform image and return masks
    model = EmbedderFineTunedDINOv2(
        input_col="data",
        output_col="cls_embedding",
        model_path=setup_fine_tuned_model(),
        model_name="vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size=2,
        grid_size=4,
    )
    transformed = model.transform(spark_df).cache()
    transformed.printSchema()
    return transformed


@pytest.fixture
def temp_parquet(transformed_df, tmp_path):
    path = tmp_path / "data"
    transformed_df.write.parquet(path.as_posix())
    return path


@pytest.mark.parametrize(
    "grid_size,expected_dim",
    [
        (4, 768),
    ],
)
def test_process_embeddings(spark, grid_size, expected_dim, temp_parquet, tmp_path):
    output = tmp_path / "output"
    task = ProcessEmbeddings(
        input_path=temp_parquet.as_posix(),
        output_path=output.as_posix(),
        sample_col="image_name",
        num_partitions=1,
        sample_id=0,
        num_sample_ids=1,
        cpu_count=4,
        grid_size=grid_size,
        sql_statement="SELECT image_name, tile, cls_embedding FROM __THIS__",
    )
    luigi.build([task], local_scheduler=True)

    # restart spark since luigi kills the spark session
    spark = get_spark(app_name="pytest")
    transformed = spark.read.parquet(f"{output}/data")

    assert transformed.count() == 2
    assert transformed.columns == [
        "image_name",
        "tile",
        "cls_embedding",
        "sample_id",
    ]
    row = transformed.select("cls_embedding").first()
    assert len(row.cls_embedding) == expected_dim
    assert all(isinstance(x, float) for x in row.cls_embedding)
    transformed.show()
