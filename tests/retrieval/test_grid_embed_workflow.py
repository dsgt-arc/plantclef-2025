import pytest
import luigi

from plantclef.spark import get_spark
from plantclef.retrieval.embed.workflow import ProcessEmbeddings


@pytest.fixture
def temp_parquet(spark_df, tmp_path):
    path = tmp_path / "data"
    spark_df.write.parquet(path.as_posix())
    return path


@pytest.mark.parametrize(
    "grid_size,mask_cols,expected_dim",
    [
        (4, ["leaf_mask", "flower_mask", "plant_mask"], 768),
    ],
)
def test_process_embeddings(
    spark,
    grid_size,
    mask_cols,
    expected_dim,
    test_data_path,
    temp_parquet,
    tmp_path,
):
    output = tmp_path / "output"
    task = ProcessEmbeddings(
        input_path=temp_parquet.as_posix(),
        output_path=output.as_posix(),
        test_data_path=test_data_path.as_posix(),
        cpu_count=4,
        sample_id=0,
        num_sample_ids=1,
        grid_size=grid_size,
        mask_cols=mask_cols,
        num_partitions=1,
    )
    luigi.build([task], local_scheduler=True)

    # restart spark since luigi kills the spark session
    spark = get_spark(app_name="pytest")
    transformed = spark.read.parquet(f"{output}/data")

    assert transformed.count() == 2
    assert transformed.columns == [
        "image_name",
        "tile",
        "leaf_embed",
        "flower_embed",
        "plant_embed",
        "sample_id",
    ]
    row = transformed.select("cls_embedding").first()
    assert len(row.cls_embedding) == expected_dim
    assert all(isinstance(x, float) for x in row.cls_embedding)
    transformed.show()
