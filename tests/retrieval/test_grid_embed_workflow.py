import numpy as np
from PIL import Image
import pytest
import luigi

from plantclef.spark import get_spark
from plantclef.serde import deserialize_image
from plantclef.retrieval.embed.workflow import ProcessEmbeddingsWithMask
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType


def test_apply_overlay(spark, test_data_path, temp_parquet):
    # join the test data with the mask data
    mask_df = spark.read.parquet(temp_parquet.as_posix())
    test_df = spark.read.parquet(test_data_path.as_posix())
    df = mask_df.join(test_df, on="image_name", how="inner")
    # apply the overlay
    overlay_udf = F.udf(ProcessEmbeddingsWithMask.apply_overlay, BinaryType())
    for mask_col in ["leaf_mask", "flower_mask", "plant_mask"]:
        overlay_col = mask_col.replace("mask", "overlay")
        df = df.withColumn(overlay_col, overlay_udf(F.col("data"), F.col(mask_col)))

    print("Joined Dataframe:")
    df.printSchema()
    leaf_overlay = df.select("leaf_overlay").first().leaf_overlay
    print(f"leaf_overlay type: {type(leaf_overlay)}")

    # ensure that the output mask is a NumPy array
    assert isinstance(leaf_overlay, bytearray)

    # decode the bytes back into a NumPy array
    mask = deserialize_image(leaf_overlay)
    assert isinstance(mask, Image.Image)

    # ensure mask has the expected dimensions (same as input image)
    img_data = df.select("data").first().data
    img = deserialize_image(img_data)
    expected_shape = img.size[::-1]
    mask = np.array(mask)
    print(f"mask shape: {mask.shape}, img shape: {expected_shape}")


@pytest.mark.parametrize(
    "grid_size,mask_cols,expected_dim",
    [
        # (4, ["leaf_mask", "flower_mask", "plant_mask"], 768),
        (4, [], 768),
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
    task = ProcessEmbeddingsWithMask(
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
    # output_cols = ["image_name", "tile", "leaf_embed", "flower_embed", "plant_embed"]
    # transformed = transformed.select(
    #     "image_name", "tile", "cls_embedding", "sample_id"
    # ).cache()
    transformed.printSchema()

    assert transformed.count() == 2
    assert transformed.columns == [
        "image_name",
        "tile",
        "mask_type",
        "cls_embedding",
    ]
    row = transformed.select("cls_embedding").first()
    assert len(row.cls_embedding) == expected_dim
    assert all(isinstance(x, float) for x in row.cls_embedding)
    transformed.show()
