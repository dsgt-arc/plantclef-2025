import typer
from plantclef.spark import get_spark
import numpy as np
from functools import partial
from plantclef.serde import deserialize_mask, serialize_mask
from pyspark.sql import functions as F


# grid each mask, and the calculate how many of those grids are not null.
def split_into_tiles(mask: np.ndarray, grid_size: int) -> np.ndarray:
    w, h = mask.shape
    grid_w, grid_h = w // grid_size, h // grid_size
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            left = i * grid_w
            upper = j * grid_h
            right = left + grid_w
            lower = upper + grid_h
            tiles.append(mask[left:right, upper:lower])
    return np.array(tiles)


def tile_mask_percentage(mask: bytearray, grid_size: int) -> list[int]:
    mask = deserialize_mask(mask)
    tiles = split_into_tiles(mask, grid_size)
    means = np.mean(tiles.reshape(tiles.shape[0], -1), axis=1)
    return means.tolist()


def merge_masks(masks: list[bytearray]) -> bytearray:
    masks = [deserialize_mask(m) for m in masks]
    merged = np.bitwise_or.reduce(masks)
    return serialize_mask(merged)


def filter_by_mask(
    embedding_path: str = typer.Argument(
        ..., help="Path to the input directory containing images."
    ),
    mask_path: str = typer.Argument(
        ..., help="Path to the input directory containing masks."
    ),
    output_path: str = typer.Argument(
        ..., help="Path to the output directory containing filtered images."
    ),
    grid_size: int = typer.Option(4, help="Size of the grid to split the image into."),
):
    """Filter tile embeddings by mask.

    This appends additional information to the dataset that is used to filter out
    tiles that are not of interest.
    """
    spark = get_spark()
    grid_emb = spark.read.parquet(embedding_path).where(
        F.col("grid") == f"{grid_size}x{grid_size}"
    )

    # note that we have a mask per column, where-as it's significantly
    # easier to work with a mask per row. In order to deal with this
    # after the factly, we have to unpivot the mask dataframe
    masks = spark.read.parquet(mask_path)
    masks = masks.unpivot(
        "image_name", [c for c in masks.columns if "mask" in c], "mask_type", "mask"
    ).cache()

    # some handy UDFs
    merge_masks_udf = F.udf(merge_masks, returnType="binary")
    tile_mask_percentage_udf = F.udf(
        partial(tile_mask_percentage, grid_size=grid_size), returnType="array<float>"
    )

    tile_mask_info = (
        masks
        # first generate a combined mask
        .where(F.col("mask_type").isin(["plant_mask", "flower_mask", "leaf_mask"]))
        .groupBy("image_name")
        .agg(F.collect_list("mask").alias("masks"))
        .select("image_name", merge_masks_udf(F.col("masks")).alias("mask"))
        # then calculate the tile mask percentage for the particular grid
        .select(
            "image_name",
            F.posexplode(tile_mask_percentage_udf("mask")).alias("tile", "pct_covered"),
        )
    )

    # join the datasets together and write it out to a new location in a partitioned way
    output_path = f"{output_path}/grid={grid_size}x{grid_size}"
    (
        grid_emb.join(tile_mask_info, ["image_name", "tile"], "inner")
        .repartition(16)
        .write.mode("overwrite")
        .parquet(output_path)
    )

    # show a bit of info about the dataset at the end
    df = spark.read.parquet(output_path)
    df.printSchema()
    df.show(5, truncate=100)
