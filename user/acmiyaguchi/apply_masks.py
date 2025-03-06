from plantclef.spark import get_spark
from pathlib import Path
from pyspark.sql import functions as F
import numpy as np
from plantclef.serde import (
    deserialize_mask,
    serialize_mask,
    deserialize_image,
    serialize_image,
)
from PIL import Image


def main(test_path: str, mask_path: str, output_path: str, use_mean_fill: bool = False):
    # we rely on the fact that we already started a spark session
    # but otherwise configure this through the environment variables

    # the spark context needs to be available to define the UDFs
    spark = get_spark()

    @F.udf("binary")
    def merge_masks(masks: list[bytearray]) -> bytearray:
        masks = [deserialize_mask(m) for m in masks]
        merged = np.bitwise_or.reduce(masks)
        return serialize_mask(merged)

    @F.udf("binary")
    def apply_mask(
        image: bytearray, mask: bytearray, fill_mean: bool = False
    ) -> bytearray:
        image = deserialize_image(image)
        mask = deserialize_mask(mask)
        masked = image * mask[:, :, None]
        if fill_mean:
            masked_pixels = masked[mask == 1]
            if masked_pixels.size > 0:
                mean_color = masked_pixels.mean(axis=0)
                masked[mask == 0] = mean_color.astype(np.uint8)
        return serialize_image(Image.fromarray(masked))

    test_images = spark.read.parquet(test_path)
    masks = spark.read.parquet(mask_path)
    masks = masks.unpivot(
        "image_name", [c for c in masks.columns if "mask" in c], "mask_type", "mask"
    )

    masked_images = (
        masks
        # aggregate masks for positive classes with decent signal
        .where(F.col("mask_type").isin(["plant_mask", "flower_mask", "leaf_mask"]))
        .groupBy("image_name")
        .agg(F.collect_list("mask").alias("masks"))
        .select("image_name", merge_masks(F.col("masks")).alias("mask"))
        # now apply those
        .join(test_images, on="image_name")
        .select(
            "image_name",
            apply_mask(F.col("data"), F.col("mask"), F.lit(use_mean_fill)).alias(
                "data"
            ),
        )
    )
    masked_images.repartition(96).write.mode("overwrite").parquet(output_path)


if __name__ == "__main__":
    spark = get_spark()
    root = Path("~/shared/plantclef/data").expanduser().as_posix()
    mask_path = f"{root}/masking/test_2024_v2/data"
    test_path = f"{root}/parquet/test_2024"
    main(
        test_path, mask_path, f"{root}/parquet/test_2024_with_masks", use_mean_fill=True
    )
    main(
        test_path,
        mask_path,
        f"{root}/parquet/test_2024_with_masks_mean_fill",
        use_mean_fill=False,
    )
