import luigi
import numpy as np

from PIL import Image
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType
from plantclef.spark import spark_resource
from plantclef.serde import deserialize_image, deserialize_mask, serialize_image


class ProcessMaskOverlay(luigi.Task):
    """Task to process the overlay masks."""

    input_path = luigi.Parameter()
    test_data_path = luigi.Parameter()
    sample_col = luigi.Parameter(default="image_name")
    sample_id = luigi.IntParameter(default=None)
    num_sample_ids = luigi.IntParameter(default=20)
    cpu_count = luigi.IntParameter(default=4)

    def requires(self):
        pass

    def transform(self, image_bytes: bytes, mask_bytes: bytes) -> bytes:
        """Overlay  the mask onto the image."""
        image_array = deserialize_image(image_bytes)
        mask_array = deserialize_mask(mask_bytes)
        mask_array = np.repeat(np.expand_dims(mask_array, axis=-1), 3, axis=-1)
        overlay_img = image_array * mask_array
        overlay_pil = Image.fromarray(overlay_img)
        overlay_bytes = serialize_image(overlay_pil)

        return overlay_bytes

    def run(self):
        kwargs = {
            "cores": self.cpu_count,
        }
        with spark_resource(**kwargs) as spark:
            # read the data and keep the sample we're currently processing
            df = (
                spark.read.parquet(self.input_path)
                .withColumn(
                    "sample_id",
                    F.crc32(F.col(self.sample_col).cast("string"))
                    % self.num_sample_ids,
                )
                .where(F.col("sample_id") == self.sample_id)
                .drop("sample_id")
            )
            test_df = spark.read.parquet(self.test_data_path)
            joined_df = df.join(test_df, on="image_name", how="inner")
            # overlay the masks onto the images
            overlay_udf = F.udf(self.transform, BinaryType())
            # apply the overlay transformation
            transformed = (
                joined_df.withColumn(
                    "leaf_overlay", overlay_udf(F.col("image"), F.col("leaf_mask"))
                )
                .withColumn(
                    "flower_overlay", overlay_udf(F.col("image"), F.col("flower_mask"))
                )
                .withColumn(
                    "plant_overlay", overlay_udf(F.col("image"), F.col("plant_mask"))
                )
            )
            transformed.printSchema()
            transformed.explain()

        return transformed
