import luigi
import typer
import numpy as np
from PIL import Image
from typing_extensions import Annotated
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType

from plantclef.model_setup import setup_fine_tuned_model
from plantclef.serde import deserialize_image, deserialize_mask, serialize_image
from .transform import EmbedderFineTunedDINOv2

# from .overlay import ProcessMaskOverlay
from plantclef.spark import spark_resource


class ProcessEmbeddings(luigi.Task):
    """Task to process embeddings."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    cpu_count = luigi.IntParameter(default=4)
    batch_size = luigi.IntParameter(default=32)
    num_partitions = luigi.OptionalIntParameter(default=20)
    # we break the dataset into a number of samples that are processed in parallel
    sample_col = luigi.Parameter(default="image_name")
    sample_id = luigi.IntParameter(default=None)
    num_sample_ids = luigi.IntParameter(default=20)
    # controls the number of partitions written to disk, must be at least the number
    # of tasks that we have in parallel to best take advantage of disk
    model_path = luigi.Parameter(default=setup_fine_tuned_model(scratch_model=True))
    model_name = luigi.Parameter(default="vit_base_patch14_reg4_dinov2.lvd142m")
    use_grid = luigi.BoolParameter(default=True)
    grid_size = luigi.IntParameter(default=3)
    sql_statement = luigi.Parameter(
        default="SELECT image_name, tile, cls_embedding FROM __THIS__"
    )

    def output(self):
        # write a partitioned dataset to disk
        return luigi.LocalTarget(
            f"{self.output_path}/sample_id={self.sample_id}/_SUCCESS"
        )

    def pipeline(self):
        model = Pipeline(
            stages=[
                EmbedderFineTunedDINOv2(
                    input_col="data",
                    output_col="cls_embedding",
                    model_path=self.model_path,
                    model_name=self.model_name,
                    use_grid=self.use_grid,
                    grid_size=self.grid_size,
                ),
                SQLTransformer(statement=self.sql_statement),
            ]
        )
        return model

    @property
    def feature_columns(self) -> list:
        return ["cls_embedding"]

    def transform(self, model, df, features) -> DataFrame:
        transformed = model.transform(df)

        for c in features:
            # check if the feature is a vector and convert it to an array
            if "array" in transformed.schema[c].simpleString():
                continue
            transformed = transformed.withColumn(c, vector_to_array(F.col(c)))
        return transformed

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

            # create the pipeline model
            pipeline_model = self.pipeline().fit(df)

            # transform the dataframe and write to disk
            transformed = self.transform(pipeline_model, df, self.feature_columns)

            transformed.printSchema()
            transformed.explain()
            (
                transformed.repartition(self.num_partitions)
                .cache()
                .write.mode("overwrite")
                .parquet(f"{self.output_path}/sample_id={self.sample_id}")
            )


class ProcessEmbeddingsWithMask(ProcessEmbeddings):
    """Task to process embeddings."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    test_data_path = luigi.Parameter()
    mask_cols = luigi.ListParameter(default=["leaf_mask", "flower_mask", "plant_mask"])
    sql_statement = luigi.Parameter(
        default="SELECT image_name, tile, mask_type, cls_embedding FROM __THIS__"
    )

    @staticmethod
    def apply_overlay(image_bytes: bytes, mask_bytes: bytes) -> bytes:
        """Overlay  the mask onto the image."""

        image = deserialize_image(image_bytes)  # returns Image.Image
        print("image size:", image.size)
        print("image type:", type(image))
        image_array = np.array(image)
        print("image_array shape:", image_array.shape)
        print("image type:", type(image_array))
        mask_array = deserialize_mask(mask_bytes)  # returns np.ndarray
        print("mask_array shape:", mask_array.shape)
        print("mask type:", type(mask_array))
        # convert to 3 channels -> (H, W, 3)
        # mask_array = np.repeat(np.expand_dims(mask_array, axis=-1), 3, axis=-1)
        mask_array = np.expand_dims(mask_array, axis=-1)
        print("mask_array shape:", mask_array.shape)
        print("mask type:", type(mask_array))
        # apply overlay
        overlay_img = image_array.copy()
        overlay_img[mask_array == 0] = 0
        print("overlay_img shape:", overlay_img.shape)
        print("overlay_img type:", type(overlay_img))
        # convert back to bytes
        overlay_pil = Image.fromarray(overlay_img)
        print("overlay_pil shape:", overlay_pil.size)
        print("overlay_pil type:", type(overlay_pil))
        overlay_bytes = serialize_image(overlay_pil)
        print("overlay_bytes type:", type(overlay_bytes))

        return overlay_bytes

    def run(self):
        kwargs = {"cores": self.cpu_count}
        with spark_resource(**kwargs) as spark:
            mask_df = spark.read.parquet(self.input_path)
            mask_df = (
                mask_df.unpivot(
                    "image_name",
                    [c for c in mask_df.columns if "mask" in c],
                    "mask_type",
                    "mask",
                )
                .withColumn(
                    "sample_id",
                    F.crc32(F.col(self.sample_col).cast("string"))
                    % self.num_sample_ids,
                )
                .where(F.col("sample_id") == self.sample_id)
                .drop("sample_id")
            )
            test_df = spark.read.parquet(self.test_data_path)
            df = test_df.join(mask_df, on="image_name", how="inner")

            apply_overlay_udf = F.udf(
                ProcessEmbeddingsWithMask.apply_overlay, returnType=BinaryType()
            )
            df = df.withColumn(
                "data", apply_overlay_udf(F.col("data"), F.col("mask"))
            ).drop("mask")

            # create the pipeline model
            pipeline_model = self.pipeline().fit(df)

            # transform the dataframe and write to disk
            transformed = self.transform(pipeline_model, df, self.feature_columns)

            transformed.printSchema()
            transformed.explain()
            (
                transformed.repartition(self.num_partitions)
                .cache()
                .write.mode("overwrite")
                .parquet(f"{self.output_path}/sample_id={self.sample_id}")
            )


def embed(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    test_data_path: Annotated[str, typer.Argument(help="Test DataFrame directory")],
    cpu_count: Annotated[int, typer.Option(help="Number of CPUs")] = 4,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    sample_id: Annotated[int, typer.Option(help="Sample ID")] = None,
    num_sample_ids: Annotated[int, typer.Option(help="Number of sample IDs")] = 20,
    grid_size: Annotated[int, typer.Option(help="Grid size")] = 4,
    num_partitions: Annotated[int, typer.Option(help="Number of partitions")] = 10,
    scheduler_host: Annotated[str, typer.Option(help="Scheduler host")] = None,
):
    # run the workflow
    kwargs = {}
    if scheduler_host:
        kwargs["scheduler_host"] = scheduler_host
    else:
        kwargs["local_scheduler"] = True

    luigi.build(
        [
            ProcessEmbeddings(
                input_path=input_path,
                output_path=output_path,
                test_data_path=test_data_path,
                cpu_count=cpu_count,
                batch_size=batch_size,
                sample_id=sample_id,
                num_sample_ids=num_sample_ids,
                grid_size=grid_size,
                num_partitions=num_partitions,
            )
        ],
        **kwargs,
    )


def embed_with_mask(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    test_data_path: Annotated[str, typer.Argument(help="Test DataFrame directory")],
    cpu_count: Annotated[int, typer.Option(help="Number of CPUs")] = 4,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    sample_id: Annotated[int, typer.Option(help="Sample ID")] = None,
    num_sample_ids: Annotated[int, typer.Option(help="Number of sample IDs")] = 20,
    grid_size: Annotated[int, typer.Option(help="Grid size")] = 4,
    num_partitions: Annotated[int, typer.Option(help="Number of partitions")] = 10,
    scheduler_host: Annotated[str, typer.Option(help="Scheduler host")] = None,
):
    # run the workflow
    kwargs = {}
    if scheduler_host:
        kwargs["scheduler_host"] = scheduler_host
    else:
        kwargs["local_scheduler"] = True

    luigi.build(
        [
            ProcessEmbeddingsWithMask(
                input_path=input_path,
                output_path=output_path,
                test_data_path=test_data_path,
                cpu_count=cpu_count,
                batch_size=batch_size,
                sample_id=sample_id,
                num_sample_ids=num_sample_ids,
                grid_size=grid_size,
                num_partitions=num_partitions,
            )
        ],
        **kwargs,
    )
