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
    test_data_path = luigi.Parameter()
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
    grid_size = luigi.IntParameter(default=4)
    mask_cols = luigi.ListParameter(default=["leaf_mask", "flower_mask", "plant_mask"])

    def output(self):
        # write a partitioned dataset to disk
        return luigi.LocalTarget(
            f"{self.output_path}/sample_id={self.sample_id}/_SUCCESS"
        )

    def pipeline(self):
        model = Pipeline(
            stages=[
                EmbedderFineTunedDINOv2(
                    input_cols=self.input_columns,
                    output_cols=self.feature_columns,
                    model_path=self.model_path,
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    grid_size=self.grid_size,
                ),
                SQLTransformer(statement=self.sql_statement),
            ]
        )
        return model

    @property
    def input_columns(self) -> list:
        # input cols that will be used by the transformation
        input_cols = self.mask_cols
        if len(self.mask_cols) == 0:
            input_cols = ["data"]
        return input_cols

    @property
    def feature_columns(self) -> list:
        # feature cols that will be created by the transformation
        feature_cols = [col.replace("mask", "embed") for col in self.mask_cols]
        if len(self.mask_cols) == 0:
            feature_cols = ["cls_embedding"]
        return feature_cols

    @property
    def sql_statement(self) -> str:
        mask_cols = self.feature_columns
        mask_cols_str = ", ".join(mask_cols)
        sql_statement = f"SELECT image_name, tile, {mask_cols_str} FROM __THIS__"
        return sql_statement

    def apply_overlay(self, image_bytes: bytes, mask_bytes: bytes) -> bytes:
        """Overlay  the mask onto the image."""

        image = deserialize_image(image_bytes)  # returns Image.Image
        image_array = np.array(image)
        print("image_array shape:", image_array.shape)
        mask_array = deserialize_mask(mask_bytes)  # returns np.ndarray
        print("mask_array shape:", mask_array.shape)
        # convert to 3 channels -> (H, W, 3)
        mask_array = np.repeat(np.expand_dims(mask_array, axis=-1), 3, axis=-1)
        # apply overlay
        overlay_img = image_array * mask_array
        # convert back to bytes
        overlay_pil = Image.fromarray(overlay_img)
        overlay_bytes = serialize_image(overlay_pil)

        return overlay_bytes

    def transform(self, model, mask_df, test_df, features, mask_cols) -> DataFrame:
        """Transform the dataframe by applying the model and overlaying the masks."""

        # join the dataframes
        df = mask_df.join(test_df, on="image_name", how="inner")

        # apply overlay transformation to mask columns
        overlay_udf = F.udf(self.apply_overlay, BinaryType())
        for mask_col in self.input_columns:
            overlay_col = mask_col.replace("mask", "overlay")
            # TODO: remove this print statement, debugging only
            print(f"Input cols: {mask_col}, {overlay_col}", flush=True)

            df = df.withColumn(
                overlay_col,
                overlay_udf(F.col("data"), F.col(mask_col)),
            )
        # TODO: remove this print statement, debugging only
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

        # run model pipeline
        transformed = model.transform(df)

        for c in self.feature_columns:
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
            mask_df = (
                spark.read.parquet(self.input_path)
                .withColumn(
                    "sample_id",
                    F.crc32(F.col(self.sample_col).cast("string"))
                    % self.num_sample_ids,
                )
                .where(F.col("sample_id") == self.sample_id)
                .drop("sample_id")
            )
            # read test data
            test_df = spark.read.parquet(self.test_data_path)

            # create the pipeline model
            pipeline_model = self.pipeline().fit(mask_df)

            # transform the dataframe and write to disk
            transformed = self.transform(
                pipeline_model, mask_df, test_df, self.feature_columns, self.mask_cols
            )

            transformed.printSchema()
            transformed.explain()
            (
                transformed.repartition(self.num_partitions)
                .cache()
                .write.mode("overwrite")
                .parquet(f"{self.output_path}/sample_id={self.sample_id}")
            )


class Workflow(luigi.Task):
    """Workflow with one task."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    test_data_path = luigi.Parameter()
    sample_id = luigi.OptionalParameter()
    num_sample_ids = luigi.IntParameter(default=20)
    cpu_count = luigi.IntParameter(default=6)
    batch_size = luigi.IntParameter(default=32)
    grid_size = luigi.IntParameter(default=4)  # 4x4 grid
    mask_cols = luigi.ListParameter(default=["leaf_mask", "flower_mask", "plant_mask"])
    num_partitions = luigi.IntParameter(default=10)

    def requires(self):
        task = ProcessEmbeddings(
            input_path=self.input_path,
            output_path=self.output_path,
            test_data_path=self.test_data_path,
            cpu_count=self.cpu_count,
            batch_size=self.batch_size,
            sample_id=0,
            num_sample_ids=1,
            grid_size=self.grid_size,
            num_partitions=self.num_partitions,
        )
        yield task


def main(
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
            Workflow(
                input_path=input_path,
                output_path=output_path,
                test_data_path=test_data_path,
                cpu_count=cpu_count,
                batch_size=batch_size,
                num_sample_ids=num_sample_ids,
                sample_id=sample_id,
                grid_size=grid_size,
                num_partitions=num_partitions,
            )
        ],
        **kwargs,
    )
