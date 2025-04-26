import luigi
import typer
import pandas as pd
from PIL import Image
from typing_extensions import Annotated
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    IntegerType,
    StructType,
    StructField,
)

from plantclef.embedding.transform import WrappedFineTunedDINOv2
from plantclef.detection.transform import WrappedGroundingDINO
from plantclef.spark import spark_resource
from plantclef.model_setup import setup_fine_tuned_model
from plantclef.serde import deserialize_image, serialize_image


def make_image_tiles_udf(grid_size: int):
    def split_into_grid(image: Image.Image):
        w, h = image.size
        grid_w, grid_h = w // grid_size, h // grid_size
        tiles = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = i * grid_w
                upper = j * grid_h
                right = left + grid_w
                lower = upper + grid_h
                tile = image.crop((left, upper, right, lower))
                tiles.append(tile)
        return tiles

    @F.pandas_udf(
        ArrayType(
            StructType(
                [
                    StructField("tile_index", IntegerType()),
                    StructField("tile", BinaryType()),
                ]
            )
        )
    )
    def image_tiles_udf(data_series: pd.Series) -> pd.Series:
        all_tiles = []
        for b in data_series:
            img = deserialize_image(b)
            tiles = split_into_grid(img)
            tile_structs = [
                {"tile_index": i, "tile": serialize_image(tile)}
                for i, tile in enumerate(tiles)
            ]
            all_tiles.append(tile_structs)
        return pd.Series(all_tiles)

    return image_tiles_udf


class ProcessTransform(luigi.Task):
    """Task to process embeddings."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    task = luigi.Parameter()  # task name: embed, detect, or classify
    cpu_count = luigi.IntParameter(default=4)
    batch_size = luigi.IntParameter(default=32)
    num_partitions = luigi.OptionalIntParameter(default=20)
    # we break the dataset into a number of samples that are processed in parallel
    sample_col = luigi.Parameter(default="image_name")
    sample_id = luigi.OptionalIntParameter(default=None)
    num_sample_ids = luigi.OptionalIntParameter(default=20)
    # controls the number of partitions written to disk, must be at least the number
    # of tasks that we have in parallel to best take advantage of disk
    use_test_data = luigi.BoolParameter(default=False)
    use_grid = luigi.BoolParameter(default=True)
    grid_size = luigi.IntParameter(default=4)
    cols = luigi.Parameter(default="image_name, species_id")
    model_path = luigi.Parameter(default=setup_fine_tuned_model(scratch_model=True))
    model_name = luigi.Parameter(default="vit_base_patch14_reg4_dinov2.lvd142m")

    @property
    def columns_to_use(self) -> str:
        # use only image_name for test set
        columns_to_use = "image_name" if self.use_test_data else self.cols
        return columns_to_use

    @property
    def sql_statement(self) -> str:
        cols = [self.columns_to_use, "output"]
        if self.use_grid:
            cols.append("tile_index")
        return f"SELECT {', '.join(cols)} FROM __THIS__"

    def output(self):
        # write a partitioned dataset to disk
        return luigi.LocalTarget(
            f"{self.output_path}/sample_id={self.sample_id}/_SUCCESS"
        )

    def get_task(self):
        if self.task == "embed":
            return WrappedFineTunedDINOv2(
                input_col="tile" if self.use_grid else "data",
                output_col="output",
                model_path=self.model_path,
                model_name=self.model_name,
                batch_size=self.batch_size,
            )
        elif self.task == "detect":
            return WrappedGroundingDINO(
                input_col="tile" if self.use_grid else "data",
                output_col="output",
                batch_size=self.batch_size,
            )

    def pipeline(self):
        model = Pipeline(
            stages=[
                self.get_task(),
                SQLTransformer(statement=self.sql_statement),
            ]
        )
        return model

    @property
    def output_columns(self) -> list:
        output_col = ["output"]
        if self.use_grid:
            output_col.append("tile_index")
        return output_col

    def transform(self, model, df) -> DataFrame:
        if self.use_grid:
            tile_udf = make_image_tiles_udf(self.grid_size)
            df_with_tiles = df.withColumn("tiles", tile_udf("data"))
            df_tiles = df_with_tiles.select(
                "image_name", F.explode("tiles").alias("tile_struct")
            )
            df = df_tiles.select(
                "image_name",
                F.col("tile_struct.tile_index").alias("tile_index"),
                F.col("tile_struct.tile").alias("tile"),
            )
        transformed = model.transform(df)
        # unpack the output column
        transformed = transformed.select(self.columns_to_use, *self.output_columns)
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
            transformed = self.transform(pipeline_model, df)

            transformed.printSchema()
            transformed.explain()
            (
                transformed.repartition(self.num_partitions)
                .write.mode("overwrite")
                .parquet(f"{self.output_path}/sample_id={self.sample_id}")
            )


class Workflow(luigi.WrapperTask):
    """Workflow with one task."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    task = luigi.Parameter()  # task name: embed, detect, or classify
    sample_id = luigi.OptionalParameter()
    num_sample_ids = luigi.IntParameter(default=20)
    cpu_count = luigi.IntParameter(default=6)
    batch_size = luigi.IntParameter(default=32)
    num_partitions = luigi.IntParameter(default=20)
    use_test_data = luigi.BoolParameter(default=False)
    use_grid = luigi.BoolParameter(default=True)
    grid_size = luigi.IntParameter(default=4)

    def requires(self):
        # either we run a single task or we run all the tasks
        if self.sample_id is not None:
            sample_ids = [self.sample_id]
        else:
            sample_ids = list(range(self.num_tasks))

        tasks = []
        for sample_id in sample_ids:
            task = ProcessTransform(
                input_path=self.input_path,
                output_path=self.output_path,
                task=self.task,
                cpu_count=self.cpu_count,
                batch_size=self.batch_size,
                num_partitions=self.num_partitions,
                sample_id=sample_id,
                num_sample_ids=self.num_sample_ids,
                use_test_data=self.use_test_data,
                use_grid=self.use_grid,
                grid_size=self.grid_size,
            )
            tasks.append(task)
        yield tasks


def main(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    task: Annotated[
        str, typer.Argument(help="Task name: embed, detect, or classify")
    ] = None,
    cpu_count: Annotated[int, typer.Option(help="Number of CPUs")] = 8,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    sample_id: Annotated[int, typer.Option(help="Sample ID")] = None,
    num_sample_ids: Annotated[int, typer.Option(help="Number of sample IDs")] = 20,
    use_test_data: Annotated[bool, typer.Option(help="Use test data")] = False,
    use_grid: Annotated[bool, typer.Option(help="Use grid")] = False,
    grid_size: Annotated[int, typer.Option(help="Grid size")] = 4,
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
                task=task,
                cpu_count=cpu_count,
                batch_size=batch_size,
                sample_id=sample_id,
                num_sample_ids=num_sample_ids,
                use_test_data=use_test_data,
                use_grid=use_grid,
                grid_size=grid_size,
            )
        ],
        **kwargs,
    )
