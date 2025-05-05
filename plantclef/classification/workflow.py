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

from plantclef.model_setup import setup_fine_tuned_model
from plantclef.classification.transform import ClasifierFineTunedDINOv2
from plantclef.spark import spark_resource
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


class ProcessClassifier(luigi.Task):
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
    use_detections = luigi.BoolParameter(default=False)
    model_path = luigi.Parameter(default=setup_fine_tuned_model(scratch_model=True))
    model_name = luigi.Parameter(default="vit_base_patch14_reg4_dinov2.lvd142m")
    use_grid = luigi.BoolParameter(default=False)
    grid_size = luigi.IntParameter(default=4)
    prior_path = luigi.Parameter(default=None)  # use Bayesian prior inference
    sql_statement = luigi.Parameter(default="SELECT image_name, logits FROM __THIS__")

    @property
    def _get_output_dir(self):
        output_dir = f"{self.output_path}/sample_id={self.sample_id}"
        return output_dir

    def output(self):
        # write a partitioned dataset to disk
        return luigi.LocalTarget(f"{self._get_output_dir}/_SUCCESS")

    def pipeline(self):
        model = Pipeline(
            stages=[
                ClasifierFineTunedDINOv2(
                    input_col="extracted_bbox" if self.use_detections else "data",
                    output_col="logits",
                    model_path=self.model_path,
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    use_grid=self.use_grid,
                    grid_size=self.grid_size,
                    prior_path=self.prior_path,
                ),
                SQLTransformer(statement=self.sql_statement),
            ]
        )
        return model

    @property
    def output_columns(self) -> list:
        output_col = ["probabilities"]
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
        if self.use_detections:
            df = df.select(
                "image_name",
                F.explode("output.extracted_bbox").alias("extracted_bbox"),
            )
        # transform the dataframe
        transformed = model.transform(df)

        # unpack the output column
        transformed = transformed.select("image_name", *self.output_columns)
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
                .cache()
                .write.mode("overwrite")
                .parquet(self._get_output_dir)
            )


class Workflow(luigi.Task):
    """Workflow with one task."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    submission_path = luigi.Parameter()
    dataset_name = luigi.Parameter()
    sample_id = luigi.OptionalParameter()
    num_sample_ids = luigi.IntParameter(default=20)
    use_detections = luigi.BoolParameter(default=False)
    cpu_count = luigi.IntParameter(default=6)
    batch_size = luigi.IntParameter(default=32)
    # set use_grid=False to perform inference on the entire image
    use_grid = luigi.BoolParameter(default=True)
    grid_size = luigi.IntParameter(default=3)  # 3x3 grid
    top_k_proba = luigi.IntParameter(default=5)  # top 5 species
    num_partitions = luigi.IntParameter(default=10)
    prior_path = luigi.Parameter(default=None)

    @property
    def _get_prior_folder_name(self):
        """Returns the folder name based on prior path."""
        prior_name = self.prior_path.split("test_2025_")[-1]
        if "cluster" in prior_name:
            prior_name = "cluster"
        elif "image" in prior_name:
            prior_name = "image"
        return prior_name

    def _get_base_output_path(self):
        """Returns the base output path with consistent directory structure."""
        base_path = self.output_path

        if self.prior_path:
            base_path = f"{base_path}_prior_{self._get_prior_folder_name}"

        if self.use_grid:
            base_path = f"{base_path}/grid_{self.grid_size}x{self.grid_size}"

        return base_path

    def requires(self):
        # either we run a single task or we run all the tasks
        if self.sample_id is not None:
            sample_ids = [self.sample_id]
        else:
            sample_ids = list(range(self.num_sample_ids))

        output_path = self._get_base_output_path()

        tasks = []
        for sample_id in sample_ids:
            task = ProcessClassifier(
                input_path=self.input_path,
                output_path=output_path,
                cpu_count=self.cpu_count,
                batch_size=self.batch_size,
                sample_id=sample_id,
                num_sample_ids=self.num_sample_ids,
                use_detections=self.use_detections,
                use_grid=self.use_grid,
                grid_size=self.grid_size,
                num_partitions=self.num_partitions,
                prior_path=self.prior_path,
            )
            tasks.append(task)

        # run ProcessInference tasks before the Submission task
        for task in tasks:
            yield task

        # # run Submission task
        # yield SubmissionTask(
        #     input_path=output_path,
        #     output_path=self.submission_path,
        #     dataset_name=self.dataset_name,
        #     top_k=self.top_k_proba,
        #     use_grid=self.use_grid,
        #     grid_size=self.grid_size,
        #     prior_path=self.prior_path,
        # )


def main(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    submission_path: Annotated[str, typer.Argument(help="Submission root directory")],
    dataset_name: Annotated[str, typer.Argument(help="Test dataset name")],
    cpu_count: Annotated[int, typer.Option(help="Number of CPUs")] = 4,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    sample_id: Annotated[int, typer.Option(help="Sample ID")] = None,
    num_sample_ids: Annotated[int, typer.Option(help="Number of sample IDs")] = 20,
    use_detections: Annotated[bool, typer.Option(help="Use detections")] = False,
    use_grid: Annotated[bool, typer.Option(help="Use grid")] = False,
    grid_size: Annotated[int, typer.Option(help="Grid size")] = 4,
    top_k_proba: Annotated[int, typer.Option(help="Top K probability")] = 5,
    num_partitions: Annotated[int, typer.Option(help="Number of partitions")] = 10,
    prior_path: Annotated[str, typer.Option(help="Prior dataframe path")] = None,
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
                submission_path=submission_path,
                dataset_name=dataset_name,
                cpu_count=cpu_count,
                batch_size=batch_size,
                sample_id=sample_id,
                num_sample_ids=num_sample_ids,
                use_detections=use_detections,
                use_grid=use_grid,
                grid_size=grid_size,
                top_k_proba=top_k_proba,
                num_partitions=num_partitions,
                prior_path=prior_path,
            )
        ],
        **kwargs,
    )
