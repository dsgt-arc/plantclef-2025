import luigi
import typer
from typing_extensions import Annotated
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from plantclef.masking.transform import WrappedMasking
from plantclef.spark import spark_resource


class ProcessMasking(luigi.Task):
    """Task to process embeddings."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    cpu_count = luigi.IntParameter(default=4)
    num_partitions = luigi.OptionalIntParameter(default=20)
    # we break the dataset into a number of samples that are processed in parallel
    sample_col = luigi.Parameter(default="image_name")
    sample_id = luigi.OptionalIntParameter(default=None)
    num_sample_ids = luigi.OptionalIntParameter(default=20)
    # controls the number of partitions written to disk, must be at least the number
    # of tasks that we have in parallel to best take advantage of disk
    sql_statement = luigi.Parameter(default="SELECT image_name, masks FROM __THIS__")
    checkpoint_path_sam = luigi.Parameter(
        default="facebook/sam-vit-huge",
    )
    checkpoint_path_groundingdino = luigi.Parameter(
        default="IDEA-Research/grounding-dino-base",
    )

    def output(self):
        # write a partitioned dataset to disk
        return luigi.LocalTarget(
            f"{self.output_path}/data/sample_id={self.sample_id}/_SUCCESS"
        )

    def pipeline(self):
        model = Pipeline(
            stages=[
                WrappedMasking(
                    input_col="data",
                    output_col="masks",
                    checkpoint_path_sam=self.checkpoint_path_sam,
                    checkpoint_path_groundingdino=self.checkpoint_path_groundingdino,
                ),
                SQLTransformer(statement=self.sql_statement),
            ]
        )
        return model

    @property
    def feature_columns(self) -> list:
        return ["masks"]

    def transform(self, model, df, features) -> DataFrame:
        transformed = model.transform(df)

        for c in features:
            # check if the feature is a vector and convert it to an array
            if "vector" in transformed.schema[c].simpleString():
                transformed = transformed.withColumn(c, vector_to_array(F.col(c)))

        transformed = (
            transformed.withColumn("leaf_mask", F.col("masks.leaf_mask"))
            .withColumn("flower_mask", F.col("masks.flower_mask"))
            .withColumn("plant_mask", F.col("masks.plant_mask"))
            .withColumn("sand_mask", F.col("masks.sand_mask"))
            .withColumn("wood_mask", F.col("masks.wood_mask"))
            .withColumn("tape_mask", F.col("masks.tape_mask"))
            .withColumn("tree_mask", F.col("masks.tree_mask"))
            .withColumn("rock_mask", F.col("masks.rock_mask"))
            .withColumn("vegetation_mask", F.col("masks.vegetation_mask"))
            .drop("masks")
        )

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
            # write dataframe to disk
            (
                transformed.repartition(self.num_partitions)
                .write.mode("overwrite")
                .parquet(f"{self.output_path}/data/sample_id={self.sample_id}")
            )


class Workflow(luigi.WrapperTask):
    """Workflow with one task."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    sample_id = luigi.OptionalParameter()
    num_sample_ids = luigi.IntParameter(default=20)
    cpu_count = luigi.IntParameter(default=6)
    num_partitions = luigi.IntParameter(default=20)

    def requires(self):
        # either we run a single task or we run all the tasks
        if self.sample_id is not None:
            sample_ids = [self.sample_id]
        else:
            sample_ids = list(range(self.num_tasks))

        tasks = []
        for sample_id in sample_ids:
            task = ProcessMasking(
                input_path=self.input_path,
                output_path=self.output_path,
                cpu_count=self.cpu_count,
                num_partitions=self.num_partitions,
                sample_id=sample_id,
                num_sample_ids=self.num_sample_ids,
            )
            tasks.append(task)
        yield tasks


def main(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    cpu_count: Annotated[int, typer.Option(help="Number of CPUs")] = 8,
    sample_id: Annotated[int, typer.Option(help="Sample ID")] = None,
    num_sample_ids: Annotated[int, typer.Option(help="Number of sample IDs")] = 20,
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
                cpu_count=cpu_count,
                num_sample_ids=num_sample_ids,
                sample_id=sample_id,
            )
        ],
        **kwargs,
    )
