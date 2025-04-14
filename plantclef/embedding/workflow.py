import luigi
import typer
from typing_extensions import Annotated
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from plantclef.embedding.transform import WrappedFineTunedDINOv2
from plantclef.spark import spark_resource
from plantclef.model_setup import setup_fine_tuned_model


class ProcessEmbeddings(luigi.Task):
    """Task to process embeddings."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
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
    cols = luigi.Parameter(default="image_name, species_id")

    @property
    def columns_to_use(self) -> str:
        # Use only image_name for test set
        columns_to_use = "image_name" if self.use_test_data else self.cols
        return columns_to_use

    @property
    def sql_statement(self) -> str:
        return f"SELECT {self.columns_to_use}, output FROM __THIS__"

    model_path = luigi.Parameter(default=setup_fine_tuned_model(scratch_model=True))
    model_name = luigi.Parameter(default="vit_base_patch14_reg4_dinov2.lvd142m")

    def output(self):
        # write a partitioned dataset to disk
        return luigi.LocalTarget(
            f"{self.output_path}/data/sample_id={self.sample_id}/_SUCCESS"
        )

    def pipeline(self):
        model = Pipeline(
            stages=[
                WrappedFineTunedDINOv2(
                    input_col="data",
                    output_col="output",
                    model_path=self.model_path,
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                ),
                SQLTransformer(statement=self.sql_statement),
            ]
        )
        return model

    @property
    def feature_columns(self) -> list:
        return ["output"]

    def transform(self, model, df, features) -> DataFrame:
        transformed = model.transform(df)
        # unpack the output column
        transformed = transformed.select(self.columns_to_use, *self.feature_columns)
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
    batch_size = luigi.IntParameter(default=32)
    num_partitions = luigi.IntParameter(default=20)
    use_test_data = luigi.BoolParameter(default=False)

    def requires(self):
        # either we run a single task or we run all the tasks
        if self.sample_id is not None:
            sample_ids = [self.sample_id]
        else:
            sample_ids = list(range(self.num_tasks))

        tasks = []
        for sample_id in sample_ids:
            task = ProcessEmbeddings(
                input_path=self.input_path,
                output_path=self.output_path,
                cpu_count=self.cpu_count,
                batch_size=self.batch_size,
                num_partitions=self.num_partitions,
                sample_id=sample_id,
                num_sample_ids=self.num_sample_ids,
                use_test_data=self.use_test_data,
            )
            tasks.append(task)
        yield tasks


def main(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    cpu_count: Annotated[int, typer.Option(help="Number of CPUs")] = 8,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    sample_id: Annotated[int, typer.Option(help="Sample ID")] = None,
    num_sample_ids: Annotated[int, typer.Option(help="Number of sample IDs")] = 20,
    use_test_data: Annotated[bool, typer.Option(help="Use test data")] = False,
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
                batch_size=batch_size,
                sample_id=sample_id,
                num_sample_ids=num_sample_ids,
                use_test_data=use_test_data,
            )
        ],
        **kwargs,
    )
