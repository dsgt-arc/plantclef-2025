import luigi
import typer
from typing_extensions import Annotated
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from plantclef.model_setup import setup_fine_tuned_model
from plantclef.classification.transform import ClasifierFineTunedDINOv2
from plantclef.classification.submission import SubmissionTask
from plantclef.spark import spark_resource


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
    model_path = luigi.Parameter(default=setup_fine_tuned_model(scratch_model=True))
    model_name = luigi.Parameter(default="vit_base_patch14_reg4_dinov2.lvd142m")
    use_grid = luigi.BoolParameter(default=True)
    grid_size = luigi.IntParameter(default=3)
    sql_statement = luigi.Parameter(default="SELECT image_name, logits FROM __THIS__")

    def output(self):
        # write a partitioned dataset to disk
        return luigi.LocalTarget(
            f"{self.output_path}/sample_id={self.sample_id}/_SUCCESS"
        )

    def pipeline(self):
        model = Pipeline(
            stages=[
                ClasifierFineTunedDINOv2(
                    input_col="data",
                    output_col="logits",
                    model_path=self.model_path,
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    use_grid=self.use_grid,
                    grid_size=self.grid_size,
                ),
                SQLTransformer(statement=self.sql_statement),
            ]
        )
        return model

    @property
    def feature_columns(self) -> list:
        return ["logits"]

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


class Workflow(luigi.Task):
    """Workflow with one task."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    submission_path = luigi.Parameter()
    dataset_name = luigi.Parameter()
    sample_id = luigi.OptionalParameter()
    num_sample_ids = luigi.IntParameter(default=20)
    cpu_count = luigi.IntParameter(default=6)
    batch_size = luigi.IntParameter(default=32)
    # set use_grid=False to perform inference on the entire image
    use_grid = luigi.BoolParameter(default=True)
    grid_size = luigi.IntParameter(default=3)  # 3x3 grid
    top_k_proba = luigi.IntParameter(default=5)  # top 5 species
    num_partitions = luigi.IntParameter(default=10)

    def requires(self):
        # either we run a single task or we run all the tasks
        if self.sample_id is not None:
            sample_ids = [self.sample_id]
        else:
            sample_ids = list(range(self.num_sample_ids))

        if self.use_grid:
            file_name = f"grid={self.grid_size}x{self.grid_size}"
            output_path = f"{self.output_path}/{file_name}"
        tasks = []
        for sample_id in sample_ids:
            task = ProcessClassifier(
                input_path=self.input_path,
                output_path=output_path,
                cpu_count=self.cpu_count,
                batch_size=self.batch_size,
                sample_id=sample_id,
                num_sample_ids=self.num_sample_ids,
                use_grid=self.use_grid,
                grid_size=self.grid_size,
                num_partitions=self.num_partitions,
            )
            tasks.append(task)

        # run ProcessInference tasks before the Submission task
        for task in tasks:
            yield task

        # run Submission task
        yield SubmissionTask(
            input_path=output_path,
            output_path=self.submission_path,
            dataset_name=self.dataset_name,
            top_k=self.top_k_proba,
            use_grid=self.use_grid,
            grid_size=self.grid_size,
        )


def main(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    submission_path: Annotated[str, typer.Argument(help="Submission root directory")],
    dataset_name: Annotated[str, typer.Argument(help="Test dataset name")],
    cpu_count: Annotated[int, typer.Option(help="Number of CPUs")] = 4,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    sample_id: Annotated[int, typer.Option(help="Sample ID")] = None,
    num_sample_ids: Annotated[int, typer.Option(help="Number of sample IDs")] = 20,
    use_grid: Annotated[bool, typer.Option(help="Use grid")] = True,
    grid_size: Annotated[int, typer.Option(help="Grid size")] = 3,
    top_k_proba: Annotated[int, typer.Option(help="Top K probability")] = 5,
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
                submission_path=submission_path,
                dataset_name=dataset_name,
                cpu_count=cpu_count,
                batch_size=batch_size,
                num_sample_ids=num_sample_ids,
                sample_id=sample_id,
                use_grid=use_grid,
                grid_size=grid_size,
                top_k_proba=top_k_proba,
                num_partitions=num_partitions,
            )
        ],
        **kwargs,
    )
