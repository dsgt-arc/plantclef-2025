import luigi
import typer
from typing_extensions import Annotated
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from plantclef.model_setup import setup_fine_tuned_model
from plantclef.embedding.transform import WrappedFineTunedDINOv2
from plantclef.spark import spark_resource


class ProcessDINOv2Pipeline(luigi.Task):
    """Task to process embeddings using a DINOv2 model."""

    output_path = luigi.Parameter()
    sql_statement = luigi.Parameter()
    model_path = luigi.Parameter(default=setup_fine_tuned_model(scratch_model=True))
    model_name = luigi.Parameter(default="vit_base_patch14_reg4_dinov2.lvd142m")
    batch_size = luigi.IntParameter(default=32)

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/metadata/_SUCCESS")

    def pipeline(self) -> Pipeline:
        dinov2_model = WrappedFineTunedDINOv2(
            input_col="data",
            output_col="cls_embedding",
            model_path=self.model_path,
            model_name=self.model_name,
            batch_size=self.batch_size,
        )
        return Pipeline(
            stages=[dinov2_model, SQLTransformer(statement=self.sql_statement)]
        )

    def run(self):
        with spark_resource() as spark:
            model = self.pipeline().fit(spark.createDataFrame([[""]], ["image_name"]))
            model.write().overwrite().save(f"{self.output_path}")


class ProcessEmbeddings(luigi.Task):
    """Task to process embeddings."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    sample_col = luigi.Parameter(default="image_name")
    sample_id = luigi.OptionalIntParameter(default=None)
    # controls the number of partitions written to disk, must be at least the number
    # of tasks that we have in parallel to best take advantage of disk
    num_sample_ids = luigi.OptionalIntParameter(default=20)
    num_partitions = luigi.OptionalIntParameter(default=20)
    cpu_count = luigi.IntParameter(default=4)
    batch_size = luigi.IntParameter(default=32)
    sql_statement = luigi.Parameter(
        default="SELECT image_name, species_id, cls_embedding FROM __THIS__"
    )

    def output(self):
        # write a partitioned dataset to disk
        return luigi.LocalTarget(
            f"{self.output_path}/data/sample_id={self.sample_id}/_SUCCESS"
        )

    def requires(self):
        return [
            ProcessDINOv2Pipeline(
                output_path=f"{self.output_path}/model",
                sql_statement=self.sql_statement,
                batch_size=self.batch_size,
            )
        ]

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
        print(f"Running task with kwargs: {kwargs}")
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

            print("Initial number of partitions:", df.rdd.getNumPartitions())
            # Coalesce to 1 partition to force serialization of GPU tasks
            # df = df.coalesce(8)
            print(
                "Number of partitions after coalescing for GPU inference:",
                df.rdd.getNumPartitions(),
            )

            model = PipelineModel.load(f"{self.output_path}/model")
            model.write().overwrite().save(f"{self.output_path}/model")
            # transform the dataframe and write to disk
            transformed = self.transform(model, df, self.feature_columns)

            transformed.printSchema()
            transformed.explain()
            (
                transformed.repartition(self.num_partitions)
                .write.mode("overwrite")
                .parquet(f"{self.output_path}/data/sample_id={self.sample_id}")
            )


class Workflow(luigi.WrapperTask):
    """Workflow with two tasks."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    sample_id = luigi.OptionalParameter()
    num_sample_ids = luigi.IntParameter(default=20)
    cpu_count = luigi.IntParameter(default=6)
    batch_size = luigi.IntParameter(default=32)
    num_partitions = luigi.IntParameter(default=20)

    def requires(self):
        # either we run a single task or we run all the tasks
        if self.sample_id is not None:
            sample_ids = [self.sample_id]
        else:
            sample_ids = list(range(self.num_tasks))

        tasks = []
        for sample_id in sample_ids:
            print(f"Creating task for sample_id: {sample_id}")
            task = ProcessEmbeddings(
                input_path=self.input_path,
                output_path=self.output_path,
                sample_id=sample_id,
                num_sample_ids=self.num_sample_ids,
                num_partitions=self.num_partitions,
                cpu_count=self.cpu_count,
                batch_size=self.batch_size,
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
    scheduler_host: Annotated[str, typer.Option(help="Scheduler host")] = None,
):
    print("Starting workflow execution...")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"CPU Count: {cpu_count}, Batch Size: {batch_size}, Sample ID: {sample_id}")

    # run the workflow
    kwargs = {}
    if scheduler_host:
        kwargs["scheduler_host"] = scheduler_host
    else:
        kwargs["local_scheduler"] = True

    print("Calling luigi.build() with parameters...")

    luigi.build(
        [
            Workflow(
                input_path=input_path,
                output_path=output_path,
                cpu_count=cpu_count,
                batch_size=batch_size,
                sample_id=sample_id,
                num_sample_ids=num_sample_ids,
            )
        ],
        **kwargs,
    )

    print("Workflow execution completed")
