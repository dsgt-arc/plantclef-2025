from argparse import ArgumentParser

import os
import luigi
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from plantclef.model_setup import setup_fine_tuned_model
from plantclef.embedding.transform import WrappedFineTunedDINOv2
from plantclef.spark import spark_resource
from plantclef.config import get_data_dir


class ProcessDINOv2Pipeline(luigi.Task):
    output_path = luigi.Parameter()
    sql_statement = luigi.Parameter()
    model_path = luigi.Parameter(default=setup_fine_tuned_model())
    batch_size = luigi.IntParameter(default=1)

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/metadata/_SUCCESS")

    def pipeline(self) -> Pipeline:
        dinov2_model = WrappedFineTunedDINOv2(
            input_col="data",
            output_col="cls_embedding",
            model_path=self.model_path,
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
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    model_path = luigi.Parameter(default=setup_fine_tuned_model())
    sample_col = luigi.Parameter(default="species_id")
    # controls the number of partitions written to disk, must be at least the number
    # of tasks that we have in parallel to best take advantage of disk
    num_partitions = luigi.OptionalIntParameter(default=500)
    sample_id = luigi.OptionalIntParameter(default=None)
    num_sample_id = luigi.OptionalIntParameter(default=50)
    batch_size = luigi.IntParameter(default=32)
    cpu_count = luigi.IntParameter(default=4)
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
                model_path=self.model_path,
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
            "spark.sql.shuffle.partitions": self.num_partitions,
        }
        print(f"Running task with kwargs: {kwargs}")
        with spark_resource(**kwargs) as spark:
            # read the data and keep the sample we're currently processing
            df = (
                spark.read.parquet(self.input_path)
                .withColumn(
                    "sample_id",
                    F.crc32(F.col(self.sample_col).cast("string")) % self.num_sample_id,
                )
                .where(F.col("sample_id") == self.sample_id)
                .drop("sample_id")
            )

            print("Initial number of partitions:", df.rdd.getNumPartitions())
            # Coalesce to 1 partition to force serialization of GPU tasks
            # df = df.coalesce(1)
            print(
                "Number of partitions after coalescing for GPU inference:",
                df.rdd.getNumPartitions(),
            )

            model = PipelineModel.load(f"{self.output_path}/model")
            model.write().overwrite().save(f"{self.output_path}/model")
            # transform the dataframe and write to disk
            transformed = self.transform(model, df, self.feature_columns)

            transformed.repartition(self.num_partitions).write.mode(
                "overwrite"
            ).parquet(f"{self.output_path}/data/sample_id={self.sample_id}")


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    model_path = luigi.Parameter(default=setup_fine_tuned_model())
    process_test_data = luigi.OptionalBoolParameter(default=False)
    use_grid = luigi.OptionalBoolParameter(default=False)
    use_only_classifier = luigi.OptionalBoolParameter(default=False)
    cpu_count = luigi.IntParameter(default=4)
    batch_size = luigi.IntParameter(default=1)

    def run(self):
        # training workflow parameters
        sample_col = "species_id"
        sql_statement = "SELECT image_name, species_id, cls_embedding FROM __THIS__"
        input_path = self.input_path
        output_path = self.output_path

        # print parameters for sanity check
        print(f"\ninput_path: {input_path}")
        print(f"output_path: {output_path}")
        print(f"sample_col: {sample_col}\n")
        yield [
            ProcessEmbeddings(
                input_path=input_path,
                output_path=output_path,
                model_path=self.model_path,
                sample_col=sample_col,
                num_partitions=500,
                sample_id=i,
                num_sample_id=20,
                batch_size=self.batch_size,
                cpu_count=self.cpu_count,
                sql_statement=sql_statement,
            )
            for i in range(20)
        ]


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Luigi pipeline")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="The batch size to use for embedding extraction",
    )
    parser.add_argument(
        "--process-test-data",
        type=bool,
        default=False,
        help="If True, set workflow to process the test data and extract embeddings",
    )
    parser.add_argument(
        "--use-grid",
        type=bool,
        default=False,
        help="If True, create a grid when using the pretrained ViT model to make predictions",
    )
    parser.add_argument(
        "--use-only-classifier",
        type=bool,
        default=False,
        help="""
        If True, use the pretrained ViT with frozen backbone, one classification head has been finetuned
        Otherwise, use the only-classifier-then-all pretrained model
        """,
    )
    parser.add_argument(
        "--scheduler-host",
        type=None,
        default=None,
        help="""Scheduler host for the Luigi workflow""",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Get the base path for the PACE parquet files
    dataset_base_path = get_data_dir()
    # Input and output paths for training workflow
    # "~/p-dsgt_clef2025-0/shared/plantclef/data"
    # input_path = f"{dataset_base_path}/parquet_files/train"
    # output_path = f"{dataset_base_path}/embeddings/train_embeddings"
    input_path = f"{dataset_base_path}/parquet_files/subset_top20_train"
    output_path = f"{dataset_base_path}/embeddings/subset_top20_embeddings"
    model_path = setup_fine_tuned_model(use_only_classifier=False)
    cpu_count = os.cpu_count()

    # parse args
    args = parse_args()

    # update input and output params for embedding the test data
    if args.process_test_data:
        input_path = f"{dataset_base_path}/parquet_files/test"
        output_path = f"{dataset_base_path}/data/embeddings/test_embeddings"

    # run the workflow
    kwargs = {}
    if args.scheduler_host:
        kwargs["scheduler_host"] = args.scheduler_host
    else:
        kwargs["local_scheduler"] = True

    luigi.build(
        [
            Workflow(
                input_path=input_path,
                output_path=output_path,
                model_path=model_path,
                process_test_data=args.process_test_data,
                use_grid=args.use_grid,
                use_only_classifier=args.use_only_classifier,
                cpu_count=cpu_count,
                batch_size=args.batch_size,
            )
        ],
        **kwargs,
    )
