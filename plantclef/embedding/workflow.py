from argparse import ArgumentParser

import luigi
import luigi.contrib.gcs
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from plantclef.model_setup import setup_pretrained_model
from plantclef.transforms import WrappedFineTunedDINOv2
from plantclef.spark import spark_resource
from plantclef.config import get_pace_data_dir


class ProcessBase(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    should_subset = luigi.BoolParameter(default=False)
    sample_col = luigi.Parameter(default="species_id")
    num_partitions = luigi.OptionalIntParameter(default=500)
    sample_id = luigi.OptionalIntParameter(default=None)
    num_sample_id = luigi.OptionalIntParameter(default=20)
    cpu_count = luigi.IntParameter(default=4)

    def output(self):
        if self.sample_id is None:
            # save both the model pipeline and the dataset
            return luigi.LocalTarget(f"{self.output_path}/_SUCCESS")
        else:
            # write a partitioned dataset to disk
            return luigi.LocalTarget(
                f"{self.output_path}/sample_id={self.sample_id}/_SUCCESS"
            )

    @property
    def feature_columns(self) -> list:
        raise NotImplementedError()

    def pipeline(self) -> Pipeline:
        raise NotImplementedError()

    def transform(self, model, df, features) -> DataFrame:
        transformed = model.transform(df)

        if self.sample_id is not None:
            transformed = (
                transformed.withColumn(
                    "sample_id",
                    F.crc32(F.col(self.sample_col).cast("string")) % self.num_sample_id,
                )
                .where(F.col("sample_id") == self.sample_id)
                .drop("sample_id")
            )

        for c in features:
            # check if the feature is a vector and convert it to an array
            if "array" in transformed.schema[c].simpleString():
                continue
            transformed = transformed.withColumn(c, vector_to_array(F.col(c)))
        return transformed

    def _get_subset(self, df):
        # Get subset of images to test pipeline
        subset_df = (
            df.where(F.col(self.sample_col).isin([1361703, 1355927]))
            .orderBy(F.rand(1000))
            .limit(200)
            .cache()
        )
        return subset_df

    def run(self):
        kwargs = {
            "cores": self.cpu_count(),
            "spark.sql.shuffle.partitions": self.num_partitions,
        }
        with spark_resource(**kwargs) as spark:
            df = spark.read.parquet(self.input_path)

            if self.should_subset:
                # Get subset of data to test the workflow
                df = self._get_subset(df=df)

            model = self.pipeline().fit(df)
            model.write().overwrite().save(f"{self.output_path}/model")
            transformed = self.transform(model, df, self.feature_columns)

            output_path = self.output_path
            if self.sample_id is not None:
                output_path = f"{self.output_path}/sample_id={self.sample_id}"

            transformed.repartition(self.num_partitions).write.mode(
                "overwrite"
            ).parquet(output_path)


class ProcessFineTunedDINOv2(ProcessBase):
    sql_statement = luigi.Parameter()
    model_path = luigi.Parameter()

    @property
    def feature_columns(self) -> list:
        return ["cls_embedding"]

    def pipeline(self):
        dinov2_model = WrappedFineTunedDINOv2(
            model_path=self.model_path,
            input_col="data",
            output_col="cls_embedding",
        )
        return Pipeline(
            stages=[dinov2_model, SQLTransformer(statement=self.sql_statement)]
        )


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    process_test_data = luigi.OptionalBoolParameter(default=False)
    use_grid = luigi.OptionalBoolParameter(default=False)
    use_only_classifier = luigi.OptionalBoolParameter(default=False)

    def run(self):
        # training workflow parameters
        subset_list = [True, False]
        sample_col = "species_id"
        dino_cls_sql_statement = (
            "SELECT image_name, species_id, cls_embedding FROM __THIS__"
        )
        input_path = self.input_path
        output_path = self.output_path

        # test workflow parameters
        if self.process_test_data or self.use_pretrained_dino_inference:
            subset_list = [False]
            sample_col = "image_name"
            dino_cls_sql_statement = "SELECT image_name, cls_embedding FROM __THIS__"

        # run jobs with subset and full-size data
        for subset in subset_list:
            if subset:
                subset_path = f"subset_{self.output_path.split('/')[-1]}"
                output_path = self.output_path.replace(
                    self.output_path.split("/")[-1], subset_path
                )

            # use fine-tuned DINOv2 model to extract embeddings from images
            model_path = setup_pretrained_model(
                use_only_classifier=self.use_only_classifier
            )
            sql_statement = dino_cls_sql_statement

            # process grid dataset
            if self.use_grid and self.process_test_data:
                sql_statement = (
                    "SELECT image_name, patch_number, cls_embedding FROM __THIS__"
                )
                # update input and output paths for dino embeddings
                input_path = f"{self.input_path}/grid_test_data"
                output_path = f"{output_path}/grid_dino"

            # print parameters for sanity check
            print(f"\ninput_path: {input_path}")
            print(f"output_path: {output_path}")
            print(f"model_path: {model_path}")
            print(f"subset: {subset}")
            print(f"sample_col: {sample_col}\n")
            yield [
                ProcessFineTunedDINOv2(
                    model_path=model_path,
                    input_path=input_path,
                    output_path=output_path,
                    test_workflow=subset,
                    sample_id=i,
                    num_sample_id=20,
                    sample_col=sample_col,
                    sql_statement=sql_statement,
                )
                for i in range(20)
            ]


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Luigi pipeline")
    parser.add_argument(
        "--process-test-data",
        type=bool,
        default=False,
        help="If True, set pipeline to process the test data and extract embeddings",
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
        Otherwise, use the only-classifier-then-all pretrained model,
        """,
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Get the base path for the PACE parquet files
    dataset_base_path = get_pace_data_dir()
    # Input and output paths for training workflow
    # "~/p-dsgt_clef2025-0/shared/plantclef/data"
    input_path = f"{dataset_base_path}/parquet_files/train"
    output_path = f"{dataset_base_path}/embeddings/train_embeddings"

    # parse args
    args = parse_args()
    process_test_data = args.process_test_data
    use_grid = args.use_grid
    use_only_classifier = args.use_only_classifier

    # update input and output params for embedding the test data
    if process_test_data:
        input_path = f"{dataset_base_path}/parquet_files/test"
        output_path = f"{dataset_base_path}/data/embeddings/test_embeddings"

    luigi.build(
        [
            Workflow(
                input_path=input_path,
                output_path=output_path,
                process_test_data=process_test_data,
                use_grid=use_grid,
                use_only_classifier=use_only_classifier,
            )
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
    )
