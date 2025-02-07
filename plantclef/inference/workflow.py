# from argparse import ArgumentParser

# import luigi
# import luigi.contrib.gcs
# from pyspark.ml import Pipeline
# from pyspark.ml.feature import SQLTransformer
# from pyspark.ml.functions import vector_to_array
# from pyspark.sql import DataFrame
# from pyspark.sql import functions as F

# from plantclef.model_setup import setup_pretrained_model
# from plantclef.transforms import PretrainedDinoV2
# from plantclef.spark import spark_resource
# from plantclef.config import get_pace_data_dir


# class ProcessBase(luigi.Task):
#     input_path = luigi.Parameter()
#     output_path = luigi.Parameter()
#     should_subset = luigi.BoolParameter(default=False)
#     sample_col = luigi.Parameter(default="species_id")
#     num_partitions = luigi.OptionalIntParameter(default=500)
#     sample_id = luigi.OptionalIntParameter(default=None)
#     num_sample_id = luigi.OptionalIntParameter(default=20)

#     def output(self):
#         if self.sample_id is None:
#             # save both the model pipeline and the dataset
#             return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/data/_SUCCESS")
#         else:
#             return luigi.contrib.gcs.GCSTarget(
#                 f"{self.output_path}/data/sample_id={self.sample_id}/_SUCCESS"
#             )

#     @property
#     def feature_columns(self) -> list:
#         raise NotImplementedError()

#     def pipeline(self) -> Pipeline:
#         raise NotImplementedError()

#     def transform(self, model, df, features) -> DataFrame:
#         transformed = model.transform(df)

#         if self.sample_id is not None:
#             transformed = (
#                 transformed.withColumn(
#                     "sample_id",
#                     F.crc32(F.col(self.sample_col).cast("string")) % self.num_sample_id,
#                 )
#                 .where(F.col("sample_id") == self.sample_id)
#                 .drop("sample_id")
#             )

#         for c in features:
#             # check if the feature is a vector and convert it to an array
#             if "array" in transformed.schema[c].simpleString():
#                 continue
#             transformed = transformed.withColumn(c, vector_to_array(F.col(c)))
#         return transformed

#     def _get_subset(self, df):
#         # Get subset of images to test pipeline
#         subset_df = (
#             df.where(F.col(self.sample_col).isin([1361703, 1355927]))
#             .orderBy(F.rand(1000))
#             .limit(200)
#             .cache()
#         )
#         return subset_df

#     def run(self):
#         with spark_resource(
#             **{"spark.sql.shuffle.partitions": self.num_partitions}
#         ) as spark:
#             df = spark.read.parquet(self.input_path)

#             if self.should_subset:
#                 # Get subset of data to test pipeline
#                 df = self._get_subset(df=df)

#             model = self.pipeline().fit(df)
#             model.write().overwrite().save(f"{self.output_path}/model")
#             transformed = self.transform(model, df, self.feature_columns)

#             if self.sample_id is None:
#                 output_path = f"{self.output_path}/data"
#             else:
#                 output_path = f"{self.output_path}/data/sample_id={self.sample_id}"

#             transformed.repartition(self.num_partitions).write.mode(
#                 "overwrite"
#             ).parquet(output_path)


# class ProcessPretrainedDinoInference(ProcessBase):
#     sql_statement = luigi.Parameter()
#     pretrained_path = luigi.Parameter()
#     use_grid = luigi.OptionalBoolParameter(default=False)
#     grid_size = luigi.OptionalIntParameter(default=3)

#     @property
#     def feature_columns(self) -> list:
#         return ["dino_logits"]

#     def pipeline(self):
#         pretrained_dino = PretrainedDinoV2(
#             pretrained_path=self.pretrained_path,
#             input_col="data",
#             output_col="dino_logits",
#             use_grid=self.use_grid,
#             grid_size=self.grid_size,
#         )
#         return Pipeline(
#             stages=[pretrained_dino, SQLTransformer(statement=self.sql_statement)]
#         )


# class Workflow(luigi.Task):
#     input_path = luigi.Parameter()
#     output_path = luigi.Parameter()
#     default_root_dir = luigi.Parameter()
#     process_test_data = luigi.OptionalBoolParameter(default=False)
#     use_cls_token = luigi.OptionalBoolParameter(default=False)
#     use_pretrained_dino_inference = luigi.OptionalBoolParameter(default=False)
#     use_grid = luigi.OptionalBoolParameter(default=False)
#     use_only_classifier = luigi.OptionalBoolParameter(default=False)
#     use_pretrained_embeddings = luigi.OptionalBoolParameter(default=False)

#     def run(self):
#         # training workflow parameters
#         subset_list = [True, False]
#         sample_col = "species_id"
#         pretrained_sql_statement = (
#             "SELECT image_name, species_id, dino_logits FROM __THIS__"
#         )

#         # test workflow parameters
#         if self.process_test_data or self.use_pretrained_dino_inference:
#             subset_list = [False]
#             sample_col = "image_name"
#             pretrained_sql_statement = "SELECT image_name, dino_logits FROM __THIS__"

#         # Run jobs with subset and full-size data
#         for subset in subset_list:
#             final_output_path = self.output_path
#             if subset:
#                 subset_path = f"subset_{self.output_path.split('/')[-1]}"
#                 final_output_path = self.output_path.replace(
#                     self.output_path.split("/")[-1], subset_path
#                 )
#             # use Pretrained model to do inference on test data
#             if self.use_pretrained_dino_inference:
#                 grid_size = 3
#                 top_k_proba = 5
#                 data_path = "dino_pretrained"
#                 # inference_dir = self.default_root_dir
#                 if self.use_only_classifier:
#                     data_path = f"{data_path}_only_classifier"
#                     # inference_dir = f"{self.default_root_dir}-only-classifier"
#                 if self.use_grid:
#                     data_path = f"{data_path}_grid_{grid_size}x{grid_size}"
#                 pretrained_path = setup_pretrained_model(self.use_only_classifier)
#                 print(f"\ninput_path: {self.input_path}")
#                 print(f"pretrained_path: {pretrained_path}")
#                 print(f"subset: {subset}")
#                 print(f"sample_col: {sample_col}\n")
#                 yield ProcessPretrainedDinoInference(
#                     pretrained_path=pretrained_path,
#                     input_path=self.input_path,
#                     output_path=f"{final_output_path}/{data_path}",
#                     should_subset=subset,
#                     sample_col=sample_col,
#                     sql_statement=pretrained_sql_statement,
#                     use_grid=self.use_grid,
#                     grid_size=grid_size,
#                 )


# def parse_args():
#     """Parse command-line arguments."""
#     # Get the base path for the PACE dataset
#     dataset_base_path = get_pace_data_dir()

#     parser = ArgumentParser(description="Luigi pipeline")
#     parser.add_argument(
#         "--train-data-path",
#         type=str,
#         default=f"{dataset_base_path}/parquet_files/train",
#         help="Root directory for training parquet data in PACE",
#     )
#     parser.add_argument(
#         "--output-name-path",
#         type=str,
#         default=f"{dataset_base_path}/embeddings/train_embeddings",
#         help="PACE path for output Parquet files",
#     )
#     parser.add_argument(
#         "--process-test-data",
#         type=bool,
#         default=False,
#         help="If True, set pipeline to process the test data and extract embeddings",
#     )
#     parser.add_argument(
#         "--use-pretrained-dino-inference",
#         type=bool,
#         default=False,
#         help="If True, use the pretrained DINOv2 ViT model to make predictions on test data",
#     )
#     parser.add_argument(
#         "--use-grid",
#         type=bool,
#         default=False,
#         help="If True, create a grid when using the pretrained ViT model to make predictions",
#     )
#     parser.add_argument(
#         "--use-pretrained-embeddings",
#         type=bool,
#         default=False,
#         help="If True, process the embeddings using the pretrained ViT model",
#     )
#     return parser.parse_args()
