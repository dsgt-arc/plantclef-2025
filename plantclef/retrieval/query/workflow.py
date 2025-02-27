import luigi
import typer
import numpy as np
import faiss
from pathlib import Path
from typing_extensions import Annotated

from pyspark.ml import Pipeline
from pyspark.sql.window import Window
from pyspark.sql.functions import lit, row_number
from pyspark.sql.types import ArrayType, FloatType, IntegerType

from plantclef.retrieval.query.index_setup import setup_index
from plantclef.retrieval.query.transform import FaissNearestNeighbors
from plantclef.spark import spark_resource


class FindNearestNeighbors(luigi.Task):
    """Task to search for nearest neighbors of test embeddings."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    index_path = luigi.Parameter(default=None)
    index_name = luigi.Parameter(default="embeddings")
    # id_map_path = luigi.Parameter(default=str(Path.home() / "scratch/plantclef/data/faiss/train"))
    k = luigi.IntParameter(default=10)
    cpu_count = luigi.IntParameter(default=4)
    num_partitions = luigi.OptionalIntParameter(default=20)
    
    def output(self):
        return luigi.LocalTarget(
            f"{self.output_path}/_SUCCESS"
        )
        
    def pipeline(self):
        return Pipeline(
            stages=[
                FaissNearestNeighbors(
                    input_col="cls_embedding",
                    index_path=self.index_path,
                    index_name=self.index_name,
                    k=self.k
                ),
            ]
        )
    
    def run(self):
        
        with spark_resource(cores=self.cpu_count) as spark:
            # df = spark.read.parquet(str(self.input_path))
            # pipeline_model = self.pipeline().fit(df)
            # transformed = pipeline_model.transform(df)
            
            if self.index_path is None:
                self.index_path = setup_index(scratch_model=True, index_name=self.index_name)
            
            index = faiss.read_index(str(self.index_path))
            emb_df = spark.read.parquet(str(self.input_path))
            
            emb_df = emb_df.drop("sample_id")

            new_schema = emb_df.schema \
                .add("distances", ArrayType(FloatType())) \
                .add("nn_ids", ArrayType(IntegerType()))

            def process_partition(rows):
                batch = list(rows)
                if not batch:
                    return
                emb_list = [row.cls_embedding for row in batch]
                embeddings = np.stack(emb_list).astype("float32")
                faiss.normalize_L2(embeddings)
                distances, ids = index.search(embeddings, self.k)
                for row, dist, nbr in zip(batch, distances, ids):
                    yield row + (dist.tolist(), nbr.tolist())

            out_rdd = emb_df.rdd.mapPartitions(process_partition)
            out_df = spark.createDataFrame(out_rdd, schema=new_schema)
            
            # emb_pd = emb_df.toPandas()
            # embeddings = np.stack(emb_pd["cls_embedding"].values).astype("float32")
            # faiss.normalize_L2(embeddings)
            # distances, ids = index.search(embeddings, self.k)
            
            # emb_pd["distances"] = [d.tolist() for d in distances]
            # emb_pd["nn_ids"] = [i.tolist() for i in ids]
            
            # out_df = spark.createDataFrame(emb_pd)
            
            out_df.printSchema()
            out_df.explain()
            (
                out_df.repartition(self.num_partitions)
                .cache()
                .write.mode("overwrite")
                .parquet(str(self.output_path))
            )


class Workflow(luigi.Task):
    """Workflow to generate train ID map and FAISS index."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    index_name = luigi.Parameter(default="embeddings")
    k = luigi.IntParameter(default=10)
    cpu_count = luigi.IntParameter(default=4)
    num_partitions = luigi.IntParameter(default=10)
    
    def requires(self):
        return FindNearestNeighbors(
            input_path=self.input_path,
            output_path=self.output_path,
            index_name=self.index_name,
            k=self.k,
            cpu_count=self.cpu_count,
            num_partitions=self.num_partitions,
        )


def main(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    index_name: Annotated[str, typer.Argument(help="Name of FAISS index")] = "embeddings",
    k: Annotated[int, typer.Option(help="Number of nearest neighbors")] = 10,
    cpu_count: Annotated[int, typer.Option(help="Number of CPUs")] = 4,
    num_partitions: Annotated[int, typer.Option(help="Number of partitions")] = 10,
    scheduler_host: Annotated[str, typer.Option(help="Scheduler host")] = None,
):
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
                index_name=index_name,
                k=k,
                cpu_count=cpu_count,
                num_partitions=num_partitions,
            )
        ],
        **kwargs,
    )