import luigi
import typer
import numpy as np
import faiss
from pathlib import Path
from typing_extensions import Annotated

from pyspark.sql.window import Window
from pyspark.sql.functions import lit, row_number

from plantclef.spark import spark_resource


class GenerateTrainIDMap(luigi.Task):
    """Task to generate map of IDs for training examples."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    cpu_count = luigi.IntParameter(default=4)
    
    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/train_id_map/_SUCCESS")
    
    def run(self):
        output_path = Path(self.output_path) / "train_id_map"
        
        with spark_resource(cores=self.cpu_count) as spark:
            emb_df = spark.read.parquet(str(self.input_path))
            # create ID map with row numbers
            id_map = (
                emb_df.withColumn("id", row_number().over(Window.orderBy(lit(0))) - 1)
                .select("id", "image_name", "species_id")
            )
            id_map.write.mode("overwrite").parquet(str(output_path))


class CreateFAISSIndex(luigi.Task):
    """Task to create FAISS index from embeddings."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    index_name = luigi.Parameter()
    cpu_count = luigi.IntParameter(default=4)
    
    def requires(self):
        return GenerateTrainIDMap(
            input_path=self.input_path,
            output_path=self.output_path,
            cpu_count=self.cpu_count
        )
    
    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/{self.index_name}.index")
    
    def run(self):
        output_path = Path(self.output_path) / f"{self.index_name}.index"
        
        with spark_resource(cores=self.cpu_count) as spark:
            emb_df = spark.read.parquet(str(self.input_path))
            emb_pd = emb_df.toPandas()
            
            dim = len(emb_pd["cls_embedding"].iloc[0])
            index = faiss.IndexFlatL2(dim)
            
            emb_df = np.stack(emb_pd["cls_embedding"].values).astype("float32")
            faiss.normalize_L2(emb_df)
            index.add(emb_df)
            
            faiss.write_index(index, str(output_path))


class Workflow(luigi.Task):
    """Workflow to generate train ID map and FAISS index."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    index_name = luigi.Parameter()
    cpu_count = luigi.IntParameter(default=4)
    
    def requires(self):
        return CreateFAISSIndex(
            input_path=self.input_path,
            output_path=self.output_path,
            index_name=self.index_name,
            cpu_count=self.cpu_count
        )


def main(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    index_name: Annotated[str, typer.Argument(help="Name of FAISS index")],
    cpu_count: Annotated[int, typer.Option(help="Number of CPUs")] = 4,
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
                cpu_count=cpu_count,
            )
        ],
        **kwargs,
    )
