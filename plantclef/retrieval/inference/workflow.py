import csv
import luigi
import os
from pathlib import Path
import typer
from typing_extensions import Annotated
from typing import List

from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, slice as spark_slice, median, min as spark_min, explode, 
    arrays_zip, percentile_approx, collect_list, expr, lit, array_join, concat
)

from plantclef.spark import spark_resource


class RunKNNInference(luigi.Task):
    """Task that runs KNN inference."""
    
    input_path = luigi.Parameter()
    output_subdir = luigi.Parameter()
    submission_path = luigi.Parameter()
    id_map_path = luigi.Parameter()
    grid_sizes = luigi.ListParameter()
    top_k = luigi.IntParameter(default=10)
    threshold_percentile = luigi.FloatParameter(default=0.1)
    threshold_mode = luigi.Parameter(default="per_image")
    cpu_count = luigi.IntParameter(default=4)
    
    def output(self):
        return luigi.LocalTarget(f"{str(self.output_subdir)}/_SUCCESS")
        
    def run(self):
        with spark_resource(cores=self.cpu_count) as spark:
            # read knn data, filter by grid size, slice k nearest neighbors
            knn_df = spark.read.parquet(str(self.input_path))
            grid_strs = [f"{s}x{s}" for s in self.grid_sizes]
            filtered_df = (
                knn_df
                .filter(knn_df.grid.isin(grid_strs))
                .withColumn("nn_ids", spark_slice(col("nn_ids"), 1, self.top_k))
                .withColumn("distances", spark_slice(col("distances"), 1, self.top_k))
            )
            
            # explode neighbors
            exploded_df = (
                filtered_df
                .withColumn("nn_info", arrays_zip("nn_ids", "distances"))
                .withColumn("nn_info", explode("nn_info"))
                .withColumn("nn_id", col("nn_info").getItem("nn_ids"))
                .withColumn("distance", col("nn_info").getItem("distances"))
                .select("image_name", "tile", "grid", "nn_id", "distance")
            )
            
            # join with id map to get image names
            id_df = spark.read.parquet(str(self.id_map_path))
            joined_df = (
                exploded_df
                .join(id_df, exploded_df.nn_id == id_df.id, "inner")
                .drop(
                    id_df["id"],
                    id_df["image_name"],
                )
            )
            
            # median distance per tile per species
            med_per_tile_species_df = (
                joined_df
                .groupBy("image_name", "tile", "grid", "species_id")
                .agg(median("distance").alias("med_distance"))
            )
            
            # minimum median distance per species
            min_med_per_species_df = (
                med_per_tile_species_df
                .groupBy("image_name", "species_id")
                .agg(spark_min("med_distance").alias("min_med_distance"))
            )
            
            # percentile thresholding (per image or global)
            if self.threshold_mode == "per_image":
                window_spec = Window.partitionBy("image_name")
                percentile_df = min_med_per_species_df.withColumn(
                    "threshold", 
                    percentile_approx("min_med_distance", self.threshold_percentile).over(window_spec)
                )
            else:
                all_percentile_df = min_med_per_species_df.agg(
                    percentile_approx("min_med_distance", self.threshold_percentile).alias("threshold")
                )
                threshold_value = all_percentile_df.collect()[0]["threshold"]
                percentile_df = min_med_per_species_df.withColumn("threshold", lit(threshold_value))
            
            # generate predictions
            species_groups_df = (
                percentile_df
                .filter(col("min_med_distance") <= col("threshold"))
                .groupBy("image_name")
                .agg(collect_list("species_id").alias("species_ids"))
            )
            
            # handle images with zero species predictions
            all_images_df = filtered_df.select("image_name").distinct()
            predictions_df = (
                all_images_df
                .join(species_groups_df, "image_name", "left")
                .withColumn("species_ids", expr("coalesce(species_ids, array())"))
            )
            
            # format and save predictions
            formatted_df = (
                predictions_df
                .withColumn(
                    "plot_id", 
                    expr("substring_index(image_name, '.', 1)")
                )
                .withColumn("species_ids", 
                    concat(
                        lit("["), 
                        array_join("species_ids", ", "), 
                        lit("]")
                    )
                )
                .select("plot_id", "species_ids")
            )
            formatted_pd = formatted_df.toPandas().sort_values("plot_id")
            formatted_pd.to_csv(str(self.submission_path), sep=";", index=False, quoting=csv.QUOTE_NONE)
            
            # write output
            with self.output().open("w"):
                pass


class KNNInferenceWorkflow(luigi.Task):
    """Workflow to run KNN Inference."""
    
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    id_map_path = luigi.Parameter()
    grid_sizes = luigi.ListParameter()
    top_k = luigi.IntParameter(default=10)
    threshold_percentile = luigi.FloatParameter(default=0.1)
    threshold_mode = luigi.Parameter(default="per_image")
    cpu_count = luigi.IntParameter(default=4)
    
    def requires(self):
        
        top_k_str = f"topk_{self.top_k}"
        grid_sizes_str = f"grid_{'_'.join(str(s) for s in self.grid_sizes)}"
        output_name = f"{top_k_str}_{grid_sizes_str}_p_{self.threshold_percentile}_{self.threshold_mode}"
        
        output_subdir = Path(self.output_path) / top_k_str / grid_sizes_str / output_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        submission_path = output_subdir / f"dsgt_run_{output_name}.csv"
        
        task = RunKNNInference(
            input_path=self.input_path,
            output_subdir=output_subdir,
            submission_path=submission_path,
            id_map_path=self.id_map_path,
            grid_sizes=self.grid_sizes,
            top_k=self.top_k,
            threshold_percentile=self.threshold_percentile,
            threshold_mode=self.threshold_mode,
            cpu_count=self.cpu_count
        )
        yield task


def main(
    input_path: Annotated[str, typer.Argument(help="Input KNN data path")],
    output_path: Annotated[str, typer.Argument(help="Output directory")],
    id_map_path: Annotated[str, typer.Argument(help="Path to ID map file")],
    grid_sizes: Annotated[List[int], typer.Argument(help="Grid sizes to filter")],
    top_k: Annotated[int, typer.Option(help="Number of nearest neighbors")] = 10,
    threshold_percentile: Annotated[float, typer.Option(help="Percentile threshold for predictions")] = 0.1,
    threshold_mode: Annotated[str, typer.Option(help="Threshold mode")] = "per_image",
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
            KNNInferenceWorkflow(
                input_path=input_path,
                output_path=output_path,
                id_map_path=id_map_path,
                grid_sizes=grid_sizes,
                top_k=top_k,
                threshold_percentile=threshold_percentile,
                threshold_mode=threshold_mode,
                cpu_count=cpu_count,
            )
        ],
        **kwargs,
    )
