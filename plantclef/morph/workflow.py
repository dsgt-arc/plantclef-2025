from plantclef.spark import get_spark
from .operations import opening, closing
from .stats import mask_mean, mask_num_components
from plantclef.serde import deserialize_mask
from typing import Iterable
import typer
from pyspark.sql import functions as F

app = typer.Typer()


def generate_mask_stats(mask, iterations: Iterable[int]) -> list[dict]:
    """Calculate the statistics for the masks."""
    stats = []
    for i in iterations:
        opened = opening(mask, i)
        closed = closing(mask, i)
        opened_closed = closing(opened, i)
        stats.append(
            {
                "iteration": i,
                "opening_mean": mask_mean(opened),
                "opening_num_components": mask_num_components(opened),
                "closing_mean": mask_mean(closed),
                "closing_num_components": mask_num_components(closed),
                "opening_closing_mean": mask_mean(opened_closed),
                "opening_closing_num_components": mask_num_components(opened_closed),
            }
        )
    return stats


def apply_mask_stats(
    df, nonmask_columns, mask_columns, iterations_max, iterations_step
):
    """Apply the mask statistics to the dataframe."""

    @F.udf("""
        array<
            struct<
                iteration:int,
                opening_mean:float,
                opening_num_components:int,
                closing_mean:float,
                closing_num_components:int,
                opening_closing_mean:float,
                opening_closing_num_components:int
            >
        >
        """)
    def _generate_mask_stats(mask: bytearray):
        mask = deserialize_mask(mask)
        return generate_mask_stats(
            mask, range(iterations_step, iterations_max, iterations_step + 1)
        )

    return df.select(
        *nonmask_columns,
        *[_generate_mask_stats(F.col(c)).alias(f"{c}_stats") for c in mask_columns],
    )


@app.command("mask-stats")
def mask_stats_workflow(
    input_path: str,
    output_path: str,
    iterations_max: int = 100,
    iterations_step: int = 5,
    num_partitions: int = 32,
):
    """Calculate the statistics for the masks.

    Example schema of the input data:

    /storage/home/hcoda1/8/amiyaguchi3/shared/plantclef/data/masking/test_2024_v2/data
        root
        |-- image_name: string (nullable = true)
        |-- leaf_mask: binary (nullable = true)
        |-- flower_mask: binary (nullable = true)
        |-- plant_mask: binary (nullable = true)
        |-- sand_mask: binary (nullable = true)
        |-- wood_mask: binary (nullable = true)
        |-- tape_mask: binary (nullable = true)
        |-- tree_mask: binary (nullable = true)
        |-- rock_mask: binary (nullable = true)
        |-- vegetation_mask: binary (nullable = true)
        |-- sample_id: integer (nullable = true)
    """
    df = get_spark().read.parquet(input_path)

    applied_df = apply_mask_stats(
        df,
        [c for c in df.columns if "mask" not in c],
        [c for c in df.columns if "mask" in c],
        iterations_max,
        iterations_step,
    ).repartition(num_partitions)

    applied_df.printSchema()
    applied_df.write.parquet(output_path, mode="overwrite")
