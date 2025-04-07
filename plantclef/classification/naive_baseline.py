import os
import csv
import typer
import pandas as pd
from pathlib import Path
from typing_extensions import Annotated
from pyspark.sql import functions as F
from plantclef.spark import get_spark


def get_plantclef_dir() -> str:
    home_dir = Path(os.path.expanduser("~"))
    return f"{home_dir}/p-dsgt_clef2025-0/shared/plantclef"


def read_spark_dataframe(spark, dataset_name: str):
    # path and dataset names
    data_path = f"{get_plantclef_dir()}/data/parquet"
    input_path = f"{data_path}/{dataset_name}"
    df = spark.read.parquet(input_path)
    return df


# groupby function to get top-K species
def get_top_species_ids(train_df, top_k: int = 10):
    """
    Get the top K species IDs from the training DataFrame.

    Args:
        train_df (DataFrame): The training DataFrame.
        K (int): The number of top species to select.

    Returns:
        list: A list of top K species IDs.
    """
    # group by species and count the number of images
    grouped_train_df = (
        train_df.groupBy(["species", "species_id"])
        .agg(F.count("species_id").alias("n"))
        .orderBy(F.col("n").desc())
    )

    # select top-K species into list
    subset_species = grouped_train_df.select("species_id").limit(top_k).collect()
    return [row["species_id"] for row in subset_species]


def format_species_ids(species_ids: list) -> str:
    """Formats the species IDs in single square brackets, separated by commas."""
    formatted_ids = ", ".join(str(id) for id in species_ids)
    return f"[{formatted_ids}]"


def prepare_and_write_submission(pandas_df: pd.DataFrame) -> pd.DataFrame:
    """Formats the Pandas DataFrame, and writes to PACE."""
    records = []
    for _, row in pandas_df.iterrows():
        logits = row["species_ids"]
        formatted_species = format_species_ids(logits)
        records.append(
            {"quadrat_id": row["image_name"], "species_ids": formatted_species}
        )

    pandas_df = pd.DataFrame(records)
    return pandas_df


def write_csv_to_pace(df, file_name: str):
    """Writes the Pandas DataFrame to a CSV file on PACE."""

    # prepare and write the submission
    submission_df = prepare_and_write_submission(df)
    project_dir = get_plantclef_dir()
    submission_path = f"{project_dir}/submissions/naive_baseline"
    output_path = f"{submission_path}/{file_name}"
    # ensure directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # write to CSV
    submission_df.to_csv(output_path, sep=",", index=False, quoting=csv.QUOTE_ALL)
    print(f"Submission file saved to: {output_path}")


def main(
    top_k: Annotated[int, typer.Option(help="Top K species to select")] = 10,
):
    # get the spark session
    spark = get_spark()

    # define the file name
    train_df = read_spark_dataframe(spark, dataset_name="train")
    test_df = read_spark_dataframe(spark, dataset_name="test_2025")

    # Get top-K species IDs
    top_species_list = get_top_species_ids(train_df, top_k=top_k)

    # select image names from test DataFrame
    image_names_df = test_df.select("image_name").orderBy(F.col("image_name"))
    image_names = [row["image_name"] for row in image_names_df.collect()]

    # create pandas DataFrame with image names
    pandas_dict = {
        "image_name": image_names,
        "species_ids": [top_species_list] * len(image_names),
    }
    pandas_df = pd.DataFrame(pandas_dict, columns=["image_name", "species_ids"])

    # format the species_ids column
    file_name = f"dsgt_naive_{top_k}.csv"
    write_csv_to_pace(pandas_df, file_name=file_name)
