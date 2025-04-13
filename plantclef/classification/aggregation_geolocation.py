import os
import ast
import csv
import typer
from typing_extensions import Annotated
import pandas as pd
from pathlib import Path


def get_pandas_dataframe(
    input_path: str,
    folder_name: str,
    file_name: str,
) -> pd.DataFrame:
    # read CSV file
    file_path = f"{folder_name}/{file_name}"
    df = pd.read_csv(f"{input_path}/{file_path}", delimiter=",")
    return df


def filter_species_by_country(
    filtered_df: pd.DataFrame,
    sub_df: pd.DataFrame,
) -> pd.DataFrame:
    sub_df = sub_df.copy()
    sub_df["species_ids"] = sub_df["species_ids"].apply(lambda x: ast.literal_eval(x))

    # get valid species_ids from joined_df
    valid_species_ids = filtered_df["species_id"].tolist()

    # filter each list of species_ids insub_df
    sub_df["species_ids"] = sub_df["species_ids"].apply(
        lambda id_list: [
            species_id for species_id in id_list if species_id in valid_species_ids
        ]
    )
    return sub_df


def format_species_ids(species_ids: list) -> str:
    """Formats the species IDs in single square brackets, separated by commas."""
    formatted_ids = ", ".join(str(id) for id in species_ids)
    return f"[{formatted_ids}]"


def prepare_and_write_submission(pandas_df: pd.DataFrame) -> pd.DataFrame:
    """Converts Spark DataFrame to Pandas, formats it, and writes to GCS."""
    records = []
    for _, row in pandas_df.iterrows():
        logits = row["species_ids"]
        formatted_species = format_species_ids(logits)
        records.append(
            {"quadrat_id": row["quadrat_id"], "species_ids": formatted_species}
        )

    pandas_df = pd.DataFrame(records)
    return pandas_df


def get_plantclef_dir() -> str:
    home_dir = Path(os.path.expanduser("~"))
    return f"{home_dir}/p-dsgt_clef2025-0/shared/plantclef/"


def write_csv_to_pace(
    df,
    file_name: str,
    testset_name: str,
    folder_name: str,
):
    """Writes the Pandas DataFrame to a CSV file on PACE."""

    # prepare and write the submission
    submission_df = prepare_and_write_submission(df)
    project_dir = get_plantclef_dir()
    submission_path = (
        f"{project_dir}/submissions/aggregation_seasons/{testset_name}/{folder_name}"
    )
    output_path = f"{submission_path}/{file_name}"
    # ensure directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # write to CSV
    submission_df.to_csv(output_path, sep=",", index=False, quoting=csv.QUOTE_ALL)
    print(f"Submission file saved to: {output_path}")


def main(
    file_name: Annotated[str, typer.Argument(help="CSV file to use for aggregation")],
    testset_name: Annotated[str, typer.Argument(help="Testset name")],
    folder_name: Annotated[str, typer.Option(help="CSV file to use for aggregation")],
):
    project_path = "~/p-dsgt_clef2025-0/shared/plantclef"
    submission_path = f"{project_path}/submissions/aggregation_seasons/{testset_name}"
    sub_df = get_pandas_dataframe(
        submission_path,
        folder_name,
        file_name,
    )

    # define the file name
    FILE_NAME = f"dsgt_run_{folder_name}.csv"

    # import filtered data based on geolocation of France
    france_path = f"{project_path}/data/france_geodata_species.csv"
    france_df = pd.read_csv(france_path)

    # get the filtered data
    final_df = filter_species_by_country(france_df, sub_df)

    sub_file_name = f"geo_agg_topk10_{FILE_NAME}"
    write_csv_to_pace(final_df, sub_file_name, testset_name, folder_name)
