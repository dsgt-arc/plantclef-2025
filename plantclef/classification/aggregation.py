import os
import re
import csv
import typer
from typing_extensions import Annotated
import pandas as pd
from pathlib import Path
from collections import Counter


def get_pandas_dataframe(
    input_path: str, folder_name: str, file_name: str
) -> pd.DataFrame:
    # read CSV file
    file_path = f"{folder_name}/{file_name}"
    df = pd.read_csv(f"{input_path}/{file_path}", delimiter=",")
    return df


# regex-based function to extract base quadrat id
def extract_base_quadrat_id(quadrat_id):
    patterns = [
        (r"^(CBN-.*?-.*?)-\d{8}$", 1),  # CBN
        (r"^(GUARDEN-.*?-.*?)-.*$", 1),  # GUARDEN-AMB
        (r"^(LISAH-.*?)-\d{8}$", 1),  # LISAH
        (r"^(OPTMix-\d+)-.*$", 1),  # OPTMix
        (r"^(RNNB-\d+-\d+)-\d{8}$", 1),  # RNNB
    ]
    for pattern, group_idx in patterns:
        match = re.match(pattern, quadrat_id)
        if match:
            return match.group(group_idx)
    return quadrat_id  # fallback


def union_agg(x):
    return list(set([item for sublist in x for item in sublist]))


# Function to aggregate species and sort them by frequency
def union_agg_sorted(x):
    species_counts = Counter([species for sublist in x for species in sublist])
    return [species for species, _ in species_counts.most_common()]


# function to aggregate species, sort them by frequency, and select top-K species
def union_agg_topk(x, top_k=5):
    species_counts = Counter([species for sublist in x for species in sublist])
    return [species for species, _ in species_counts.most_common(top_k)]


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
    input_path: Annotated[str, typer.Argument(help="Input root dir of the CSV file")],
    testset_name: Annotated[str, typer.Argument(help="Testset name")],
    folder_name: Annotated[str, typer.Option(help="CSV file to use for aggregation")],
    top_k: Annotated[int, typer.Option(help="Top K species to select")] = 10,
):
    # define the file name
    FILE_NAME = f"dsgt_run_{folder_name}.csv"
    df = get_pandas_dataframe(input_path, folder_name, FILE_NAME)

    # apply extract_base_quadrat_id function to the quadrat_id column
    df["base_quadrat_id"] = df["quadrat_id"].apply(extract_base_quadrat_id)

    # convert species_ids from string to list of integers
    df["species_ids"] = df["species_ids"].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )
    df["species_ids"] = df["species_ids"].apply(lambda x: list(map(int, x)))

    # aggregate species and sort them by frequency
    df_union = (
        df.groupby("base_quadrat_id")["species_ids"]
        .apply(union_agg_sorted)
        .reset_index()
    )
    # merge aggregated species back to the original DataFrame
    df_merged = df.merge(
        df_union,
        on="base_quadrat_id",
        how="left",
        suffixes=("_original", "_aggregated"),
    )
    # select the required columns
    df_final = df_merged[["quadrat_id", "species_ids_aggregated"]].rename(
        columns={"species_ids_aggregated": "species_ids"}
    )
    # prepare and write the submission
    sub_file_name = f"agg_{FILE_NAME}"
    write_csv_to_pace(df_final, sub_file_name, testset_name, folder_name)

    # top-K species aggregation
    # group by base_quadrat_id and apply the updated aggregation function with top K filtering
    df_union_topk = (
        df.groupby("base_quadrat_id")["species_ids"]
        .apply(lambda x: union_agg_topk(x, top_k))
        .reset_index()
    )
    # merge the aggregated species_ids back to the original df
    df_merged_topk = df.merge(
        df_union_topk,
        on="base_quadrat_id",
        how="left",
        suffixes=("_original", "_aggregated"),
    )

    # select only the required columns: original quadrat_id and aggregated species_ids
    df_final_topk = df_merged_topk[["quadrat_id", "species_ids_aggregated"]].rename(
        columns={"species_ids_aggregated": "species_ids"}
    )
    topk_file_name = f"agg_topk{top_k}_{FILE_NAME}"
    write_csv_to_pace(df_final_topk, topk_file_name, testset_name, folder_name)
