import os
import csv
import typer
from typing_extensions import Annotated
import pandas as pd
from pathlib import Path


def get_pandas_dataframe(
    input_path: str, folder_name: str, file_name: str
) -> pd.DataFrame:
    # read CSV file
    file_path = f"{folder_name}/{file_name}"
    df = pd.read_csv(f"{input_path}/{file_path}", delimiter=";")
    return df


# union function
def union_species_ids(knn_df, clf_df):
    merged_df = knn_df.merge(
        clf_df, on="plot_id", how="outer", suffixes=("_knn", "_clf")
    )
    merged_df["species_ids"] = merged_df.apply(
        lambda row: list(
            set(
                eval(row["species_ids_knn"]) if pd.notna(row["species_ids_knn"]) else []
            )
            | set(
                eval(row["species_ids_clf"]) if pd.notna(row["species_ids_clf"]) else []
            )
        ),
        axis=1,
    )
    return merged_df[["plot_id", "species_ids"]]


# inner join function
def inner_join_species_ids(knn_df, clf_df):
    merged_df = knn_df.merge(
        clf_df, on="plot_id", how="inner", suffixes=("_knn", "_clf")
    )
    merged_df["species_ids"] = merged_df.apply(
        lambda row: list(
            set(eval(row["species_ids_knn"])) & set(eval(row["species_ids_clf"]))
        ),
        axis=1,
    )
    return merged_df[["plot_id", "species_ids"]]


# define Jaccard similarity-based ensemble function
def jaccard_ensemble(knn_df, clf_df, threshold=0.5):
    merged_df = knn_df.merge(
        clf_df, on="plot_id", how="outer", suffixes=("_knn", "_clf")
    )

    def compute_jaccard(row):
        species_knn = (
            set(eval(row["species_ids_knn"]))
            if pd.notna(row["species_ids_knn"])
            else set()
        )
        species_clf = (
            set(eval(row["species_ids_clf"]))
            if pd.notna(row["species_ids_clf"])
            else set()
        )

        if not species_knn or not species_clf:
            return list(species_knn | species_clf)  # if one is empty, return the other

        intersection = species_knn & species_clf
        union = species_knn | species_clf
        jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0

        if jaccard_similarity >= threshold:
            return list(union)  # high similarity -> use union
        else:
            return (
                list(intersection) if intersection else list(species_knn)
            )  # low similarity -> use intersection

    merged_df["species_ids"] = merged_df.apply(compute_jaccard, axis=1)
    return merged_df[["plot_id", "species_ids"]]


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
        records.append({"plot_id": row["plot_id"], "species_ids": formatted_species})

    pandas_df = pd.DataFrame(records)
    return pandas_df


def get_plantclef_dir() -> str:
    home_dir = Path(os.path.expanduser("~"))
    return f"{home_dir}/p-dsgt_clef2025-0/shared/plantclef/"


def write_csv_to_pace(df, file_name: str):
    """Writes the Pandas DataFrame to a CSV file in GCS."""

    # prepare and write the submission
    submission_df = prepare_and_write_submission(df)
    project_dir = get_plantclef_dir()
    submission_path = f"{project_dir}/submissions/ensemble"
    output_path = f"{submission_path}/{file_name}"
    # ensure directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # write to CSV
    submission_df.to_csv(output_path, sep=";", index=False, quoting=csv.QUOTE_NONE)
    print(f"Submission file saved to: {output_path}")


def main(
    input_path: Annotated[str, typer.Argument(help="Input root dir of the CSV file")],
    knn_folder_name: Annotated[
        str, typer.Argument(help="KNN CSV file to use for ensemble")
    ],
    clf_folder_name: Annotated[
        str, typer.Argument(help="Classification CSV file to use for ensemble")
    ],
    jaccard_threshold: Annotated[
        float, typer.Option(help="Threshold for Jaccard Similarity")
    ] = 0.5,
):
    # define the file name
    KNN_FILE_NAME = f"dsgt_run_{knn_folder_name.split('/')[-1]}.csv"
    CLF_FILE_NAME = f"dsgt_run_{clf_folder_name}.csv"
    knn_df = get_pandas_dataframe(input_path, knn_folder_name, KNN_FILE_NAME)
    clf_df = get_pandas_dataframe(input_path, clf_folder_name, CLF_FILE_NAME)

    # union and inner join
    union_df = union_species_ids(knn_df, clf_df)
    # inner_join_df = inner_join_species_ids(knn_df, clf_df)
    # jaccard similarity
    jaccard_df = jaccard_ensemble(knn_df, clf_df, threshold=jaccard_threshold)

    # write DFs
    union_file_name = f"union_{CLF_FILE_NAME}_{KNN_FILE_NAME}"
    write_csv_to_pace(union_df, union_file_name)
    jaccard_file_name = f"jacc_{CLF_FILE_NAME}_{KNN_FILE_NAME}"
    write_csv_to_pace(jaccard_df, jaccard_file_name)
