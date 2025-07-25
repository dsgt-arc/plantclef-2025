import os
import re
import pandas as pd
import json
import typer
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool
from pygbif import species, occurrences as occ
from typing import Annotated

app = typer.Typer()


def get_occurrences(taxon_key):
    response = occ.search(taxonKey=taxon_key, limit=1000000)
    return response["results"]


def load_species_data(
    data_path: str,
    file_name: str = "species_metadata.csv",
) -> pd.DataFrame:
    input_path = f"{data_path}/{file_name}"
    df = pd.read_csv(input_path)
    return df["species"].tolist()


def get_occurrences_for_species(args):
    species_name, output_dir = args

    # create filename first to check if it exists
    file_name = re.sub(r"[^a-z0-9_]", "_", species_name.lower().replace(" ", "_"))
    out_file = os.path.join(output_dir, f"{file_name}.json")

    # check if the file already exists and skip if it does
    if os.path.exists(out_file):
        return None  # Skip processing this species

    # If file doesn't exist, proceed with fetching data
    taxon_key = species.name_backbone(species_name)["usageKey"]
    response = occ.search(taxonKey=taxon_key, limit=1000000)
    species_data = {}
    countries = []
    for result in response["results"]:
        country = result.get(
            "country", "Unknown"
        )  # handle cases where "country" might be missing
        countries.append(country)
    counter_countries = Counter(countries)
    species_data[species_name] = {
        country: count for country, count in counter_countries.items()
    }

    # write to JSON
    with open(out_file, "w") as f:
        json.dump(species_data, f, indent=2)

    return species_data


@app.command()
def fetch(
    num_processes: Annotated[
        int, typer.Option(help="Number of processes to use for fetching data")
    ] = 6,
    ouptut_folder: Annotated[
        str, typer.Option(help="Output directory for the fetched data")
    ] = "country_counts",
):
    # Set output directory and create if not exists
    data_path = "~/p-dsgt_clef2025-0/shared/plantclef/data"
    output_dir = os.path.expanduser(f"{data_path}/genai/02_gbif/{ouptut_folder}")
    os.makedirs(output_dir, exist_ok=True)
    # load species data
    species_list = load_species_data(data_path, file_name="species_metadata.csv")

    # Create args tuples with both species and output_dir
    args_list = [(species, output_dir) for species in species_list]

    # multiprocessing
    with Pool(processes=num_processes) as pool:
        list(
            tqdm(
                pool.imap(get_occurrences_for_species, args_list),
                total=len(species_list),
                desc="Processing species",
            )
        )


if __name__ == "__main__":
    app()
