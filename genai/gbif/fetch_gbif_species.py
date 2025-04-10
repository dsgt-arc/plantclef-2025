import os
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool
from pygbif import species, occurrences as occ


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


# Set output directory and create if not exists
data_path = "~/p-dsgt_clef2025-0/shared/plantclef/data"
output_dir = os.path.expanduser(f"{data_path}/genai/02_gbif")
os.makedirs(output_dir, exist_ok=True)
# load species data
species_list = load_species_data(data_path, file_name="species_metadata.csv")


def get_occurrences_for_species(species_name):
    taxon_key = species.name_backbone(species_name)["usageKey"]
    response = occ.search(taxonKey=taxon_key, limit=1000000)
    countries = set()
    for result in response["results"]:
        country = result.get(
            "country", "Unknown"
        )  # handle cases where "country" might be missing
        countries.add(country)
        # add country to species_dict
    species_data = {species_name: list(country)}

    # write to JSON
    out_file = os.path.join(output_dir, f"{species_name.replace(" ", "_")}.json")
    with open(out_file, "w") as f:
        json.dump(species_data, f, indent=2)

    return species_data


if __name__ == "__main__":
    species_list = load_species_data()
    num_processes = 6

    with Pool(processes=num_processes) as pool:
        list(
            tqdm(
                pool.imap(get_occurrences_for_species, species_list),
                total=len(species_list),
                desc="Processing species",
            )
        )
