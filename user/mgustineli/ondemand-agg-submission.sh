#!/bin/bash
set -xe

# print system info
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/scratch/plantclef/venv/bin/activate

# we select a submission .csv file that was created from the classification workflow
# stored in the submissions folder. The folder name is the name of the submission folder,
# and top_k is the number of species to be selected for the final aggregation for each species
scratch_data_dir=$(realpath ~/scratch/plantclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/
folder_name="topk_20_species_grid_4x4"  # without the .csv
top_k=10

# run the Python script
plantclef classification aggregation \
    $project_data_dir/submissions \
    --folder-name $folder_name \
    --top-k $top_k
