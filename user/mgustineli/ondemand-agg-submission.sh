#!/bin/bash
set -xe

# print system info
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/scratch/plantclef/.venv/bin/activate

# we select a submission .csv file that was created from the classification workflow
# stored in the submissions folder. The folder name is the name of the submission folder,
# and top_k is the number of species to be selected for the final aggregation for each species
scratch_data_dir=$(realpath ~/scratch/plantclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef
testset_name=test_2025
top_k_species=10
top_k_logits=10
grid_size=4
folder_name="topk_${top_k_logits}_species_grid_${grid_size}x${grid_size}"  # without the .csv

# run the aggregation script
plantclef classification aggregation \
    $project_data_dir/submissions/$testset_name \
    $testset_name \
    --folder-name $folder_name \
    --top-k $top_k_species \

# run the geolocation script on the aggregation submisssion results
file_name="agg_topk${top_k_species}_dsgt_run_${folder_name}.csv"
plantclef classification aggregation_geolocation \
    $file_name \
    $testset_name \
    --folder-name $folder_name \
