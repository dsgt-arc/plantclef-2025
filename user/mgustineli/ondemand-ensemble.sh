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
knn_folder_name="knn/topk_10/grid_2_3_4_5_6/topk_10_grid_2_3_4_5_6_p_0.03_global"
clf_folder_name="topk_10_species_grid_4x4"  # without the .csv
jaccard_threshold=0.2

# run the Python script
plantclef ensemble workflow \
    $project_data_dir/submissions \
    $knn_folder_name \
    $clf_folder_name \
    --jaccard-threshold $jaccard_threshold
