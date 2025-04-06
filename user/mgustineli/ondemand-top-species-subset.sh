#!/bin/bash
set -xe

# print system info
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/scratch/plantclef/venv/bin/activate

# set environment variables
export PYSPARK_DRIVER_MEMORY=20g
export PYSPARK_EXECUTOR_MEMORY=20g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp

# define variables
scratch_data_dir=$(realpath ~/scratch/plantclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data
top_n=5

# Run Python script
plantclef preprocessing create_top_species_subset \
    $project_data_dir/parquet/train \
    $project_data_dir/parquet/subset_top${top_n}_train \
    --cores 10 \
    --memory 20g \
    --top-n $top_n
