#!/bin/bash
set -xe

# Print system info
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# Activate the environment
source ~/clef/plantclef-2025/scripts/activate.sh

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"  # Check if PyTorch can access the GPU
nvidia-smi                                                  # Check GPU usage

# Start the NVIDIA monitoring job in the background
NVIDIA_LOG_FILE=Report-nvidia-logs.ndjson
python ~/clef/plantclef-2025/scripts/nvidia-logs.sh monitor "$NVIDIA_LOG_FILE" --interval 15 &
nvidia_logs_pid=$!
echo "Started NVIDIA monitoring process with PID ${nvidia_logs_pid}"

# Set environment variables
export PYSPARK_DRIVER_MEMORY=10g
export PYSPARK_EXECUTOR_MEMORY=10g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp

# Define paths
scratch_data_dir=$(realpath ~/scratch/plantclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data
dataset_name=test_2024
num_rows=20

# Run the Python script
plantclef preprocessing create_test_subset \
    $project_data_dir/parquet/$dataset_name \
    $project_data_dir/parquet/${dataset_name}_subset${num_rows} \
    --num-rows $num_rows

# Parse the NVIDIA monitoring output
python ~/clef/plantclef-2025/scripts/nvidia-logs.sh parse $NVIDIA_LOG_FILE
