#!/bin/bash
set -xe
# export NO_REINSTALL=1

# Print system info
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# Activate the environment
source ~/clef/plantclef-2025/scripts/activate.sh

# Check GPU availability
GPU_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [[ "$GPU_AVAILABLE" == "True" ]]; then
    echo "GPU detected. Running in GPU mode."

    # Check GPU usage
    nvidia-smi

    # Start the NVIDIA monitoring job in the background
    NVIDIA_LOG_FILE=Report-nvidia-logs.ndjson
    python ~/clef/plantclef-2025/scripts/nvidia-logs.sh monitor "$NVIDIA_LOG_FILE" --interval 15 &
    nvidia_logs_pid=$!
    echo "Started NVIDIA monitoring process with PID ${nvidia_logs_pid}"
else
    echo "No GPU detected. Running in CPU-only mode."
fi

# Set environment variables
export PYSPARK_DRIVER_MEMORY=10g
export PYSPARK_EXECUTOR_MEMORY=10g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp

# Define paths
scratch_data_dir=$(realpath ~/scratch/plantclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data
dataset_name=test_2024_subset20

# Run the Python script
plantclef masking workflow \
    $project_data_dir/parquet/$dataset_name \
    $project_data_dir/masking/${dataset_name}_v2 \
    --cpu-count 1 \
    --num-sample-ids 1 \
    --sample-id 0

# If GPU was used, parse the NVIDIA monitoring output
if [[ "$GPU_AVAILABLE" == "True" ]]; then
    python ~/clef/plantclef-2025/scripts/nvidia-logs.sh parse $NVIDIA_LOG_FILE
fi
