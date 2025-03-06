#!/bin/bash
set -xe

# print system info
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/scratch/plantclef/venv/bin/activate

# check GPU availability
python -c "import torch; print(torch.cuda.is_available())"  # Check if PyTorch can access the GPU
nvidia-smi                                                  # Check GPU usage

# start the NVIDIA monitoring job in the background
NVIDIA_LOG_FILE=Report-nvidia-logs.ndjson
python ~/clef/plantclef-2025/scripts/nvidia-logs.sh monitor "$NVIDIA_LOG_FILE" --interval 15 &
nvidia_logs_pid=$!
echo "Started NVIDIA monitoring process with PID ${nvidia_logs_pid}"

# set environment variables
export PYSPARK_DRIVER_MEMORY=10g
export PYSPARK_EXECUTOR_MEMORY=10g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp

# define paths
scratch_data_dir=$(realpath ~/scratch/plantclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data
dataset_name=test_2024_subset20_v2
grid_size=4
file_name=${dataset_name}_grid=${grid_size}x${grid_size}

# run the Python script
plantclef retrieval embed workflow \
    $project_data_dir/masking/$dataset_name \
    $project_data_dir/embeddings/$file_name \
    --cpu-count 1 \
    --batch-size 1 \
    --num-sample-ids 1 \
    --sample-id 0 \
    --grid-size $grid_size \

# parse the nvida monitoring output
python ~/clef/plantclef-2025/scripts/nvidia-logs.sh parse Report-${SLURM_JOB_ID}-nvidia-logs.ndjson
