#!/bin/bash
#SBATCH --job-name=plantclef-mask --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --gres=gpu:1 -C RTX6000 --cpus-per-task=6 --mem-per-gpu=64G
#SBATCH --time=6:00:00 --qos=inferno
#SBATCH --output=Report-%j.log --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

set -xe
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/clef/plantclef-2025/scripts/activate.sh
nvidia-smi

# start the nvidia monitoring job in the background using SLURM job id
NVIDIA_LOG_FILE=Report-${SLURM_JOB_ID}-nvidia-logs.ndjson
python ~/clef/plantclef-2025/scripts/nvidia-logs.sh monitor "$NVIDIA_LOG_FILE" --interval 15 &
nvidia_logs_pid=$!
echo "Started NVIDIA monitoring process with PID ${nvidia_logs_pid}"

scratch_data_dir=$(realpath ~/scratch/plantclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data
dataset_name=test_2024_dev_anthony
plantclef masking workflow \
    $project_data_dir/parquet/$dataset_name \
    $scratch_data_dir/masking/${dataset_name} \
    --cpu-count 2 \
    --num-sample-ids 20 \
    --num-partitions 5

# parse the nvida monitoring output
python ~/clef/plantclef-2025/scripts/nvidia-logs.sh parse Report-${SLURM_JOB_ID}-nvidia-logs.ndjson
