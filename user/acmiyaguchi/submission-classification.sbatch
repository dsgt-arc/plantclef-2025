#!/bin/bash
#SBATCH --job-name=plantclef-submission --account=paceship-dsgt_clef2025 --qos=embers
#SBATCH --nodes=1 --gres=gpu:V100:1 --cpus-per-task=6 --mem-per-gpu=64G
#SBATCH --time=120 --output=Report-%j.log --mail-type=END,FAIL
#SBATCH --mail-user=acmiyaguchi@gatech.edu

# print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/clef/plantclef-2025/scripts/activate.sh
set -xe

# check GPU availability
python -c "import torch; print(torch.cuda.is_available())"  # Check if PyTorch can access the GPU
nvidia-smi                                                  # Check GPU usage

# start the nvidia monitoring job in the background using SLURM job id
NVIDIA_LOG_FILE=Report-${SLURM_JOB_ID}-nvidia-logs.ndjson
python ~/clef/plantclef-2025/scripts/nvidia-logs.sh monitor "$NVIDIA_LOG_FILE" --interval 15 &
nvidia_logs_pid=$!
echo "Started NVIDIA monitoring process with PID ${nvidia_logs_pid}"


project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data
submission_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/submissions
# dataset_name=test_2024_with_masks
dataset_name=test_2024_with_masks_mean_fill

export PYSPARK_DRIVER_MEMORY=40g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp
plantclef classification workflow \
    $project_data_dir/parquet/$dataset_name \
    $project_data_dir/logits/${dataset_name} \
    $submission_dir \
    --cpu-count 6 \
    --batch-size 1 \
    --grid-size ${GRID_SIZE:-6} \
    --top-k-proba ${TOPK:-5} \
    --num-sample-ids ${NUM_SAMPLE_IDS:-1}

# parse the nvida monitoring output
python ~/clef/plantclef-2025/scripts/nvidia-logs.sh parse Report-${SLURM_JOB_ID}-nvidia-logs.ndjson
