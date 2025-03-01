#!/bin/bash
#SBATCH --job-name=plantclef-embed              # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --gres=gpu:V100:1                       # Number of GPUs per node
##SBATCH --gres=gpu:1                            # GPU resource
##SBATCH -C RTX6000                              # GPU type
#SBATCH --cpus-per-task=6                       # Number of cores per task
#SBATCH --mem-per-gpu=64G                       # Memory per core
#SBATCH --time=120                              # Duration of the job (1h20min)
#SBATCH --qos=embers                            # QOS Name
#SBATCH --output=Report-%j.log                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=murilogustineli@gatech.edu  # E-mail address for notifications

# for parallel runs, break into 5 jobs 5 at a time e.g.
##SBATCH --array=0-4%5

set -xe
export NO_REINSTALL=1

# print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/clef/plantclef-2025/scripts/activate.sh

# check GPU availability
python -c "import torch; print(torch.cuda.is_available())"  # Check if PyTorch can access the GPU
nvidia-smi                                                  # Check GPU usage

# start the nvidia monitoring job in the background using SLURM job id
NVIDIA_LOG_FILE=Report-${SLURM_JOB_ID}-nvidia-logs.ndjson
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
file_name="${dataset_name}_grid=${grid_size}x${grid_size}"

# run the Python script
plantclef retrieval embed workflow \
    $project_data_dir/masking/$dataset_name \
    $project_data_dir/embeddings/$file_name \
    --cpu-count 4 \
    --batch-size 1 \
    --num-sample-ids 1 \
    --sample-id 0 \
    --grid-size $grid_size \

# parse the nvida monitoring output
python ~/clef/plantclef-2025/scripts/nvidia-logs.sh parse Report-${SLURM_JOB_ID}-nvidia-logs.ndjson
