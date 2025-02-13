#!/bin/bash
#SBATCH --job-name=plantclef-dinov2-embedding   # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node
#SBATCH --gres=gpu:V100:1                       # Number of GPUs per node
#SBATCH --cpus-per-task=12                      # Number of cores per task
#SBATCH --mem-per-gpu=192G                      # Memory per core
#SBATCH --time=10:00:00                         # Duration of the job (8 hours)
#SBATCH --qos=inferno                           # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=murilogustineli@gatech.edu  # E-mail address for notifications

# for parallel runs, break into 10 jobs 10 at a time e.g.
##SBATCH --array=0-9%10

set -xe
# Activate the environment
source ~/clef/plantclef-2025/scripts/activate

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"  # Check if PyTorch can access the GPU
nvidia-smi                                                  # Check GPU usage

# start the nvidia monitoring job in the background using slurm job id
NVIDIA_LOG_FILE=Report-${SLURM_JOB_ID}-nvidia-logs.ndjson
nvidia-logs monitor $NVIDIA_LOG_FILE &
nvidia_logs_pid=$!

# Set environment variables
export PYSPARK_DRIVER_MEMORY=160g
export PYSPARK_EXECUTOR_MEMORY=160g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp


# Run Python script
python ~/clef/plantclef-2025/plantclef/embedding/workflow.py
