#!/bin/bash
#SBATCH --job-name=plantclef-dinov2-embedding   # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node
#SBATCH --gres=gpu:1                            # GPU resource
#SBATCH -C RTX6000                              # GPU type
#SBATCH --cpus-per-task=1                       # Number of cores per task
#SBATCH --mem-per-gpu=8G                        # Memory per core
#SBATCH --time=2:00:00                          # Duration of the job (2 hours)
#SBATCH --qos=inferno                           # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=murilogustineli@gatech.edu  # E-mail address for notifications
set -xe

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

# Run Pytest script
pytest -vv -s ~/clef/plantclef-2025/tests/test_gpu_workflow.py
