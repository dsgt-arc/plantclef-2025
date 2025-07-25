#!/bin/bash
#SBATCH --job-name=plantclef-embed              # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH --nodes=1                               # Number of nodes
##SBATCH --gres=gpu:V100:1                       # Number of GPUs per node
#SBATCH --gres=gpu:1                            # Number of GPUs per node
#SBATCH -C RTX6000                              # GPU type
#SBATCH --cpus-per-task=6                       # Number of cores per task
#SBATCH --mem-per-gpu=64G                               # Memory per core
#SBATCH --time=10:00:00                         # Duration of the job (8 hours)
#SBATCH --qos=inferno                           # QOS Name
#SBATCH --output=Report-%j.log                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=murilogustineli@gatech.edu  # E-mail address for notifications

# for parallel runs, break into 10 jobs 10 at a time e.g.
##SBATCH --array=0-1%0

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

# set environment variables
export PYSPARK_DRIVER_MEMORY=10g
export PYSPARK_EXECUTOR_MEMORY=10g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp

# define paths
scratch_data_dir=$(realpath ~/scratch/plantclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef
dataset_name=crop_resize_128_subset_top5_train

# run workflow
plantclef embedding test_workflow \
    $scratch_data_dir/parquet/$dataset_name \
    $scratch_data_dir/embeddings/${SLURM_JOB_ID}/$dataset_name \
    --sample-id 0
