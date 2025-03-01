#!/bin/bash
#SBATCH --job-name=plantclef-query              # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --cpus-per-task=16                       # Number of cores per task
#SBATCH --mem-per-cpu=32G                        # Memory per core
#SBATCH --time=120                               # Duration of the job
#SBATCH --qos=embers                            # QOS Name
#SBATCH --output=Report-%j.log                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=acheung@gatech.edu          # E-mail address for notifications

set -xe
export NO_REINSTALL=1

# print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/clef/plantclef-2025/scripts/activate.sh

# set environment variables
export PYSPARK_DRIVER_MEMORY=10g
export PYSPARK_EXECUTOR_MEMORY=10g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp

# define paths
scratch_data_dir=$(realpath ~/scratch/plantclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data
dataset_name=test_2024
grid_size=3
k=50

k_folder="k=${k}"
grid_folder="grid=${grid_size}x${grid_size}"

# run Python script
plantclef retrieval query workflow \
    $scratch_data_dir/embeddings/test/$dataset_name/$grid_folder \
    $scratch_data_dir/knn/$dataset_name/$k_folder/$grid_folder \
    --k $k \
    --cpu-count 16
