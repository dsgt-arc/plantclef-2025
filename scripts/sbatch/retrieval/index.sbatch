#!/bin/bash
#SBATCH --job-name=plantclef-index              # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --cpus-per-task=16                      # Number of cores per task
#SBATCH --mem-per-cpu=8G                        # Memory per core
#SBATCH --time=60                               # Duration of the job
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
dataset_name=train
id_map_name=train_id_map
index_name=embeddings

# run Python script
plantclef retrieval index workflow \
    $scratch_data_dir/embeddings/$dataset_name \
    $scratch_data_dir/faiss/$dataset_name/$id_map_name \
    $scratch_data_dir/faiss/$dataset_name/{$index_name}.index \
    embeddings \
    --cpu-count 16
