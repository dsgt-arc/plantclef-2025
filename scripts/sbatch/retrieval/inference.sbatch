#!/bin/bash
#SBATCH --job-name=plantclef-inference          # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --cpus-per-task=16                      # Number of cores per task
#SBATCH --mem-per-cpu=8G                        # Memory per core
#SBATCH --time=60                               # Duration of the job (minutes)
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
scratch_plantclef_dir=$(realpath ~/scratch/plantclef)
project_plantclef_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef
scratch_data_dir="${scratch_plantclef_dir}/data"
project_data_dir="${project_plantclef_dir}/data"
scratch_submission_dir="${scratch_plantclef_dir}/submissions/knn"
project_submission_dir="${project_plantclef_dir}/submissions/knn"
dataset_name=test_2024

grid_sizes=(3 4 6)
top_k=5
threshold_percentile=0.05
threshold_mode="per_image"

knn_path="${scratch_data_dir}/knn/${dataset_name}/k=50"
id_map_path="${scratch_data_dir}/faiss/train/train_id_map"
grid_sizes_args=$(printf "%s " "${grid_sizes[@]}")

plantclef retrieval inference workflow \
    $knn_path \
    $scratch_submission_dir \
    $id_map_path \
    $grid_sizes_args \
    --top-k $top_k \
    --threshold-percentile $threshold_percentile \
    --threshold-mode $threshold_mode \
    --cpu-count 16
