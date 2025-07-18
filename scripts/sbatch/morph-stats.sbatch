#!/bin/bash
#SBATCH --job-name=plantclef-morph-stats --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --cpus-per-task=24 --mem-per-cpu=4G
#SBATCH --time=2:00:00 --qos=embers
#SBATCH --output=Report-%j.log --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu
#SBATCH --array=0-19%10

echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
export NO_REINSTALL=1
source ~/clef/plantclef-2025/scripts/activate.sh

set -xe

# 24 * 4G = 96G
export PYSPARK_EXECUTOR_CORES=24
export PYSPARK_DRIVER_MEMORY=92g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data
plantclef morph mask-stats \
    $project_data_dir/masking/test_2024_v2/data \
    $project_data_dir/morph/stats/test_2024_v2 \
    --num-sample-ids 20 \
    --sample-id ${SLURM_ARRAY_TASK_ID:-0}
