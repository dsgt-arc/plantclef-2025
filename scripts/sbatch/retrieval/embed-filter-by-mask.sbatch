#!/bin/bash
#SBATCH --job-name=plantclef-embed-filter --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --cpus-per-task=16 --mem-per-cpu=4G
#SBATCH --time=2:00:00 --qos=embers
#SBATCH --output=Report-%j.log --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu
#SBATCH --array=2-6%10

echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
export NO_REINSTALL=1
source ~/clef/plantclef-2025/scripts/activate.sh

set -xe

# 24 * 4G = 96G
export PYSPARK_EXECUTOR_CORES=16
export PYSPARK_DRIVER_MEMORY=60g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data
plantclef retrieval embed filter-by-mask \
    $project_data_dir/embeddings/test/test_2024 \
    $project_data_dir/masking/test_2024_v2/data \
    $project_data_dir/embeddings/test/test_2024_with_mask_coverage \
    --grid-size ${SLURM_ARRAY_TASK_ID} \
