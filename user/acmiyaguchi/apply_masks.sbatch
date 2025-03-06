#!/bin/bash
#SBATCH --job-name=plantclef-mask --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --cpus-per-task=16 --mem-per-cpu=4G
#SBATCH --time=3:00:00 --qos=inferno
#SBATCH --output=Report-%j.log --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/clef/plantclef-2025/scripts/activate.sh

set -xe
export PYSPARK_EXECUTOR_CORES=16
export PYSPARK_DRIVER_MEMORY=60g
export SPARK_LOCAL_DIR=$TMPDIR/spark
python apply_masks.py
