#!/bin/bash
set -xe
export NO_REINSTALL=1

# print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/scratch/plantclef/.venv/bin/activate

# check GPU availability
python -c "import torch; print(torch.cuda.is_available())"  # Check if PyTorch can access the GPU
nvidia-smi                                                  # Check GPU usage

# set environment variables
export PYSPARK_DRIVER_MEMORY=10g
export PYSPARK_EXECUTOR_MEMORY=10g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp

# define paths
scratch_data_dir=$(realpath ~/scratch/plantclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data
submission_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/submissions
dataset_name=test_2025
folder_name=${dataset_name}_detection_v1
prior_dir=test_2025_image_prior_probabilities

# run Python script
# give a --prior-path to use the prior probabilities to reweight the logits
plantclef classification workflow \
    $project_data_dir/detection/$dataset_name/$folder_name \
    $project_data_dir/logits/${folder_name}_detection \
    $submission_dir \
    $dataset_name \
    --cpu-count 6 \
    --batch-size 1 \
    --sample-id 0 \
    --num-sample-ids 1 \
    --top-k-proba 10 \
    --use-detections \
    # --prior-path $project_data_dir/prior/$prior_dir \
