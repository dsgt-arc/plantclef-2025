set -xe

# print job info
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
dataset_name=test_2025
file_name=${dataset_name}_grid=${grid_size}x${grid_size}

# run the Python script
plantclef embedding workflow \
    $project_data_dir/parquet/$dataset_name \
    $project_data_dir/embeddings/$dataset_name/${dataset_name}_embed_logits \
    --cpu-count 4 \
    --batch-size 1 \
    --sample-id 0 \
    --num-sample-ids 1 \
    --use-test-data \
