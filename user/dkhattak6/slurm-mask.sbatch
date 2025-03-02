#!/bin/bash
#SBATCH --job-name=mask2parquet   # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node
##SBATCH --gres=gpu:V100:1                       # Number of GPUs per node
#SBATCH --gres=gpu:1                            # Number of GPUs per node
#SBATCH -C RTX6000                              # GPU type
#SBATCH --cpus-per-task=6                       # Number of cores per task
#SBATCH --mem-per-gpu=64G                               # Memory per core
#SBATCH --time=2:00:00                         # Duration of the job (8 hours)
#SBATCH --qos=inferno                           # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=dkhattak6@gatech.edu  # E-mail address for notifications

#set -xe

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# Load and activate Python module
module load python/3.12.5
module load cuda/11.8.0
module load libxml2/2.10.3
echo "Launched CUDA"
nvcc --version
source ~/scratch/.venv/bin/activate

python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
export PYSPARK_DRIVER_MEMORY=32g
export PYSPARK_EXECUTOR_MEMORY=32g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp

# Run Python script
python ~/clef/plantclef-2025/plantclef/preprocessing/mask_to_parquet.py
