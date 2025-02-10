#!/bin/bash
#SBATCH --job-name=plantclef-dinov2-embedding   # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH -N1 --gres=gpu:V100:1                   # Number of nodes and cores required
#SBATCH --cpus-per-task=1                       # Number of cores per task
#SBATCH --mem-per-gpu=32G                       # Memory per core
#SBATCH --time=8:00:00                          # Duration of the job (8 hours)
#SBATCH --qos=inferno                           # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=murilogustineli@gatech.edu  # E-mail address for notifications
set -xe

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# Directory setup
SCRATCH_DIR=~/scratch/plantclef
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR
# Python environment setup
python3 -m venv venv                          # Create virtual environment
source venv/bin/activate                      # Activate the environment
pip install --upgrade pip                     # Upgrade pip
pip install -e ~/clef/plantclef-2025          # Installs package

# Load and activate Python module
module load python/3.12.5                                   # Load Python module
source ~/scratch/plantclef/venv/bin/activate                # Activate the environment
python -c "import torch; print(torch.cuda.is_available())"  # Check if PyTorch can access the GPU
nvidia-smi                                                  # Check GPU usage

# Run Pytest script
pytest -vv -s ~/clef/plantclef-2025/tests/test_gpu_transform.py
