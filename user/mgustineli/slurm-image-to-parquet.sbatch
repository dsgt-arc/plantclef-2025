#!/bin/bash
#SBATCH --job-name=SlurmImage2Parquet           # Job name
#SBATCH --account=paceship-dsgt_clef2025        # Charge account
#SBATCH --nodes=1                               # Number of nodes required
#SBATCH --ntasks-per-node=1                     # Number of parallel tasks per node
#SBATCH --cpus-per-task=4                       # Number of CPU cores per task
#SBATCH --mem-per-cpu=16G                       # Total memory per core for the job
#SBATCH --time=1:00:00                          # Duration of the job (1 hour)
#SBATCH --qos=inferno                           # QOS Name (inferno or embers)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=murilogustineli@gatech.edu  # E-mail address for notifications
set -xe

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# Uncomment the following lines if you're running this script for the first time
# # Directory setup
# SCRATCH_DIR=~/scratch/plantclef
# mkdir -p $SCRATCH_DIR
# cd $SCRATCH_DIR
# # Python environment setup
# python3 -m venv venv                          # Create virtual environment
# source venv/bin/activate                      # Activate the environment
# pip install --upgrade pip                     # Upgrade pip
# pip install -e ~/clef/plantclef-2025          # Installs package

# Load and activate Python module
module load python/3.12.5                       # Load Python module
source ~/scratch/plantclef/venv/bin/activate    # Activate the environment

# Run Python script
python ~/clef/plantclef-2025/plantclef/image_to_parquet.py
