#!/bin/bash
#SBATCH --job-name=SlurmImage2Parquet           # Job name
#SBATCH --account=paceship-dsgt_clef2025        # Charge account
#SBATCH --nodes=1                               # Number of nodes required
#SBATCH --ntasks-per-node=1                     # Number of parallel tasks per node
#SBATCH --cpus-per-task=6                       # Number of CPU cores per task
#SBATCH --mem-per-cpu=16G                       # Total memory per core for the job
#SBATCH --time=8:00:00                          # Duration of the job (8 hours)
#SBATCH --qos=inferno                           # QOS Name (inferno or embers)
#SBATCH -oReport-%j.log                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=murilogustineli@gatech.edu  # E-mail address for notifications
set -xe

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# Activate the environment
source ~/clef/plantclef-2025/scripts/activate.sh

# Set environment variables
export PYSPARK_DRIVER_MEMORY=20g
export PYSPARK_EXECUTOR_MEMORY=20g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp

# Run Python script
python ~/clef/plantclef-2025/plantclef/create_top_species_subset.py --top-n 10
