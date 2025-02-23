#!/bin/bash
#SBATCH --job-name=SlurmImage2Parquet           # Job name
#SBATCH --account=paceship-dsgt_clef2025        # Charge account
#SBATCH --nodes=1                               # Number of nodes required
#SBATCH --ntasks-per-node=1                     # Number of parallel tasks per node
#SBATCH --cpus-per-task=4                       # Number of CPU cores per task
#SBATCH --mem-per-cpu=16G                       # Total memory per core for the job
#SBATCH --time=8:00:00                          # Duration of the job (8 hours)
#SBATCH --qos=inferno                           # QOS Name (inferno or embers)
#SBATCH -oReport-%j.log                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=murilogustineli@gatech.edu  # E-mail address for notifications

set -xe
export NO_REINSTALL=1

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/clef/plantclef-2025/scripts/activate.sh


# define variables
scratch_data_dir=$(realpath ~/scratch/plantclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data
dataset_name=test_2024

# Run Python script
plantclef preprocessing test_to_parquet \
    $project_data_dir/$dataset_name \
    $project_data_dir/parquet/$dataset_name \
    --cores 6 \
    --memory 16g
