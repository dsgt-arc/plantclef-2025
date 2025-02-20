#!/bin/bash
#SBATCH --job-name=rsync --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t20 -qinferno -oReport-%j.out
set -e

# User specific aliases and functions
export PATH=/storage/coda1/p-dsgt_clef2025/0/shared/bin:$PATH

# Define the project and scratch directories
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef/data/embeddings
scratch_dir=$(realpath ~/scratch/plantclef/data/embeddings)

# Remove any trailing slash from the prefix argument
prefix=${1%/}

# Exit if the source (scratch) directory does not exist
if [ ! -d $scratch_dir/$prefix ]; then
    echo "Directory $scratch_dir/$prefix does not exist"
    exit 1
fi

# Now copy directories from scratch into project
rclone sync -v --progress --transfers 12 $scratch_dir/$prefix/ $project_dir/$prefix
