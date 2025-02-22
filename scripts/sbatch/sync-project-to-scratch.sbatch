#!/bin/bash
#SBATCH --job-name=rsync --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t20 -qinferno -oReport-%j.out
set -e

# User specific aliases and functions
export PATH=/storage/coda1/p-dsgt_clef2025/0/shared/bin:$PATH

project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/plantclef
scratch_dir=$(realpath ~/scratch/plantclef)
# trim trailing slash
prefix=${1%/}

# exit if prefix doesn't exist
if [ ! -d $project_dir/$prefix ]; then
    echo "Directory $project_dir/$prefix does not exist"
    exit 1
fi

# now copy directories from project into scratch
rclone sync -v --progress --transfers 12 $project_dir/$prefix/ $scratch_dir/$prefix
