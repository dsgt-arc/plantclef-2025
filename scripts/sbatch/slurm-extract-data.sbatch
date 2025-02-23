#!/bin/bash
#SBATCH --job-name=extract-data --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t20 -qinferno -oReport-%j.out
set -e

# User specific aliases and functions
export PATH=/storage/coda1/p-dsgt_clef2025/0/shared/bin:$PATH

# activate the environment
source ~/clef/plantclef-2025/scripts/activate.sh

# extract data
cd ~/clef/plantclef-2025/scripts
extract-data.sh
