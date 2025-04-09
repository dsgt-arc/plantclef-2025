#!/bin/bash
set -xe

# print system info
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/scratch/plantclef/venv/bin/activate

# naive baseline approach
top_k=10

# run the Python script
plantclef classification naive_baseline \
    --top-k $top_k
