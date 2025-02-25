#!/usr/bin/env bash
# usage: source scripts/slurm-venv.sh [venv_root]

set -euo pipefail

# Determine the directory of this script.
# Assuming this script is located at: PROJECT_ROOT/scripts/utils/slurm-venv.sh
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
# The project root is two levels up (from scripts/utils to scripts, then to project root)
PROJECT_ROOT="$(realpath "$(dirname "$(dirname "$SCRIPT_DIR")")")"
echo "Project root: $PROJECT_ROOT"

# Optional argument to specify the virtual environment root (default: ~/scratch/plantclef)
VENV_PARENT_ROOT="${1:-$HOME/scratch/plantclef}"
VENV_PARENT_ROOT="$(realpath "$VENV_PARENT_ROOT")"

# Load the required Python and CUDA modules
module load python/3.10
module load cuda/11.8
# Set CPATH to include Python’s include directory
export PYTHON_ROOT=$(python -c 'import sys; print(sys.base_prefix)')
export CPATH="${PYTHON_ROOT}/include/python3.10:${CPATH:-}"

# Ensure pip is installed and upgrade pip
python -m ensurepip
python -m pip install --upgrade pip uv

# Create the virtual environment directory if it doesn't exist
mkdir -p "$VENV_PARENT_ROOT"
pushd "$VENV_PARENT_ROOT" > /dev/null

# Create and activate the virtual environment
echo "Creating virtual environment in ${VENV_PARENT_ROOT}/mask-venv..."
uv venv mask-venv
source mask-venv/bin/activate
uv pip install --upgrade pip wheel ninja

# Verify the environment setup
echo "Python Path: $(which python)"
echo "Python Version: $(python --version)"
echo "Pip Path: $(which pip)"
echo "Pip Version: $(pip --version)"
echo "CUDA Version: $(nvcc --version)"

# Install dependencies unless NO_REINSTALL is set
if [[ -z ${NO_REINSTALL:-} ]]; then
    echo "Installing required packages for GroundingDINO and SAM..."
    uv pip install -r "$PROJECT_ROOT/requirements-sam.txt"
    uv pip install -e "$PROJECT_ROOT"
fi

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
# Find CUDA directory dynamically
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "CUDA_HOME set to: $CUDA_HOME"

# Install GroundedSAM
# https://github.com/IDEA-Research/Grounded-Segment-Anything?tab=readme-ov-file#install-without-docker
cd ~/scratch/plantclef
if [ ! -d "Grounded-Segment-Anything" ]; then
    git clone --recurse-submodules https://github.com/IDEA-Research/Grounded-Segment-Anything.git
fi
uv pip install \
    ./Grounded-Segment-Anything/GroundingDINO \
    ./Grounded-Segment-Anything/segment_anything

# install osx
# echo "Installing Grounded-SAM-OSX..."
# git submodule update --init --recursive
# cd grounded-sam-osx && bash install.sh
# cd ..

# install recognize-anything
uv pip install git+https://github.com/xinyu1205/recognize-anything.git

popd > /dev/null

echo "Installation complete."
