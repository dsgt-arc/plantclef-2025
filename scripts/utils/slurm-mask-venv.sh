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
module load cuda/11.8.0

# Set CPATH to include Python’s include directory
PYTHON_ROOT=$(python -c 'import sys; print(sys.base_prefix)')
export CPATH="${PYTHON_ROOT}/include/python3.10:${CPATH:-}"

# Ensure pip is installed and upgrade pip
python -m ensurepip --default-pip
python -m pip install --upgrade pip

# Create the virtual environment directory if it doesn't exist
mkdir -p "$VENV_PARENT_ROOT"
pushd "$VENV_PARENT_ROOT" > /dev/null

# Create and activate the virtual environment
echo "Creating virtual environment in ${VENV_PARENT_ROOT}/mask-venv..."
python -m venv mask-venv
source mask-venv/bin/activate

# Verify the environment setup
echo "Python Path: $(which python)"
echo "Python Version: $(python --version)"
echo "Pip Path: $(which pip)"
echo "Pip Version: $(pip --version)"
echo "CUDA Version: $(nvcc --version)"

# Install dependencies unless NO_REINSTALL is set
if [[ -z ${NO_REINSTALL:-} ]]; then
    echo "Installing required packages for GroundingDINO and SAM..."
    pip install --upgrade pip
    # Look for requirements.txt in the project root
    if [[ -f "$PROJECT_ROOT/requirements-sam.txt" ]]; then
        pip install -r "$PROJECT_ROOT/requirements-sam.txt"
    else
        echo "Warning: $PROJECT_ROOT/requirements-sam.txt not found."
    fi
    # Install the project in editable mode from the project root (where pyproject.toml resides)
    pip install -e "$PROJECT_ROOT"
fi

# install PyTorch manually
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install GroundedSAM
# https://github.com/IDEA-Research/Grounded-Segment-Anything?tab=readme-ov-file#install-without-docker
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
# Find CUDA directory dynamically
CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_HOME=$CUDA_HOME
echo "CUDA_HOME set to: $CUDA_HOME"

# install SAM
echo "Installing GroundedSAM..."
cd ~/scratch/plantclef
if [ ! -d "Grounded-Segment-Anything" ]; then
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
fi
cd Grounded-Segment-Anything

# install SAM
echo "Installing SAM..."
python -m pip install -e segment_anything

# install GroundingDINO
echo "Installing GroundingDINO..."
pip install wheel
pip install ninja
pip install --no-build-isolation -e GroundingDINO
pip install --upgrade diffusers[torch]

# install osx
echo "Installing Grounded-SAM-OSX..."
git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh
cd ..

# install RAM & Tag2Text
if [ ! -d "recognize-anything" ]; then
    git clone https://github.com/xinyu1205/recognize-anything.git
fi
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/

popd > /dev/null

echo "Installation complete."
