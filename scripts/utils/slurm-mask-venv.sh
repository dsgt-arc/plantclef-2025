#!/usr/bin/env bash
# usage: source scripts/slurm-venv.sh [venv_root]

# Determine the directory of this script.
# Assuming this script is located at: PROJECT_ROOT/scripts/utils/slurm-venv.sh
# The project root is two levels up (from scripts/utils to scripts, then to project root)
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PROJECT_ROOT="$(realpath "$(dirname "$(dirname "$SCRIPT_DIR")")")"
echo "Project root: $PROJECT_ROOT"

# Optional argument to specify the virtual environment root (default: ~/scratch/plantclef)
VENV_PARENT_ROOT="${1:-$HOME/scratch/plantclef}"
VENV_PARENT_ROOT="$(realpath "$VENV_PARENT_ROOT")"

# Load the required Python and CUDA modules
# Set CPATH to include Python’s include directory
module load cuda/11.8
module load python/3.11

echo "Creating virtual environment in ${VENV_PARENT_ROOT}/mask-venv..."
mkdir -p "$VENV_PARENT_ROOT" && pushd "$VENV_PARENT_ROOT" > /dev/null
if ! command -v uv &> /dev/null; then
    python -m ensurepip
    pip install --upgrade pip uv
fi
uv venv mask-venv
source mask-venv/bin/activate
uv pip install --upgrade pip wheel ninja

# use the includes from the virtualenv
export PYTHON_ROOT=$(python -c 'import sys; print(sys.base_prefix)')
export CPATH="${PYTHON_ROOT}/include/python3.11:${CPATH:-}"
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

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

python <<EOF
import torch
print("checking running torch version")
print(torch.cuda.is_available())
print(torch.version.cuda)
EOF

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"

# Install GroundedSAM
# https://github.com/IDEA-Research/Grounded-Segment-Anything?tab=readme-ov-file#install-without-docker
cd ~/scratch/plantclef
if [ ! -d "Grounded-Segment-Anything" ]; then
    git clone --recurse-submodules https://github.com/IDEA-Research/Grounded-Segment-Anything.git
fi
pushd Grounded-Segment-Anything/GroundingDINO
python setup.py install
popd
uv pip install \
    ./Grounded-Segment-Anything/segment_anything \
    --extra-index-url https://download.pytorch.org/whl/cu118

# install osx
# echo "Installing Grounded-SAM-OSX..."
# git submodule update --init --recursive
# cd grounded-sam-osx && bash install.sh
# cd ..

# install recognize-anything
uv pip install git+https://github.com/xinyu1205/recognize-anything.git

popd > /dev/null

echo "Installation complete."
