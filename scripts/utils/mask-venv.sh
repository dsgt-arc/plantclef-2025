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

# upgrade pip
pip install --upgrade pip

# Grounded-Segment-Anything repo
echo "Installing required packages for GroundingDINO and SAM..."
echo "Installing GroundedSAM..."
cd ~/scratch/plantclef
if [ ! -d "Grounded-Segment-Anything" ]; then
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
fi

# install Grounded-Segment-Anything
echo "Installing Grounded-Segment-Anything..."
cd ~/scratch/plantclef/Grounded-Segment-Anything
pip install -q -r requirements.txt

# install GroundingDINO and segment_anything
echo "Installing GroundingDINO and segment_anything..."
cd ~/scratch/plantclef/Grounded-Segment-Anything/GroundingDINO
pip install -q .

# install segment_anything
cd ~/scratch/plantclef/Grounded-Segment-Anything/segment_anything
pip install -q .

# Install dependencies unless NO_REINSTALL is set
if [[ -z ${NO_REINSTALL:-} ]]; then
    echo "Installing required packages for GroundingDINO and SAM..."
    pip install --upgrade pip
    # Look for requirements.txt in the project root
    if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    else
        echo "Warning: $PROJECT_ROOT/requirements.txt not found."
    fi
    # Install the project in editable mode from the project root (where pyproject.toml resides)
    pip install -e "$PROJECT_ROOT"
fi

popd > /dev/null

echo "Installation complete."
