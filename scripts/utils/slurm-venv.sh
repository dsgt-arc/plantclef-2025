#!/usr/bin/env bash
# usage: source scripts/slurm-venv.sh [venv_root]

# Determine the directory of this script.
# Assuming this script is located at: PROJECT_ROOT/scripts/utils/slurm-venv.sh
# The project root is two levels up (from scripts/utils to scripts, then to project root)
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PROJECT_ROOT="$(realpath "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Optional argument to specify the virtual environment root (default: ~/scratch/plantclef)
VENV_PARENT_ROOT="${1:-$HOME/scratch/plantclef}"
VENV_PARENT_ROOT="$(realpath "$VENV_PARENT_ROOT")"

# Load the required Python module
# Set CPATH to include Python’s include directory
module load python/3.10
PYTHON_ROOT=$(python -c 'import sys; print(sys.base_prefix)')
export CPATH="${PYTHON_ROOT}/include/python3.10:${CPATH:-}"

# Create the virtual environment directory if it doesn't exist
mkdir -p "$VENV_PARENT_ROOT"
pushd "$VENV_PARENT_ROOT" > /dev/null

# Add local/bin to PATH
export PATH=$HOME/.local/bin:$PATH

# Create and activate the virtual environment
if ! command -v uv &> /dev/null; then
    python -m ensurepip
    pip install --upgrade pip uv
fi
if [[ ! -d venv ]]; then
    echo "Creating virtual environment in ${VENV_PARENT_ROOT}/venv ..."
    uv venv venv
fi
source venv/bin/activate

# Install dependencies unless NO_REINSTALL is set
if [[ -z ${NO_REINSTALL:-}  ]]; then
    uv pip install -r "$PROJECT_ROOT/requirements.txt"
    uv pip install -e "$PROJECT_ROOT";
fi

# Verify the environment setup
echo "Python Path: $(which python)"
echo "Python Version: $(python --version)"
echo "Pip Path: $(which pip)"
echo "Pip Version: $(pip --version)"

popd > /dev/null
