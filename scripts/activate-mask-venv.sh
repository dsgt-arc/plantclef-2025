#!/usr/bin/env bash
#
# Activates the environment. Ensure this is run before
# running any other scripts in this directory.
#
# Run with:
#    source scripts/activate.sh

# Determine the directory of this script
SCRIPT_PARENT_ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# Add the scripts directory to the PATH
export PATH="$PATH:$SCRIPT_PARENT_ROOT"

# Set the temporary root for pytest (using $HOME instead of ~ for better portability)
PYTEST_DEBUG_TEMPROOT="$HOME/scratch/pytest-tmp"
mkdir -p "$PYTEST_DEBUG_TEMPROOT"
export PYTEST_DEBUG_TEMPROOT

XDG_CACHE_HOME="$HOME/scratch/.cache"
export XDG_CACHE_HOME

# Source the SLURM virtual environment setup script
source "$SCRIPT_PARENT_ROOT/utils/slurm-mask-venv.sh"
