#!/usr/bin/env bash
#
# Activates the environment. Ensure this is run before
# running any other scripts in this directory.
#
# Run with:
#    source scripts/activate
#

SCRIPT_PARENT_ROOT=$(
    dirname ${BASH_SOURCE[0]} | realpath $(cat -)
)

# adds the scripts directory to the path
PATH="$PATH:$SCRIPT_PARENT_ROOT"
# for large models and datasets from huggingface/transformers
PYTEST_DEBUG_TEMPROOT=~/scratch/pytest-tmp
mkdir -p $PYTEST_DEBUG_TEMPROOT

# NOTE: uv cache is set by pyproject.toml
source $SCRIPT_PARENT_ROOT/utils/slurm-venv.sh

# exports
export PATH
export PYTEST_DEBUG_TEMPROOT
