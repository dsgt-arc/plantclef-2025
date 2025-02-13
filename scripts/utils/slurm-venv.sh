#!/usr/bin/env bash
# usage: source scripts/slurm-venv.sh [venv_root]
SCRIPT_PARENT_ROOT=$(
    dirname ${BASH_SOURCE[0]} \
    | dirname $(cat -) \
    | realpath $(cat -)
)

# choose the module depending if this being run on slurm or not
MODULE_PATH=$(dirname $SCRIPT_PARENT_ROOT)
# MODULE_PATH=${SLURM_SUBMIT_DIR:-$MODULE_PATH}
# optional argument to specify the venv root
VENV_PARENT_ROOT=${1:-~/scratch/plantclef}
VENV_PARENT_ROOT=$(realpath $VENV_PARENT_ROOT)

# use an updated version of python and set the include path for wheels
module load python/3.10
export CPATH=$PYTHON_ROOT/include/python3.10:$CPATH

# create a virtual environment with uv
python -m pip install --upgrade pip uv
mkdir -p $VENV_PARENT_ROOT
pushd $VENV_PARENT_ROOT
uv venv
source .venv/bin/activate

# check for NO_REINSTALL flag
if [[ -z ${NO_REINSTALL:-} ]]; then
    uv pip install -r $MODULE_PATH/requirements.txt
    uv pip install -e $MODULE_PATH
fi
popd
