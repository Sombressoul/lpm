#!/usr/bin/env bash

# Defines
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CONDA_ENV_NAME="lpm-python-3.11"
VENV_ENV_NAME="lpm"
PYTHON_VERSION="3.11"

# Check current user and retrieve sudo access
if [ "$EUID" = 0 ]; then
    echo "Don't run this script as root."
    exit
else
    echo "Enter sudo pass (it is necessary for internal operations)."
    sudo whoami >>/dev/null
fi

# Check conda availability.
if ! command -v conda &>/dev/null; then
    echo "conda could not be found. Please install conda first."
    exit
fi

# Install sys deps.
sudo apt update
sudo apt install -y python3-venv

# Init conda.
source $($HOME/anaconda3/bin/conda info --base)/etc/profile.d/conda.sh &>/dev/null
source $(conda info --base)/etc/profile.d/conda.sh &>/dev/null

# Create temporaty conda environment.
conda create -y -n $CONDA_ENV_NAME -c conda-forge -c main python=$PYTHON_VERSION pip
conda activate $CONDA_ENV_NAME

if [ $CONDA_PROMPT_MODIFIER != "($CONDA_ENV_NAME)" ]; then
    echo "Cannot activate conda environment: $CONDA_ENV_NAME"
    echo "Active modifier: $CONDA_PROMPT_MODIFIER"
    exit
fi

# Create virtual environment with target python version.
python -m venv --clear --prompt $VENV_ENV_NAME $PROJECT_DIR/.venv
ln -s $PROJECT_DIR/.venv/bin/activate $PROJECT_DIR/.activate &>/dev/null
conda deactivate

# Deactivate all conda environments if there are any (just in case).
while [ "$CONDA_PROMPT_MODIFIER" ]; do
    echo "Deactivating conda environment: $CONDA_PROMPT_MODIFIER"
    conda deactivate
done

# Activate venv.
source .activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Install dev dependencies.
python -m pip install -r requirements_dev.txt

# Install deps from GIT.
mkdir -p $PROJECT_DIR/repos/TransformerEngine
cd $PROJECT_DIR/repos/TransformerEngine
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git .
git submodule update --init --recursive
# Building TransformerEngine from source requires ninja and wheel to be installed.
python -m pip install "ninja==1.11.1.1" "wheel==0.42.0"
export NVTE_FRAMEWORK="pytorch"
export MAX_JOBS=1
python -m pip install --no-build-isolation .
cd $PROJECT_DIR

# Deactivate venv.
deactivate
