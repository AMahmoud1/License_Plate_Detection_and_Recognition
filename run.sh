#!/bin/bash

# Name of the Conda virtual environment
VENV_NAME="license_plate_venv"

# Check if the Conda environment exists
env_exists=$(conda env list | grep "^$VENV_NAME")

if [ -z "$env_exists" ]; then
    echo "Conda environment '$VENV_NAME' does not exist. Creating..."
    conda create -y -n $VENV_NAME python=3.9
else
    echo "Conda environment '$VENV_NAME' already exists."
fi

# Activate the Conda environment
source activate $VENV_NAME || conda activate $VENV_NAME

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt
echo "All packages installed."

