#!/bin/bash

set -e

VENV_DIR="it2_idsp"

echo "Setting up a Python virtual environment..."
# Check if the virtual environment already exists
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
    echo "Virtual environment created at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate
echo "Virtual environment activated."

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip


pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer
pip install bitsandbytes scipy accelerate datasets
pip install sentencepiece

echo "Cloning IndicTransToolkit repository..."
if [ ! -d "IndicTransToolkit" ]; then
    git clone https://github.com/VarunGumma/IndicTransToolkit.git
else
    echo "IndicTransToolkit already cloned."
fi

echo "Installing IndicTransToolkit in editable mode..."
cd IndicTransToolkit
python3 -m pip install --editable ./
cd ..

echo "IndicTransToolkit setup complete."
