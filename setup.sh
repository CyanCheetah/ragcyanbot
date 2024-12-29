#!/bin/bash

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install build requirements
brew install cmake

# Install Python packages
pip install -r requirements.txt

# Install llama-cpp-python with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Run setup script
python setup.py

# Test LLaMA
python test_llama.py 