#!/bin/bash

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements/base.txt
pip install -r requirements/gpu.txt  # if GPU available

# Create necessary directories if they don't exist
mkdir -p data/{processed,raw,test_samples}
mkdir -p weights/pconv/{unet,vgg16}
mkdir -p temp_weights

# Verify weights
python scripts/weight_conversion/verify_weights.py

# Install the package in development mode
pip install -e .


'''
setup.sh:

One-time setup script
Run only when:

Setting up the project first time
Installing new dependencies
Updating environment


Not needed every time you run app.py
'''