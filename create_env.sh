#!/bin/bash
set -e

CONDA_DIR="/home/vishal/miniconda3"
ENV_NAME="monopoly"

echo "=========================================================="
echo "Installing pip dependencies in monopoly using local CUDA..."
echo "=========================================================="
# Since the environment already exists, we can directly run pip in it
$CONDA_DIR/envs/$ENV_NAME/bin/pip install -U "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
$CONDA_DIR/envs/$ENV_NAME/bin/pip install streamlit plotly pandas matplotlib jinja2

echo "=========================================================="
echo "Environment setup complete!"
echo "=========================================================="
