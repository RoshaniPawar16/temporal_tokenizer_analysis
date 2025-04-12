#!/bin/bash
#SBATCH --job-name=temporal_analysis
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=temporal_analysis_%j.log

# Enhanced error handling
set -e  # Exit immediately if a command exits with a non-zero status

# Function to handle errors and clean up
cleanup() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Error detected, exit code: $EXIT_CODE"
        echo "Saving any partial results..."
        # Run a simple Python script to save whatever state we can
        python -c '
import pickle
import time
import sys
import os
from pathlib import Path

# Try to import key modules and save any data they have
try:
    cache_dir = Path("./hf_cache")
    checkpoint_dir = cache_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to pickle anything useful from the globals
    emergency_data = {}
    
    # Try to save the module state
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    emergency_path = checkpoint_dir / f"emergency_recovery_{timestamp}.pkl"
    
    with open(emergency_path, "wb") as f:
        pickle.dump(emergency_data, f)
    
    print(f"Emergency recovery data saved to {emergency_path}")
except Exception as e:
    print(f"Emergency recovery failed: {e}")
'
    fi
    
    echo "Cleaning up..."
    # Add any cleanup tasks here
    
    echo "Job exited with status $EXIT_CODE"
    exit $EXIT_CODE
}

# Set up the trap
trap cleanup EXIT

# Display information about the job
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"

# Check which Python modules are available
echo "Available Python modules:"
module spider python

# Try to load an appropriate Python module based on what's available
# First try a more generic Python module
module load python || module load python3 || echo "Failed to load Python module, checking for built-in Python"

# Check if Python is already available without modules
which python || which python3 || echo "No Python found in PATH"
python --version || python3 --version || echo "Could not get Python version"

# Create and activate a virtual environment
# Use Python directly if available, otherwise try python3
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv || python3 -m venv venv || echo "Failed to create virtual environment"
fi

if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "WARNING: Working without a virtual environment"
fi

# Install required packages including datasets for British Library
echo "Installing required packages..."
pip install --no-cache-dir transformers numpy matplotlib seaborn pandas scipy cvxpy tqdm datasets huggingface_hub bs4 psutil retrying || echo "Failed to install some packages"

# Set up cache for Hugging Face datasets with memory limits
export HF_DATASETS_CACHE="./hf_cache"
mkdir -p $HF_DATASETS_CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=4000000000  # 4GB max memory for datasets

# Configure environment for better memory usage
export TOKENIZERS_PARALLELISM=false  # Disable parallel tokenization to reduce memory
export OMP_NUM_THREADS=4  # Limit number of OpenMP threads

# Set chunk size for processing to avoid memory issues
export PYTHONUNBUFFERED=1

export HF_DATASETS_TRUST_REMOTE_CODE=1

# Run analysis with memory-optimized settings
echo "Running uniform distribution analysis..."
PYTHONMEMMON=1 python run_on_maxwell.py --tokenizer gpt2 --distribution uniform --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Uniform distribution analysis failed"

echo "Running recency bias analysis..."
PYTHONMEMMON=1 python run_on_maxwell.py --tokenizer gpt2 --distribution recency_bias --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Recency bias analysis failed"

echo "Running historical bias analysis..."
PYTHONMEMMON=1 python run_on_maxwell.py --tokenizer gpt2 --distribution historical_bias --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Historical bias analysis failed"

echo "Running bimodal distribution analysis..."
PYTHONMEMMON=1 python run_on_maxwell.py --tokenizer gpt2 --distribution bimodal --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Bimodal distribution analysis failed"

echo "Job completed at: $(date)"