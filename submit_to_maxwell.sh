#!/bin/bash

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

#SBATCH --job-name=temporal_analysis
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=temporal_analysis_%j.log

# Display information about the job
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"

# Load required modules for Python
module load anaconda3/2022.10

# Create and activate a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages including datasets for British Library
echo "Installing required packages..."
pip install --no-cache-dir transformers numpy matplotlib seaborn pandas scipy cvxpy tqdm datasets huggingface_hub bs4 psutil retrying

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
PYTHONMEMMON=1 python run_on_maxwell.py --tokenizer gpt2 --distribution uniform --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30

echo "Running recency bias analysis..."
PYTHONMEMMON=1 python run_on_maxwell.py --tokenizer gpt2 --distribution recency_bias --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30

echo "Running historical bias analysis..."
PYTHONMEMMON=1 python run_on_maxwell.py --tokenizer gpt2 --distribution historical_bias --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30

echo "Running bimodal distribution analysis..."
PYTHONMEMMON=1 python run_on_maxwell.py --tokenizer gpt2 --distribution bimodal --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30

echo "Job completed at: $(date)"