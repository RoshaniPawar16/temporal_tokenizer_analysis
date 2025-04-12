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
    fi
    
    echo "Cleaning up..."
    echo "Job exited with status $EXIT_CODE"
    exit $EXIT_CODE
}

# Set up the trap
trap cleanup EXIT

# Display information about the job
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"

# Create a clean conda environment instead of venv
echo "Setting up conda environment..."
module load anaconda3  # Try to load any anaconda module

# If previous conda environment exists, remove it to avoid conflicts
if [ -d "./conda_env" ]; then
    rm -rf ./conda_env
fi

# Create a new conda environment with specific Python version
conda create -y -p ./conda_env python=3.9
source activate ./conda_env

# Verify Python version
python --version

# Install required packages
echo "Installing required packages..."
pip install --no-cache-dir numpy==1.22.4 scipy matplotlib seaborn pandas cvxpy tqdm datasets huggingface_hub transformers bs4 psutil retrying

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
python run_on_maxwell.py --tokenizer gpt2 --distribution uniform --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Uniform distribution analysis failed"

echo "Running recency bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution recency_bias --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Recency bias analysis failed"

echo "Running historical bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution historical_bias --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Historical bias analysis failed"

echo "Running bimodal distribution analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution bimodal --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Bimodal distribution analysis failed"

echo "Job completed at: $(date)"