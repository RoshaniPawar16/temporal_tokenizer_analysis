#!/bin/bash
#SBATCH --job-name=test_temporal
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=test_temporal_%j.out
#SBATCH --error=test_temporal_%j.err

# Display information about the job
echo "Running test on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"

# Load modules
module purge
module load python/3.9.12

# Verify Python version
echo "Python version:"
python --version

# Install required packages to a local directory
echo "Installing required packages..."
pip install --user transformers datasets numpy matplotlib seaborn pandas scipy cvxpy tqdm huggingface_hub bs4 requests psutil

# Configure environment variables for Hugging Face
export HF_HOME="./hf_cache"
export HF_HUB_CACHE="./hf_cache/hub"
export HF_DATASETS_CACHE="./hf_cache/datasets"
mkdir -p $HF_HOME $HF_HUB_CACHE $HF_DATASETS_CACHE

# Configure environment for better memory usage
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Run test with minimal data
python run_on_maxwell.py \
    --tokenizer gpt2 \
    --distribution uniform \
    --test_mode \
    --test_size_mb 5 \
    --test_decades "1960s,2000s" \
    --texts_per_decade 20 \
    --bootstrap \
    --bootstrap_iterations 2 \
    --verbose

echo "Test completed at: $(date)"