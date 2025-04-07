#!/bin/bash
#SBATCH --job-name=temporal_analysis
#SBATCH --time=15:00:00
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
pip install --no-cache-dir transformers numpy matplotlib seaborn pandas scipy cvxpy tqdm datasets huggingface_hub

# Set up cache for Hugging Face datasets with memory limits
export HF_DATASETS_CACHE="./hf_cache"
mkdir -p $HF_DATASETS_CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=4000000000  # 4GB max memory for datasets

# Configure environment for better memory usage
export TOKENIZERS_PARALLELISM=false  # Disable parallel tokenization to reduce memory
export OMP_NUM_THREADS=4  # Limit number of OpenMP threads

# Set chunk size for processing to avoid memory issues
export PYTHONUNBUFFERED=1

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