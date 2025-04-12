#!/bin/bash
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

# Clean up any previous environment to avoid conflicts
if [ -d "venv" ]; then
    rm -rf venv
fi

# Find which system Python to use
echo "Looking for available Python versions..."
which python3.9 || which python3.8 || which python3.7 || which python3.6 || which python3 || which python

# Use the latest available Python
PYTHON_CMD=$(which python3.9 || which python3.8 || which python3.7 || which python3.6 || which python3 || which python)

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# First upgrade pip
$PYTHON_CMD -m pip install --upgrade pip

# Create a fresh virtual environment
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Install dependencies flexibly (without specifying versions)
echo "Installing dependencies..."
pip install --no-cache-dir numpy scipy matplotlib seaborn pandas
pip install --no-cache-dir cvxpy tqdm bs4 psutil retrying
pip install --no-cache-dir transformers datasets huggingface_hub

# Configure environment
export HF_DATASETS_CACHE="./hf_cache"
mkdir -p $HF_DATASETS_CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=4000000000
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Run analyses
echo "Running uniform distribution analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution uniform --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Uniform distribution analysis failed"

echo "Running recency bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution recency_bias --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Recency bias analysis failed"

echo "Running historical bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution historical_bias --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Historical bias analysis failed"

echo "Running bimodal distribution analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution bimodal --texts_per_decade 10000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 || echo "Bimodal distribution analysis failed"

echo "Job completed at: $(date)"