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
trap 'echo "Error occurred, cleaning up"; exit 1' ERR

# Display information about the job
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"

# Clean up any previous environment to avoid conflicts
if [ -d "venv" ]; then
    rm -rf venv
fi

# Find which system Python to use - prefer 3.9 if available
PYTHON_CMD=""
if command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "No suitable Python found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Create a fresh virtual environment
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Install a compatible numpy version first
pip install --no-cache-dir numpy==1.22.4

# Install other dependencies
pip install --no-cache-dir scipy matplotlib seaborn pandas cvxpy tqdm 
pip install --no-cache-dir transformers datasets huggingface_hub bs4 psutil retrying

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