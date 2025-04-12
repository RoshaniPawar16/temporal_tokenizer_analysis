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

# Clean up any previous environment
if [ -d "venv" ]; then
    rm -rf venv
fi

# Find Python version
PYTHON_CMD=$(which python3)
echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Create virtual environment
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip --no-cache-dir

# Install all required packages - note that seaborn is included
pip install --no-cache-dir numpy scipy matplotlib pandas tqdm seaborn
pip install --no-cache-dir beautifulsoup4 requests bs4
pip install --no-cache-dir transformers datasets huggingface_hub
pip install --no-cache-dir cvxpy psutil

# Configure environment
export HF_DATASETS_CACHE="./hf_cache"
mkdir -p $HF_DATASETS_CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=4000000000
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Skipping the problematic sed command - manual fix has been applied
echo "Skipping automatic dataset_manager.py modification - manual fix applied"

# First run a minimal test to make sure imports work
echo "Testing imports..."
python -c "import numpy; import scipy; import matplotlib.pyplot; import seaborn; import cvxpy; print('Basic imports successful')" || echo "Basic imports failed"

# Create directory for results
mkdir -p results/figures
mkdir -p results/distributions
mkdir -p results/bootstrap

# Run each analysis with error handling and reduced iterations for testing
echo "Running uniform distribution analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution uniform --texts_per_decade 5000 --target_size_gb 0.5 --bootstrap --bootstrap_iterations 10 || {
    echo "Uniform distribution analysis failed with exit code $?";
    echo "Saving trace...";
    python -c "import traceback; traceback.print_exc()" > error_uniform.log;
}

echo "Running recency bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution recency_bias --texts_per_decade 5000 --target_size_gb 0.5 --bootstrap --bootstrap_iterations 10 || {
    echo "Recency bias analysis failed with exit code $?";
    echo "Saving trace...";
    python -c "import traceback; traceback.print_exc()" > error_recency.log;
}

echo "Running historical bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution historical_bias --texts_per_decade 5000 --target_size_gb 0.5 --bootstrap --bootstrap_iterations 10 || {
    echo "Historical bias analysis failed with exit code $?";
    echo "Saving trace...";
    python -c "import traceback; traceback.print_exc()" > error_historical.log;
}

echo "Running bimodal distribution analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution bimodal --texts_per_decade 5000 --target_size_gb 0.5 --bootstrap --bootstrap_iterations 10 || {
    echo "Bimodal distribution analysis failed with exit code $?";
    echo "Saving trace...";
    python -c "import traceback; traceback.print_exc()" > error_bimodal.log;
}

echo "Job completed at: $(date)"