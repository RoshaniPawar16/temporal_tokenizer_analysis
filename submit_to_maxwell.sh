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

# Install packages with --only-binary flag to avoid compilation
pip install --only-binary=:all: numpy scipy matplotlib pandas tqdm --no-cache-dir
pip install --only-binary=:all: beautifulsoup4 requests --no-cache-dir

# Try a minimal set of packages for your analysis
# Avoid packages that require compilation
pip install --only-binary=:all: transformers --no-cache-dir || echo "Failed to install transformers"

# Configure environment
export HF_DATASETS_CACHE="./hf_cache"
mkdir -p $HF_DATASETS_CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=4000000000
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Let's modify the approach to fix dataset_manager.py issue
echo "Modifying dataset_manager.py to fix tuple unpacking issue..."
sed -i 's/for decade, volume in volume_check\.items()/volume_check, all_sufficient = self.verify_dataset_volumes(controlled_dataset)\nfor decade, volume in volume_check.items()/' src/data/dataset_manager.py || echo "Failed to fix dataset_manager.py"

# First run a minimal test to make sure imports work
echo "Testing imports..."
python -c "import numpy; import scipy; import matplotlib.pyplot; print('Basic imports successful')" || echo "Basic imports failed"

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