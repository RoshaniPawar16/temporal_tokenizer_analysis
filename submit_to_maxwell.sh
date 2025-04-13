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

# Load required modules for Python
module load python/3.9.5

# Verify Python version
echo "Python version:"
python --version

# Create and activate a conda environment (more reliable than venv)
if [ ! -d "conda_env" ]; then
    echo "Creating conda environment..."
    module load anaconda3/2022.10
    conda create -p ./conda_env python=3.9.5 -y
fi

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./conda_env

# Verify environment Python version
echo "Conda environment Python version:"
python --version

# Install required packages with specific versions
echo "Installing required packages..."
pip install --no-cache-dir \
    transformers==4.30.0 \
    datasets==2.14.0 \
    numpy==1.24.3 \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    pandas==2.0.3 \
    scipy==1.10.1 \
    cvxpy==1.3.1 \
    tqdm==4.65.0 \
    huggingface_hub==0.16.4 \
    bs4==0.0.1 \
    requests==2.31.0

# Configure environment variables for Hugging Face
export HF_HOME="./hf_cache"
export HF_HUB_CACHE="./hf_cache/hub"
export HF_DATASETS_CACHE="./hf_cache/datasets"
mkdir -p $HF_HOME $HF_HUB_CACHE $HF_DATASETS_CACHE

# Configure proxy if needed (uncomment and modify if required)
# export HTTP_PROXY="http://proxy.abdn.ac.uk:8080"
# export HTTPS_PROXY="http://proxy.abdn.ac.uk:8080"

# Configure environment for better memory usage
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Test dataset access - verify before proceeding to main analysis
echo "Testing dataset access..."
python -c "
from datasets import load_dataset
try:
    # Test with a small dataset that should load quickly
    data = load_dataset('csv', data_files={'test': 'path/to/small_test.csv'}, split='test', trust_remote_code=True)
    print('Dataset test successful')
except Exception as e:
    print(f'Dataset test failed: {e}')
"

# Run analysis with memory-optimized settings
echo "Running uniform distribution analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution uniform --texts_per_decade 2000 --target_size_gb 0.5 --bootstrap --bootstrap_iterations 30

echo "Running recency bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution recency_bias --texts_per_decade 2000 --target_size_gb 0.5 --bootstrap --bootstrap_iterations 30

echo "Running historical bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution historical_bias --texts_per_decade 2000 --target_size_gb 0.5 --bootstrap --bootstrap_iterations 30

echo "Running bimodal distribution analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution bimodal --texts_per_decade 2000 --target_size_gb 0.5 --bootstrap --bootstrap_iterations 30

echo "Job completed at: $(date)"