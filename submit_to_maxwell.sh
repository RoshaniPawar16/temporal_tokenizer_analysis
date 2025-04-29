#!/bin/bash
#SBATCH --job-name=temporal_analysis
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --account=sncs
#SBATCH --gres=gpu:1                
#SBATCH --partition=a100_full       
#SBATCH --output=temporal_analysis_%j.out
#SBATCH --error=temporal_analysis_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t06rp23@abdn.ac.uk  

# Display information about the job
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"

# Purge all modules to avoid conflicts
module purge
# Load Python 3.9.12
module load python/3.9.12

# Verify Python version
echo "Python version:"
python --version

# Install packages to a local directory without using venv
# This approach avoids issues with Python version mismatch
echo "Setting up pip install directory..."
mkdir -p $HOME/.local/pip/temporal_analysis
export PYTHONPATH=$HOME/.local/pip/temporal_analysis:$PYTHONPATH
export PIP_TARGET=$HOME/.local/pip/temporal_analysis

# Install required packages with specific versions
echo "Installing required packages..."
pip install --no-cache-dir \
    transformers \
    datasets \
    numpy \
    matplotlib \
    seaborn \
    pandas \
    scipy \
    cvxpy \
    tqdm \
    huggingface_hub \
    bs4 \
    requests \
    psutil \
    tiktoken

# Configure environment variables for Hugging Face
export HF_HOME="./hf_cache"
export HF_HUB_CACHE="./hf_cache/hub"
export HF_DATASETS_CACHE="./hf_cache/datasets"
mkdir -p $HF_HOME $HF_HUB_CACHE $HF_DATASETS_CACHE

# Configure environment for better memory usage
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Verify installation by checking NumPy version
echo "Checking NumPy installation:"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Test dataset access - use a simple test
echo "Testing basic imports..."
python -c "import transformers; import datasets; print('Basic imports successful')"

# # Run analysis with increased data volume and filtering
# echo "Running analysis for all distributions..."
# python run_on_maxwell.py --tokenizer gpt2 --distribution all --texts_per_decade 5000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 --verbose

# # Run analysis with reduced memory requirements
# echo "Running uniform distribution analysis..."
# python run_on_maxwell.py --tokenizer gpt2 --distribution uniform --texts_per_decade 2000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30 --verbose

# echo "Running recency bias analysis..."
# python run_on_maxwell.py --tokenizer gpt2 --distribution recency_bias --texts_per_decade 2000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30

# echo "Running historical bias analysis..."
# python run_on_maxwell.py --tokenizer gpt2 --distribution historical_bias --texts_per_decade 2000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30

# echo "Running bimodal distribution analysis..."
# python run_on_maxwell.py --tokenizer gpt2 --distribution bimodal --texts_per_decade 2000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30

# # Optional: Only run comparison if individual runs succeeded
# echo "Running all distributions comparison..."
# python run_on_maxwell.py --tokenizer gpt2 --distribution all --texts_per_decade 2000 --target_size_gb 1.0 --bootstrap --bootstrap_iterations 30

# # Run multi-model analysis for all distributions
# echo "Running multi-model analysis..."

# # Define all distributions
# DISTRIBUTIONS=(uniform recency_bias historical_bias bimodal)

# # Run multi-model analysis for each distribution
# for dist in "${DISTRIBUTIONS[@]}"; do
#     echo "Running analysis for $dist distribution with multiple models..."
#     python run_multimodel_analysis.py \
#         --models gpt2,bert-base-uncased,roberta-base \
#         --distribution $dist \
#         --target_size_gb 1.0 \
#         --bootstrap_iterations 30 \
#         --texts_per_decade 5000
    
#     # Clean up resources between runs
#     python -c "import gc; gc.collect()"
#     sleep 10
# done

# Run multi-model analysis for all distributions
echo "Running multi-model temporal distribution analysis..."

# Define all distributions
DISTRIBUTIONS=(uniform recency_bias historical_bias bimodal)

# Use our recommended models with distilgpt2 included
# MODELS="gpt2,bert-base-uncased,roberta-base,llama,mistral"
MODELS="gpt2,bert-base-uncased,roberta-base,xlm-roberta-base,t5-small,electra-small-discriminator,llama,gpt2-medium,distilgpt2,distilbert-base-uncased,albert-base-v2,mistral"

# Run multi-model analysis for each distribution with enhanced output
for dist in "${DISTRIBUTIONS[@]}"; do
    echo ""
    echo "=================================================================="
    echo "STARTING ANALYSIS FOR $dist DISTRIBUTION"
    echo "Models: $MODELS"
    echo "Time: $(date)"
    echo "=================================================================="
    echo ""
    
    python run_multimodel_analysis.py \
        --models $MODELS \
        --distribution $dist \
        --target_size_gb 1.0 \
        --bootstrap_iterations 20 \
        --top_n_tokens 40 \
        --texts_per_decade 5000 \
        --enhanced
    
    # More thorough cleanup between runs
    echo "Cleaning up resources..."
    python -c "
import gc
import sys
gc.collect()
print(f'Memory cleaned up, objects collected: {gc.collect()}')
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('CUDA cache cleared')
except ImportError:
    pass
"
    # Give the system time to release resources
    echo "Waiting for resource release..."
    sleep 30
    
    echo ""
    echo "=================================================================="
    echo "COMPLETED ANALYSIS FOR $dist DISTRIBUTION"
    echo "Time: $(date)"
    echo "=================================================================="
    echo ""
done


echo "Job completed at: $(date)"