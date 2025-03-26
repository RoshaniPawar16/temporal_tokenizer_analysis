#!/bin/bash
#SBATCH --job-name=temporal_analysis
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

# Install required packages
echo "Installing required packages..."
pip install --no-cache-dir transformers numpy matplotlib seaborn pandas cvxpy tqdm

# Verify installations
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"

# Run the analysis script with different configurations
echo "Running uniform distribution analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution uniform --texts_per_decade 100 --bootstrap

echo "Running recency bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution recency_bias --texts_per_decade 100 --bootstrap

echo "Running historical bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution historical_bias --texts_per_decade 100 --bootstrap

echo "Running bimodal distribution analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution bimodal --texts_per_decade 100 --bootstrap

# Compare to other tokenizer
echo "Running comparison with gpt2-medium..."
python run_on_maxwell.py --tokenizer gpt2-medium --distribution recency_bias --texts_per_decade 100

echo "Job completed at: $(date)"