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

# Load required modules (adjust as needed for Maxwell environment)
module load python/3.9

# Create and activate a virtual environment if needed
# python -m venv .venv
# source .venv/bin/activate

# Install required packages if needed
# pip install transformers numpy matplotlib seaborn cvxpy pandas

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