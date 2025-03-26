#!/bin/bash
#SBATCH --job-name=temporal_analysis
#SBATCH --time=24:00:00
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

# Install required packages
echo "Installing required packages..."
pip install --no-cache-dir transformers numpy matplotlib seaborn pandas scipy cvxpy tqdm

# Verify installations
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import cvxpy; print('CVXPY version:', cvxpy.__version__)"

# Set high data volume for more reliable results
TEXTS_PER_DECADE=10000

# Run analysis for all distribution patterns with bootstrap validation
echo "Running analyses with high data volume (${TEXTS_PER_DECADE} texts per decade)..."

echo "Running uniform distribution analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution uniform --texts_per_decade ${TEXTS_PER_DECADE} --bootstrap --bootstrap_iterations 100

echo "Running recency bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution recency_bias --texts_per_decade ${TEXTS_PER_DECADE} --bootstrap --bootstrap_iterations 100

echo "Running historical bias analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution historical_bias --texts_per_decade ${TEXTS_PER_DECADE} --bootstrap --bootstrap_iterations 100

echo "Running bimodal distribution analysis..."
python run_on_maxwell.py --tokenizer gpt2 --distribution bimodal --texts_per_decade ${TEXTS_PER_DECADE} --bootstrap --bootstrap_iterations 100

# Compare results across different tokenizers (using recency_bias as the test case)
echo "Running tokenizer comparison with gpt2-medium..."
python run_on_maxwell.py --tokenizer gpt2-medium --distribution recency_bias --texts_per_decade ${TEXTS_PER_DECADE} --bootstrap

echo "Running tokenizer comparison with bert-base-uncased..."
python run_on_maxwell.py --tokenizer bert-base-uncased --distribution recency_bias --texts_per_decade ${TEXTS_PER_DECADE} --bootstrap

# Create comparison visualizations
echo "Creating comprehensive comparison visualizations..."
python -c "
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from pathlib import Path

# Load all result files
result_files = glob.glob('results/distributions/*_distribution.json')
results = {}

for file in result_files:
    with open(file, 'r') as f:
        data = json.load(f)
        key = f\"{data['tokenizer']}_{Path(file).stem.split('_')[1]}\"
        results[key] = data

# Create comparison figure
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.title('log10(MSE) Comparison')
tokenizers = set(k.split('_')[0] for k in results)
distributions = set(k.split('_')[1] for k in results)

# Plot by tokenizer
for dist in distributions:
    values = [results.get(f'{tok}_{dist}', {}).get('evaluation', {}).get('log10_mse', 0) 
             for tok in tokenizers]
    plt.bar(list(tokenizers), values, label=dist)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('results/final_comparison.png', dpi=300)
"

echo "Job completed at: $(date)"