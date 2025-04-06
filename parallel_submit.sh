#!/bin/bash
#SBATCH --job-name=temporal_parallel
#SBATCH --time=15:00:00    # Same time limit as original
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8  # Same CPU count as original
#SBATCH --mem=64G          # Same memory as original
#SBATCH --output=temporal_parallel_%A_%a.log
#SBATCH --array=0-5%3      # Run 6 tasks, max 3 at a time

# Display information about the job
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"

# Load Python module available on Maxwell
module load python/3.9.5

# Define the analyses to run in parallel - format: "distribution:tokenizer"
ANALYSES=(
    "uniform:gpt2"
    "recency_bias:gpt2"
    "historical_bias:gpt2"
    "bimodal:gpt2"
    "recency_bias:gpt2-medium"
    "recency_bias:bert-base-uncased"
)

# Extract the specific analysis for this array task
TASK=${ANALYSES[$SLURM_ARRAY_TASK_ID]}
DISTRIBUTION=${TASK%:*}
TOKENIZER=${TASK#*:}

echo "Running analysis: Distribution=$DISTRIBUTION, Tokenizer=$TOKENIZER"

# Create a working directory for this task to isolate environments
WORK_DIR="analysis_${DISTRIBUTION}_${TOKENIZER}"
mkdir -p $WORK_DIR
cd $WORK_DIR

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

# Set up cache for Hugging Face datasets (shared across all tasks)
export HF_DATASETS_CACHE="../hf_cache"
mkdir -p $HF_DATASETS_CACHE

# Verify installations
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import cvxpy; print('CVXPY version:', cvxpy.__version__)"
python -c "import datasets; print('Datasets version:', datasets.__version__)"

# Only download dataset in first task to avoid redundant downloads
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    # Pre-fetch British Library dataset to avoid re-downloading during analysis
    echo "Pre-fetching British Library dataset to local cache (this may take some time)..."
    python -c "from datasets import load_dataset; print('Starting dataset download...'); load_dataset('TheBritishLibrary/blbooks', '1500_1899', trust_remote_code=True, cache_dir='../hf_cache'); print('Dataset pre-fetching complete')"
else
    echo "Skipping dataset pre-fetch as it should be handled by task 0"
fi

# Set high data volume for more reliable results and match Hayase et al.
TEXTS_PER_DECADE=10000
TARGET_SIZE_GB=1.5  # Match paper's 1GB per category for analysis

echo "Running analysis with high data volume (${TEXTS_PER_DECADE} texts per decade, ${TARGET_SIZE_GB}GB per category)..."

# Run the analysis with the parameters for this task
echo "Running $DISTRIBUTION distribution analysis with $TOKENIZER..."
python ../run_on_maxwell.py --tokenizer $TOKENIZER --distribution $DISTRIBUTION --texts_per_decade ${TEXTS_PER_DECADE} --target_size_gb ${TARGET_SIZE_GB} --bootstrap --bootstrap_iterations 50

# Return to original directory
cd ..

echo "Analysis of $DISTRIBUTION with $TOKENIZER completed at: $(date)"

# Create visualization job submission script if this is the last task
if [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    echo "Creating visualization job submission script..."
    cat > viz_submit.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=temporal_viz
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=temporal_viz_%j.log

# Display information about the job
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"

# Load Python module
module load python/3.9.5

# Create and activate environment
source analysis_uniform_gpt2/venv/bin/activate

# Create comparison visualizations
echo "Creating comprehensive comparison visualizations..."
python -c "
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from pathlib import Path
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load all result files
result_files = glob.glob('results/distributions/*_distribution.json')
results = {}

for file in result_files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            key = f\"{data['tokenizer']}_{Path(file).stem.split('_')[1]}\"
            results[key] = data
    except Exception as e:
        print(f'Error loading {file}: {e}')

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
print('Saved comparison visualization to results/final_comparison.png')
"

echo "Visualization completed at: $(date)"
EOF

    chmod +x viz_submit.sh
    echo "Submitting visualization job..."
    sbatch --dependency=afterok:$SLURM_ARRAY_JOB_ID viz_submit.sh
fi

echo "Job completed at: $(date)"