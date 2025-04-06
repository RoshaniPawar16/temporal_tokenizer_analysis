#!/bin/bash
#SBATCH --job-name=temporal_parallel
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/temporal_parallel_%A_%a.log # Store logs in a 'logs' subdir
#SBATCH --error=logs/temporal_parallel_%A_%a.err  # Store errors separately
#SBATCH --array=0-5%3

# --- Setup ---
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM Job ID: $SLURM_ARRAY_JOB_ID"

# --- Load Module FIRST ---
# Ensure the correct Python is available before creating/activating venv
echo "Loading Python module..."
module load python/3.9.5
# Check if module loaded successfully (optional but good practice)
if ! command -v python &> /dev/null; then
    echo "ERROR: Python command not found after loading module."
    exit 1
fi
echo "Python executable path after module load: $(which python)"
echo "Python version: $(python --version)"

# --- Define Analysis Task ---
ANALYSES=(
    "uniform:gpt2"
    "recency_bias:gpt2"
    "historical_bias:gpt2"
    "bimodal:gpt2"
    "recency_bias:gpt2-medium"
    "recency_bias:bert-base-uncased"
)
TASK=${ANALYSES[$SLURM_ARRAY_TASK_ID]}
DISTRIBUTION=${TASK%:*}
TOKENIZER=${TASK#*:}
echo "Running analysis: Distribution=$DISTRIBUTION, Tokenizer=$TOKENIZER"

# --- Environment Setup ---
# Option 1: Shared Environment (Recommended for efficiency)
VENV_DIR="../shared_venv" # Place venv outside task-specific dirs
PYTHON_EXEC="$VENV_DIR/bin/python"
PIP_EXEC="$VENV_DIR/bin/pip"

# Create shared venv only in task 0 to avoid race conditions
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating shared virtual environment in $VENV_DIR..."
        # Use the *loaded* python to create the venv
        python -m venv "$VENV_DIR"
        # Upgrade pip within the new environment
        "$PIP_EXEC" install --upgrade pip
        # Install requirements
        echo "Installing required packages in shared venv..."
        "$PIP_EXEC" install --no-cache-dir transformers numpy matplotlib seaborn pandas scipy cvxpy tqdm datasets huggingface_hub
    else
        echo "Shared virtual environment $VENV_DIR already exists."
        # Optionally, update packages if needed
        # echo "Updating packages in shared venv..."
        # "$PIP_EXEC" install --upgrade --no-cache-dir transformers numpy matplotlib seaborn pandas scipy cvxpy tqdm datasets huggingface_hub
    fi
else
    # Other tasks wait briefly to ensure task 0 likely finished venv creation
    sleep 15
    # Check if venv exists, exit if task 0 failed
    if [ ! -d "$VENV_DIR" ]; then
        echo "ERROR: Shared virtual environment $VENV_DIR not found. Task 0 might have failed."
        exit 1
    fi
fi

# --- Cache Setup ---
export HF_DATASETS_CACHE="../hf_cache" # Shared HF cache
mkdir -p "$HF_DATASETS_CACHE"

# --- Dataset Pre-fetch (Task 0 Only) ---
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "Pre-fetching British Library dataset to cache $HF_DATASETS_CACHE..."
    # Use the Python from the venv to run the pre-fetch command
    "$PYTHON_EXEC" -c "from datasets import load_dataset; print('Starting dataset download...'); load_dataset('TheBritishLibrary/blbooks', '1500_1899', trust_remote_code=True, cache_dir='$HF_DATASETS_CACHE'); print('Dataset pre-fetching complete')"
    PREFETCH_STATUS=$?
    if [ $PREFETCH_STATUS -ne 0 ]; then
        echo "ERROR: Dataset pre-fetching failed."
        # Optionally create a signal file so other tasks know not to run
        touch ../PREFETCH_FAILED
        exit 1
    fi
else
    # Wait for task 0 potentially longer if pre-fetch is slow
    echo "Waiting for dataset pre-fetch by task 0..."
    sleep 60 # Adjust sleep time as needed
    if [ -f "../PREFETCH_FAILED" ]; then
        echo "ERROR: Dataset pre-fetch failed in task 0. Exiting."
        exit 1
    fi
    # Verify dataset exists in cache (optional)
    # if ! [ -d "$HF_DATASETS_CACHE/TheBritishLibrary___blbooks" ]; then
    #     echo "WARNING: Dataset cache directory not found after waiting. Proceeding anyway..."
    # fi
fi


# --- Run Analysis ---
TEXTS_PER_DECADE=10000
TARGET_SIZE_GB=1.5

echo "Running analysis with high data volume (${TEXTS_PER_DECADE} texts per decade, ${TARGET_SIZE_GB}GB per category)..."

# Create a task-specific working directory (optional, but good for logs/outputs)
WORK_DIR="analysis_${DISTRIBUTION}_${TOKENIZER}"
mkdir -p $WORK_DIR
cd $WORK_DIR || exit 1 # Exit if cd fails

# Run the Python script using the virtual environment's Python
echo "Running $DISTRIBUTION distribution analysis with $TOKENIZER..."
"$PYTHON_EXEC" ../run_on_maxwell.py \
    --tokenizer "$TOKENIZER" \
    --distribution "$DISTRIBUTION" \
    --texts_per_decade "$TEXTS_PER_DECADE" \
    --target_size_gb "$TARGET_SIZE_GB" \
    --bootstrap \
    --bootstrap_iterations 50

ANALYSIS_STATUS=$?
cd .. # Return to original directory

if [ $ANALYSIS_STATUS -ne 0 ]; then
    echo "ERROR: Analysis script failed for $DISTRIBUTION with $TOKENIZER."
    exit 1 # Exit the SLURM task with an error status
fi

echo "Analysis of $DISTRIBUTION with $TOKENIZER completed at: $(date)"

# --- Visualization Submission (Task 5 Only) ---
# Note: This part assumes the visualization script uses the same shared venv
if [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    echo "Creating visualization job submission script..."
    cat > viz_submit.sh << EOF
#!/bin/bash
#SBATCH --job-name=temporal_viz
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/temporal_viz_%j.log # Store viz logs too
#SBATCH --error=logs/temporal_viz_%j.err

echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"

# Load Python module (necessary for the batch script's shell)
module load python/3.9.5

# Use the shared environment for visualization
PYTHON_EXEC="$VENV_DIR/bin/python"

# Create results directory if it doesn't exist
mkdir -p results

# Run the Python visualization code
echo "Creating comprehensive comparison visualizations..."
"$PYTHON_EXEC" -c "
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from pathlib import Path
import os
import pandas as pd # Import pandas for easier data handling

print('Starting visualization...')
# Load all result files from subdirectories
result_files = glob.glob('analysis_*/results/distributions/*_distribution.json')
print(f'Found {len(result_files)} result files.')
results_data = []

for file in result_files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            # Extract info needed for plotting
            results_data.append({
                'tokenizer': data.get('tokenizer', 'unknown'),
                'distribution': data.get('distribution', 'unknown'),
                'log10_mse': data.get('evaluation', {}).get('log10_mse', None),
                'avg_overlap': data.get('evaluation', {}).get('average_token_overlap', None),
                # Add other metrics if available and desired
            })
    except Exception as e:
        print(f'Error loading or processing {file}: {e}')

if not results_data:
    print('No valid results found to visualize.')
    exit()

df = pd.DataFrame(results_data)
df = df.dropna(subset=['log10_mse']) # Drop rows where MSE couldn't be calculated

print('Results DataFrame Head:')
print(df.head())

# Create comparison figure (Example: Bar plot of MSE)
plt.figure(figsize=(15, 8))
sns.barplot(data=df, x='tokenizer', y='log10_mse', hue='distribution')
plt.title('Comparison of log10(MSE) across Tokenizers and Distributions')
plt.ylabel('log10(Mean Squared Error)')
plt.xlabel('Tokenizer')
plt.xticks(rotation=15, ha='right')
plt.legend(title='Distribution', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.grid(axis='y', linestyle='--', alpha=0.7)

output_path = 'results/final_mse_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Saved comparison visualization to {output_path}')

# Add more plots here if needed (e.g., for token overlap)

plt.close() # Close the plot figure
print('Visualization script finished.')
"

echo "Visualization completed at: $(date)"
EOF

    chmod +x viz_submit.sh
    echo "Submitting visualization job..."
    # Make sure the dependency is correct - depends on the whole array job
    sbatch --dependency=afterok:$SLURM_ARRAY_JOB_ID viz_submit.sh
fi

echo "Job task $SLURM_ARRAY_TASK_ID completed at: $(date)"