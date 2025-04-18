#!/bin/bash
#SBATCH --job-name=test_temporal
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=test_temporal_%j.out
#SBATCH --error=test_temporal_%j.err

# Display information about the job
echo "Running test on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"

# Load modules
module purge
module load python/3.9.12

# Verify Python version
echo "Python version:"
python --version

# Run test with minimal data
python run_on_maxwell.py \
    --tokenizer gpt2 \
    --distribution uniform \
    --test_mode \
    --test_size_mb 5 \
    --test_decades "1960s,2000s" \
    --texts_per_decade 20 \
    --bootstrap \
    --bootstrap_iterations 2 \
    --verbose

echo "Test completed at: $(date)"