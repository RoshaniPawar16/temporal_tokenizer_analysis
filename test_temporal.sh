#!/bin/bash
#SBATCH --job-name=test_temporal
#SBATCH --time=00:30:00  # Reduced from 1 hour to 30 minutes
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

# Install required packages to a local directory
echo "Installing required packages..."
pip install --user transformers datasets numpy matplotlib seaborn pandas scipy cvxpy tqdm huggingface_hub bs4 requests psutil

# Configure environment variables for Hugging Face
export HF_HOME="./hf_cache"
export HF_HUB_CACHE="./hf_cache/hub"
export HF_DATASETS_CACHE="./hf_cache/datasets"
mkdir -p $HF_HOME $HF_HUB_CACHE $HF_DATASETS_CACHE

# Configure environment for better memory usage
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Create a minimalist test script to bypass the data loading
cat > run_minimal_test.py << 'EOF'
"""
Minimal test script to verify the code functionality without loading large datasets.
"""
import logging
import sys
import time
from pathlib import Path

# Import the necessary modules
from src.validation.temporal_inference import TemporalDistributionInference
from src.validation.statistical_validator import TemporalValidator
from src.validation.evaluation_metrics import TemporalEvaluationMetrics
from src.config import TIME_PERIODS, RESULTS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('minimal_test')

# Create simple test data with decade-specific vocabulary
test_data = {
    "1960s": [
        "This text from the 1960s mentions Apollo, lunar missions, Vietnam War, civil rights, and hippie counterculture.",
        "The 1960s were marked by the space race, cold war tensions, and the Beatles' popularity.",
        "In the 1960s, television became widespread with many households acquiring TV sets."
    ],
    "2000s": [
        "In the 2000s, technologies like Google, Facebook, and smartphones became popular.",
        "The 9/11 attacks in 2001 and the 2008 financial crisis shaped the 2000s decade.",
        "Mobile phones and the internet transformed communication in the 2000s."
    ]
}

def run_test():
    """Run a minimal test to verify the code functionality."""
    logger.info("Starting minimal pipeline test...")
    
    # Initialize components
    inference = TemporalDistributionInference(tokenizer_name="gpt2")
    evaluator = TemporalEvaluationMetrics()
    
    # Set up results directory
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Test analyzing decade patterns
    logger.info("Testing pattern analysis...")
    decade_patterns = inference.analyze_decade_patterns(test_data)
    
    if not decade_patterns:
        logger.error("Failed to generate patterns!")
        return False
    
    logger.info(f"Successfully generated patterns for {len(decade_patterns)} decades")
    
    # Test distribution inference
    logger.info("Testing distribution inference...")
    distribution = inference.infer_temporal_distribution(
        decade_patterns,
        remove_top_tokens=True,
        top_n=5
    )
    
    logger.info(f"Inferred distribution: {distribution}")
    
    # Test bootstrap validation
    logger.info("Testing bootstrap validation...")
    validator = TemporalValidator(
        inference_method=lambda texts: inference.infer_temporal_distribution(
            inference.analyze_decade_patterns(texts)
        )
    )
    
    confidence_intervals = validator.bootstrap_analysis(
        decade_texts=test_data,
        n_bootstrap=2,
        sample_ratio=0.8
    )
    
    logger.info(f"Generated confidence intervals: {confidence_intervals}")
    
    # Test evaluation metrics
    ground_truth = {"1960s": 0.5, "2000s": 0.5}
    evaluation = evaluator.evaluate_distribution(distribution, ground_truth, "gpt2")
    
    logger.info(f"Evaluation metrics: {evaluation['distribution_metrics']}")
    
    return True

if __name__ == "__main__":
    start_time = time.time()
    success = run_test()
    end_time = time.time()
    
    if success:
        logger.info(f"Test completed successfully in {end_time - start_time:.2f} seconds")
    else:
        logger.error(f"Test failed after {end_time - start_time:.2f} seconds")
EOF

# Run the minimal test script instead of the full pipeline
echo "Running minimal test script..."
python run_minimal_test.py

echo "Test completed at: $(date)"