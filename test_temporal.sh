#!/bin/bash
#SBATCH --job-name=test_temporal
#SBATCH --time=00:45:00
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

# Create a file to override the dataset loading for test mode
cat > test_dataset_override.py << 'EOF'
"""
Test dataset override to provide synthetic data that follows the professor's recommendations.
"""
import logging
import random
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def create_test_dataset(decades, texts_per_decade=10):
    """Create a minimal test dataset with decade-specific vocabulary."""
    logger.info(f"Creating synthetic test dataset for {len(decades)} decades with {texts_per_decade} texts each")
    
    # Decade-specific vocabulary - use MORE distinctive vocabulary patterns
    decade_vocab = {
        "1950s": ["atomic", "television", "radio", "nuclear", "Soviet", "space race",
                 "hydrogen bomb", "satellite", "automation", "transistor radio"],
        "1960s": ["Apollo", "lunar", "Vietnam War", "civil rights", "hippie", "counterculture",
                 "microchip", "women's liberation", "mainframe", "NASA"],
        "1970s": ["computerized", "digital", "microprocessor", "environmentalism", 
                 "floppy disk", "pocket calculator", "video game", "oil crisis",
                 "punk rock", "Star Wars"],
        "1980s": ["personal computer", "IBM PC", "Apple Macintosh", "MS-DOS", 
                 "MTV", "VHS", "Walkman", "compact disc", "fax machine", "mobile phone"],
        "1990s": ["Internet", "World Wide Web", "email", "dot-com", "website", "browser", 
                 "Windows 95", "modem", "chat room", "DVD"],
        "2000s": ["smartphone", "Google", "Facebook", "social media", "blog", "Wikipedia", 
                 "YouTube", "broadband", "iPod", "Wi-Fi"],
        "2010s": ["social networking", "smartphone app", "tablet", "streaming", "cloud computing", 
                 "Bitcoin", "artificial intelligence", "machine learning", "Instagram", "Twitter"],
        "2020s": ["pandemic", "COVID-19", "Zoom", "remote work", "blockchain", "NFT", 
                 "cryptocurrency", "TikTok", "climate crisis", "vaccine"]
    }
    
    # Create dataset
    controlled_dataset = {}
    
    for decade in decades:
        decade_texts = []
        vocab = decade_vocab.get(decade, ["generic term 1", "generic term 2"])
        
        for i in range(texts_per_decade):
            # Create a text with decade-specific vocabulary to ensure distinct patterns
            text = f"This is test text {i} from the {decade}. "
            text += "The following terms were common in this era: "
            
            # Add several decade-specific terms
            decade_terms = random.sample(vocab, min(5, len(vocab)))
            text += ", ".join(decade_terms)
            
            # Add more content to make texts substantial
            text += f" During the {decade}, these concepts were especially important. "
            text += "People often discussed these ideas in literature and media. "
            text += f"The {decade} represented a time of significant change in society. "
            text += "Technological and cultural developments shaped this period. "
            
            # Add some more decade terms to reinforce the patterns
            more_terms = random.sample(vocab, min(3, len(vocab)))
            text += f"Other notable concepts included {', '.join(more_terms)}."
            
            # Ensure each text has multiple instances of each pattern
            for term in random.sample(vocab, min(2, len(vocab))):
                text += f" The concept of {term} was particularly significant during this time period. "
            
            decade_texts.append((text, "test_synthetic"))
        
        controlled_dataset[decade] = decade_texts
        logger.info(f"Created {len(decade_texts)} synthetic texts for {decade}")
    
    return controlled_dataset
EOF

# Update the test_mode_patch.py file to completely skip data loading
cat > test_mode_patch.py << 'EOF'
"""
Patch for run_on_maxwell.py to use test dataset override.
"""
import sys
import os
from pathlib import Path

# Add test dataset override to path
sys.path.insert(0, os.getcwd())
from test_dataset_override import create_test_dataset

# Add flag to detect this is being run by our test
os.environ['RUNNING_TEST_MODE'] = 'true'

# Main script execution continues from here
import run_on_maxwell
from run_on_maxwell import run_analysis, configure_logging, setup_directories, limit_text_truncation_warnings
import argparse

# Save original function for later
original_run_analysis = run_on_maxwell.run_analysis

def patched_run_analysis(args):
    """Patched run_analysis to use test dataset override."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Configure logging and setup
    log_filename = run_on_maxwell.configure_logging(args)
    run_on_maxwell.limit_text_truncation_warnings('src.validation.temporal_inference')
    run_on_maxwell.limit_text_truncation_warnings('src.data.dataset_manager')
    results_dir = run_on_maxwell.setup_directories()
    
    # Parse test decades
    test_decades = [d.strip() for d in args.test_decades.split(",")]
    logger.info(f"TEST PATCH: Using synthetic data for {test_decades}")
    
    # Create test dataset with synthetic data - COMPLETELY BYPASS ANY REAL DATA LOADING
    controlled_dataset = create_test_dataset(test_decades, args.texts_per_decade)
    
    # Get distributions
    distributions = run_on_maxwell.define_distributions()
    dist_info = distributions[args.distribution]
    selected_dist = dist_info["distribution"]
    
    # Normalize distribution to only include test decades
    modified_dist = {}
    total = 0
    for decade in test_decades:
        if decade in selected_dist:
            modified_dist[decade] = selected_dist[decade]
            total += modified_dist[decade]
    
    # Normalize to sum to 1
    if total > 0:
        selected_dist = {decade: value/total for decade, value in modified_dist.items()}
    
    logger.info(f"TEST PATCH: Using distribution: {selected_dist}")
    
    # Set up components - USE DIRECT IMPORTS TO AVOID LOADING FULL MODULES
    from src.data.dataset_manager import TemporalDatasetManager
    dataset_manager = TemporalDatasetManager()
    
    from src.validation.temporal_inference import TemporalDistributionInference
    inference = TemporalDistributionInference(tokenizer_name=args.tokenizer)
    
    from src.validation.statistical_validator import TemporalValidator
    validator = TemporalValidator(
        inference_method=lambda texts: inference.infer_temporal_distribution(
            inference.analyze_decade_patterns(texts)
        )
    )
    
    from src.validation.evaluation_metrics import TemporalEvaluationMetrics
    evaluator = TemporalEvaluationMetrics()
    
    # Extract text from tuples
    decade_texts = {}
    for decade, texts in controlled_dataset.items():
        decade_texts[decade] = [item[0] if isinstance(item, tuple) else item for item in texts]
    
    # Process dataset
    chunked_decade_texts = {}
    for decade, texts in decade_texts.items():
        if not texts:
            continue
            
        # Use simplified chunking for test
        chunked_decade_texts[decade] = texts[:5]  # Just use the first 5
        logger.info(f"Using {len(chunked_decade_texts[decade])} texts for {decade}")
    
    # Run analysis with the professor's suggestion for top token removal
    import time
    start_time = time.time()
    
    # Analyze decade patterns
    logger.info("Analyzing decade patterns...")
    decade_patterns = inference.analyze_decade_patterns(chunked_decade_texts)
    
    # Infer distribution with top 5 token removal as per professor's suggestion
    logger.info("Inferring distribution with removal of top 5 tokens...")
    distribution = inference.infer_temporal_distribution(
        decade_patterns,
        remove_top_tokens=True,
        top_n=5  # Professor's suggestion
    )
    
    # Construct results
    results = {
        "tokenizer": args.tokenizer,
        "distribution": distribution,
        "distinctive_patterns": inference.find_distinctive_patterns(decade_patterns)
    }
    
    inference_time = time.time() - start_time
    
    # Evaluate results
    logger.info("Evaluating results...")
    evaluation = evaluator.evaluate_distribution(
        results["distribution"],
        selected_dist,
        model_name=args.tokenizer
    )
    
    # Save and visualize results
    run_on_maxwell.save_distribution_results(results, evaluation, f"{args.tokenizer}_{args.distribution}_{run_on_maxwell.datetime.now().strftime('%Y%m%d_%H%M%S')}", results_dir)
    run_on_maxwell.log_evaluation_metrics(evaluation, inference_time, args)
    run_on_maxwell.create_comparison_visualizations(results["distribution"], selected_dist, args.distribution, args.tokenizer, results_dir)
    
    # Bootstrap validation
    if args.bootstrap:
        logger.info(f"Performing bootstrap validation with {args.bootstrap_iterations} iterations...")
        
        try:
            # Create the safe inference wrapper
            safe_inference_wrapper = run_on_maxwell.create_inference_wrapper(inference)
            bootstrap_validator = TemporalValidator(inference_method=safe_inference_wrapper)
            
            # Run bootstrap
            confidence_intervals = bootstrap_validator.bootstrap_analysis(
                decade_texts=controlled_dataset,
                n_bootstrap=args.bootstrap_iterations,
                sample_ratio=0.8
            )
            
            # Save bootstrap results
            bootstrap_path = results_dir / "bootstrap" / f"{args.tokenizer}_{args.distribution}_{run_on_maxwell.datetime.now().strftime('%Y%m%d_%H%M%S')}_bootstrap.json"
            with open(bootstrap_path, 'w') as f:
                import json
                ci_json = {}
                for decade, stats in confidence_intervals.items():
                    ci_json[decade] = {k: float(v) for k, v in stats.items() if not isinstance(v, list)}
                json.dump(ci_json, f, indent=2)
            
            # Visualize
            run_on_maxwell.create_bootstrap_visualization(
                results["distribution"], 
                selected_dist,
                confidence_intervals, 
                args.distribution,
                args.tokenizer, 
                results_dir
            )
            
        except Exception as e:
            logger.error(f"Error in bootstrap validation: {e}")
    
    logger.info(f"TEST PATCH: Analysis completed successfully")

# Replace run_analysis with our patched version
run_on_maxwell.run_analysis = patched_run_analysis

# Create parser with test arguments
parser = argparse.ArgumentParser(description="Run temporal distribution inference test")
parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to analyze")
parser.add_argument("--distribution", type=str, default="uniform", help="Distribution pattern to test")
parser.add_argument("--test_mode", action="store_true", help="Run in test mode with minimal data")
parser.add_argument("--test_size_mb", type=float, default=1.0, help="Size of test data in MB per decade")
parser.add_argument("--test_decades", type=str, default="1960s,2000s", help="Comma-separated list of decades to use")
parser.add_argument("--texts_per_decade", type=int, default=5, help="Number of texts per decade")
parser.add_argument("--bootstrap", action="store_true", help="Perform bootstrap validation")
parser.add_argument("--bootstrap_iterations", type=int, default=2, help="Number of bootstrap iterations")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
parser.add_argument("--allow_synthetic_fallback", action="store_true", help="Allow synthetic fallback for missing decades")
parser.add_argument("--force_fresh", action="store_true", help="Force fresh dataset creation")
parser.add_argument("--force_quality", action="store_true", help="Force quality requirements")
parser.add_argument("--apply_enhancements", action="store_true", help="Apply targeted enhancements")
parser.add_argument("--target_size_gb", type=float, default=0.005, help="Target size in GB")

args = parser.parse_args([
    "--tokenizer", "gpt2",
    "--distribution", "uniform",
    "--test_mode",
    "--test_size_mb", "1",
    "--test_decades", "1960s,2000s",
    "--texts_per_decade", "5",
    "--bootstrap",
    "--bootstrap_iterations", "2",
    "--allow_synthetic_fallback",
    "--verbose"
])

# Run the patched analysis
run_analysis(args)
EOF

# Run the modified test with shorter timeout
echo "Running test for removing top tokens..."
python test_mode_patch.py

echo "Test completed at: $(date)"