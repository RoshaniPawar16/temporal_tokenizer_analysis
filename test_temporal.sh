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

# Update run_on_maxwell.py to use the test dataset override with ENHANCED OUTPUT
cat > test_mode_patch.py << 'EOF'
"""
Patch for run_on_maxwell.py to use test dataset override with ENHANCED OUTPUT.
"""
import sys
import os
import json
from pathlib import Path

# Add test dataset override to path
sys.path.insert(0, os.getcwd())
from test_dataset_override import create_test_dataset

# Add flag to detect this is being run by our test
os.environ['RUNNING_TEST_MODE'] = 'true'

# Main script execution continues from here
from run_on_maxwell import run_analysis
import argparse

# Create function to print section headers for better visibility
def print_section(title):
    """Print a clearly visible section header"""
    print("\n" + "="*80)
    print(f"  {title.upper()}")
    print("="*80)
    sys.stdout.flush()

# Create function to pretty print dictionaries
def print_dict(data, title=None):
    """Pretty print a dictionary"""
    if title:
        print(f"\n--- {title} ---")
    
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"  {k}: {v}")
    else:
        print(f"  {data}")
    
    sys.stdout.flush()

# Patch run_analysis function
def patched_run_analysis(args):
    """Patched run_analysis to use test dataset override with enhanced output."""
    print_section("STARTING TEST RUN")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Distribution: {args.distribution}")
    print(f"Test decades: {args.test_decades}")
    print(f"Texts per decade: {args.texts_per_decade}")
    
    import logging
    logger = logging.getLogger(__name__)
    
    # Import required modules with better error handling
    import time
    start_time = time.time()
    import run_on_maxwell
    
    try:
        # Configure logging and setup
        print_section("SETTING UP TEST ENVIRONMENT")
        log_filename = run_on_maxwell.configure_logging(args)
        print(f"Log file: {log_filename}")
        
        run_on_maxwell.limit_text_truncation_warnings('src.validation.temporal_inference')
        run_on_maxwell.limit_text_truncation_warnings('src.data.dataset_manager')
        results_dir = run_on_maxwell.setup_directories()
        print(f"Results directory: {results_dir}")
        
        # Parse test decades
        test_decades = [d.strip() for d in args.test_decades.split(",")]
        print(f"Using synthetic data for decades: {test_decades}")
        
        # Create test dataset with synthetic data - COMPLETELY BYPASS ANY REAL DATA LOADING
        print_section("CREATING SYNTHETIC TEST DATA")
        controlled_dataset = create_test_dataset(test_decades, args.texts_per_decade)
        
        # Get distributions
        print_section("CONFIGURING TEST DISTRIBUTIONS")
        distributions = run_on_maxwell.define_distributions()
        dist_info = distributions[args.distribution]
        
        print(f"Original distribution: {dist_info['distribution']}")
        
        # Normalize distribution to only include test decades
        selected_dist = {}
        total = 0
        for decade in test_decades:
            if decade in dist_info["distribution"]:
                selected_dist[decade] = dist_info["distribution"][decade]
                total += selected_dist[decade]
        
        # Normalize to sum to 1
        if total > 0:
            selected_dist = {decade: value/total for decade, value in selected_dist.items()}
        
        print(f"Test ground truth distribution: {selected_dist}")
        
        # Set up components
        print_section("INITIALIZING ANALYSIS COMPONENTS")
        from src.data.dataset_manager import TemporalDatasetManager
        dataset_manager = TemporalDatasetManager()
        
        from src.validation.temporal_inference import TemporalDistributionInference
        inference = TemporalDistributionInference(tokenizer_name=args.tokenizer)
        print(f"Using tokenizer: {inference.tokenizer_name}")
        print(f"Number of merge rules: {len(inference.merge_rules)}")
        
        from src.validation.statistical_validator import TemporalValidator
        validator = TemporalValidator(
            inference_method=lambda texts: inference.infer_temporal_distribution(
                inference.analyze_decade_patterns(texts)
            )
        )
        
        from src.validation.evaluation_metrics import TemporalEvaluationMetrics
        evaluator = TemporalEvaluationMetrics()
        
        # Extract text from tuples
        print_section("PREPARING TEXT DATA")
        decade_texts = {}
        for decade, texts in controlled_dataset.items():
            decade_texts[decade] = [item[0] if isinstance(item, tuple) else item for item in texts]
            print(f"  {decade}: {len(decade_texts[decade])} texts")
            
            # Print a sample of the first text
            if decade_texts[decade]:
                sample = decade_texts[decade][0][:100] + "..."
                print(f"  Sample: {sample}")
        
        # Process dataset
        chunked_decade_texts = {}
        for decade, texts in decade_texts.items():
            if not texts:
                continue
            
            chunked_decade_texts[decade] = texts
            print(f"  {decade}: {len(chunked_decade_texts[decade])} texts after chunking")
        
        # Run analysis with the professor's suggestion for top token removal
        print_section("ANALYZING DECADE PATTERNS")
        analysis_start = time.time()
        
        # Analyze decade patterns
        decade_patterns = inference.analyze_decade_patterns(chunked_decade_texts)
        
        print(f"Generated patterns for {len(decade_patterns)} decades")
        for decade, patterns in decade_patterns.items():
            if isinstance(patterns, dict):
                total_tokens = patterns.get('total_tokens', 0)
                merge_rules = len(patterns.get('merge_rules', {}))
                print(f"  {decade}: {total_tokens} total tokens, {merge_rules} merge rules")
                
                # Print top 5 most frequent merge rules
                if 'merge_rules' in patterns and patterns['merge_rules']:
                    top_rules = sorted(patterns['merge_rules'].items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"  Top merge rules:")
                    for rule, count in top_rules:
                        print(f"    {rule}: {count}")
        
        print_section("INFERRING TEMPORAL DISTRIBUTION (WITH TOP 5 TOKEN REMOVAL)")
        # Infer distribution with top 5 token removal as per professor's suggestion
        distribution = inference.infer_temporal_distribution(
            decade_patterns,
            remove_top_tokens=True,
            top_n=5  # Professor's suggestion
        )
        
        # Print the inferred distribution
        print("\nINFERRED DISTRIBUTION:")
        for decade, value in distribution.items():
            print(f"  {decade}: {value:.4f} ({value*100:.1f}%)")
        
        # Construct results
        results = {
            "tokenizer": args.tokenizer,
            "distribution": distribution,
            "distinctive_patterns": inference.find_distinctive_patterns(decade_patterns)
        }
        
        inference_time = time.time() - analysis_start
        print(f"\nInference completed in {inference_time:.2f} seconds")
        
        # Evaluate results
        print_section("EVALUATING RESULTS")
        evaluation = evaluator.evaluate_distribution(
            results["distribution"],
            selected_dist,
            model_name=args.tokenizer
        )
        
        # Print evaluation metrics
        print("EVALUATION METRICS:")
        dist_metrics = evaluation["distribution_metrics"]
        print(f"  log10(MSE): {dist_metrics['log10_mse']:.4f}")
        print(f"  MAE: {dist_metrics['mae']:.4f}")
        print(f"  Jensen-Shannon Distance: {dist_metrics['js_distance']:.4f}")
        print(f"  Rank Correlation: {evaluation['decade_metrics']['rank_correlation']:.4f}")
        
        # Print representation analysis
        rep_analysis = evaluation["decade_metrics"]["representation_analysis"]
        if rep_analysis["over_represented"]:
            print("\nOVER-REPRESENTED DECADES:")
            for decade, value in sorted(rep_analysis["over_represented"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {decade}: +{value:.1%}")
                
        if rep_analysis["under_represented"]:
            print("\nUNDER-REPRESENTED DECADES:")
            for decade, value in sorted(rep_analysis["under_represented"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {decade}: -{value:.1%}")
        
        # Save and visualize results
        print_section("SAVING RESULTS")
        run_id = f"{args.tokenizer}_{args.distribution}_{run_on_maxwell.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_on_maxwell.save_distribution_results(results, evaluation, run_id, results_dir)
        print(f"Results saved with ID: {run_id}")
        
        # Show file paths
        dist_path = results_dir / "distributions" / f"{run_id}_distribution.json"
        patterns_path = results_dir / "distributions" / f"{run_id}_patterns.json"
        print(f"Distribution file: {dist_path}")
        print(f"Patterns file: {patterns_path}")
        
        # Load and print the saved distribution file for verification
        if dist_path.exists():
            try:
                with open(dist_path, 'r') as f:
                    saved_data = json.load(f)
                    print("\nSAVED DISTRIBUTION (from file):")
                    saved_dist = saved_data.get("distribution", {})
                    for decade, value in saved_dist.items():
                        print(f"  {decade}: {value:.4f} ({value*100:.1f}%)")
            except Exception as e:
                print(f"Error reading saved distribution: {e}")
        
        # Create visualizations
        print_section("CREATING VISUALIZATIONS")
        run_on_maxwell.create_comparison_visualizations(
            results["distribution"], 
            selected_dist, 
            args.distribution, 
            args.tokenizer, 
            results_dir
        )
        
        # Print figure paths
        comp_fig = results_dir / "figures" / f"{args.tokenizer}_{args.distribution}_comparison.png"
        error_fig = results_dir / "figures" / f"{args.tokenizer}_{args.distribution}_error.png"
        print(f"Comparison visualization: {comp_fig}")
        print(f"Error visualization: {error_fig}")
        
        # Bootstrap validation
        if args.bootstrap:
            print_section("PERFORMING BOOTSTRAP VALIDATION")
            print(f"Running {args.bootstrap_iterations} bootstrap iterations...")
            
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
                
                # Print bootstrap results
                print("\nBOOTSTRAP RESULTS (95% CONFIDENCE INTERVALS):")
                for decade, stats in confidence_intervals.items():
                    mean = stats.get("mean", 0)
                    lower = stats.get("lower_ci", 0)
                    upper = stats.get("upper_ci", 0)
                    print(f"  {decade}: {mean:.4f} ({mean*100:.1f}%), CI [{lower:.4f}, {upper:.4f}]")
                
                # Save bootstrap results
                bootstrap_path = results_dir / "bootstrap" / f"{run_id}_bootstrap.json"
                with open(bootstrap_path, 'w') as f:
                    ci_json = {}
                    for decade, stats in confidence_intervals.items():
                        ci_json[decade] = {k: float(v) for k, v in stats.items() if not isinstance(v, list)}
                    json.dump(ci_json, f, indent=2)
                
                print(f"Bootstrap results saved to: {bootstrap_path}")
                
                # Visualize
                bootstrap_fig = results_dir / "figures" / f"{args.tokenizer}_{args.distribution}_bootstrap.png"
                run_on_maxwell.create_bootstrap_visualization(
                    results["distribution"], 
                    selected_dist,
                    confidence_intervals, 
                    args.distribution,
                    args.tokenizer, 
                    results_dir
                )
                print(f"Bootstrap visualization: {bootstrap_fig}")
                
            except Exception as e:
                print(f"Error in bootstrap validation: {e}")
                import traceback
                traceback.print_exc()
        
        # Completion summary
        total_time = time.time() - start_time
        print_section("TEST COMPLETED SUCCESSFULLY")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Result files located in: {results_dir}")
        
    except Exception as e:
        print_section("ERROR OCCURRED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Replace run_analysis with our patched version
import run_on_maxwell
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

# Create arguments with both decades to test
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

# Run the modified test with more detailed output
echo "Running test for removing top tokens..."
python test_mode_patch.py

echo "Test completed at: $(date)"