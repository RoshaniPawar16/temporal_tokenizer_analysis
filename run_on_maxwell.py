"""
Main script for running temporal distribution inference on Maxwell HPC.

This script handles dataset creation, analysis, and evaluation with 
various distribution patterns and tokenizers.
"""

import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from datetime import datetime

from src.data.dataset_manager import TemporalDatasetManager
from src.validation.temporal_inference import TemporalDistributionInference
from src.validation.statistical_validator import TemporalValidator
from src.validation.evaluation_metrics import TemporalEvaluationMetrics
from src.config import TIME_PERIODS, RESULTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Set up visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

def setup_directories():
    """Create necessary directories for results and figures."""
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different result types
    (results_dir / "distributions").mkdir(exist_ok=True)
    (results_dir / "figures").mkdir(exist_ok=True)
    (results_dir / "metrics").mkdir(exist_ok=True)
    (results_dir / "bootstrap").mkdir(exist_ok=True)
    
    return results_dir

def define_distributions():
    """Define test distributions for evaluation."""
    return {
        "uniform": {
            "name": "Uniform Distribution",
            "description": "Equal representation across all decades",
            "distribution": {decade: 1.0/len(TIME_PERIODS) for decade in TIME_PERIODS.keys()}
        },
        "recency_bias": {
            "name": "Recency Bias",
            "description": "Higher representation for recent decades",
            "distribution": {
                "1950s": 0.05, "1960s": 0.05, "1970s": 0.10, "1980s": 0.10,
                "1990s": 0.15, "2000s": 0.20, "2010s": 0.25, "2020s": 0.10
            }
        },
        "historical_bias": {
            "name": "Historical Bias",
            "description": "Higher representation for older decades",
            "distribution": {
                "1950s": 0.25, "1960s": 0.20, "1970s": 0.15, "1980s": 0.10,
                "1990s": 0.10, "2000s": 0.10, "2010s": 0.05, "2020s": 0.05
            }
        },
        "bimodal": {
            "name": "Bimodal Distribution",
            "description": "Peaks in earliest and latest decades",
            "distribution": {
                "1950s": 0.20, "1960s": 0.10, "1970s": 0.05, "1980s": 0.05,
                "1990s": 0.05, "2000s": 0.05, "2010s": 0.20, "2020s": 0.30
            }
        }
    }

def run_analysis(args):
    """Run the complete analysis with specified parameters."""
    # Set up directories
    results_dir = setup_directories()
    
    # Get distributions
    distributions = define_distributions()
    
    # Validate distribution choice
    if args.distribution not in distributions:
        logger.error(f"Unknown distribution: {args.distribution}")
        logger.info(f"Available distributions: {list(distributions.keys())}")
        return
    
    # Get selected distribution
    dist_info = distributions[args.distribution]
    selected_dist = dist_info["distribution"]
    
    logger.info(f"Running analysis for {dist_info['name']} with {args.tokenizer} tokenizer")
    logger.info(f"Using {args.texts_per_decade} texts per decade")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.tokenizer}_{args.distribution}_{timestamp}"
    
    # Initialize components
    dataset_manager = TemporalDatasetManager()
    inference = TemporalDistributionInference(tokenizer_name=args.tokenizer)
    validator = TemporalValidator(
        inference_method=lambda texts: inference.infer_temporal_distribution(
            inference.analyze_decade_patterns(texts)
        )
    )
    evaluator = TemporalEvaluationMetrics()
    
    # Create controlled dataset
    logger.info("Creating controlled dataset...")
    controlled_dataset = dataset_manager.create_controlled_dataset(
        distribution=selected_dist,
        total_texts=args.texts_per_decade * len(selected_dist)
    )
    
    # Extract just texts (without source info)
    decade_texts = {decade: [text for text, _ in texts] 
                   for decade, texts in controlled_dataset.items()}
    
    # Run inference
    logger.info("Running tokenizer analysis...")
    start_time = time.time()
    results = inference.run_analysis(decade_texts)
    inference_time = time.time() - start_time
    
    # Evaluate results
    logger.info("Evaluating results...")
    evaluation = evaluator.evaluate_distribution(
        results["distribution"],
        selected_dist,
        model_name=args.tokenizer
    )
    
    # Save detailed results
    save_distribution_results(results, evaluation, run_id, results_dir)
    
    # Output evaluation metrics
    log_evaluation_metrics(evaluation, inference_time, args)
    
    # Create comparison visualizations
    create_comparison_visualizations(results["distribution"], selected_dist, 
                                   args.distribution, args.tokenizer, results_dir)
    
    # Bootstrap validation (if requested)
    if args.bootstrap:
        logger.info(f"Performing bootstrap validation with {args.bootstrap_iterations} iterations...")
        confidence_intervals = validator.bootstrap_analysis(
            decade_texts=decade_texts,
            n_bootstrap=args.bootstrap_iterations,
            sample_ratio=0.8
        )
        
        # Save bootstrap results
        bootstrap_path = results_dir / "bootstrap" / f"{run_id}_bootstrap.json"
        with open(bootstrap_path, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            ci_json = {}
            for decade, stats in confidence_intervals.items():
                ci_json[decade] = {k: float(v) for k, v in stats.items()}
            json.dump(ci_json, f, indent=2)
        
        # Visualize with confidence intervals
        create_bootstrap_visualization(results["distribution"], selected_dist, 
                                     confidence_intervals, args.distribution, 
                                     args.tokenizer, results_dir)
        
        # Calculate reliability metrics
        reliability_metrics = calculate_reliability_metrics(confidence_intervals)
        logger.info(f"Reliability metrics:")
        logger.info(f"  Reliability score: {reliability_metrics['reliability_score']:.1f}/100")
        logger.info(f"  Coefficient of variation: {reliability_metrics['coefficient_of_variation']:.2f}")
        logger.info(f"  Normalized CI width: {reliability_metrics['normalized_ci_width']:.2f}")
    
    logger.info(f"Analysis completed for {args.distribution} with {args.tokenizer}")
    
def save_distribution_results(results, evaluation, run_id, results_dir):
    """Save detailed analysis results."""
    # Save inferred distribution
    dist_path = results_dir / "distributions" / f"{run_id}_distribution.json"
    with open(dist_path, 'w') as f:
        json.dump({
            "tokenizer": results["tokenizer"],
            "distribution": {k: float(v) for k, v in results["distribution"].items()},
            "evaluation": {
                "log10_mse": float(evaluation["distribution_metrics"]["log10_mse"]),
                "mae": float(evaluation["distribution_metrics"]["mae"]),
                "js_distance": float(evaluation["distribution_metrics"]["js_distance"]),
                "rank_correlation": float(evaluation["decade_metrics"]["rank_correlation"])
            }
        }, f, indent=2)
    
    # Save distinctive patterns
    patterns_path = results_dir / "distributions" / f"{run_id}_patterns.json"
    with open(patterns_path, 'w') as f:
        # Convert tuples to lists for JSON serialization
        patterns_json = {}
        for decade, patterns in results["distinctive_patterns"].items():
            patterns_json[decade] = [[p, float(s)] for p, s in patterns]
        json.dump(patterns_json, f, indent=2)

def log_evaluation_metrics(evaluation, inference_time, args):
    """Log detailed evaluation metrics."""
    logger.info(f"Evaluation results for {args.tokenizer} on {args.distribution} distribution:")
    logger.info(f"  log10(MSE): {evaluation['distribution_metrics']['log10_mse']:.2f}")
    logger.info(f"  MAE: {evaluation['distribution_metrics']['mae']:.4f}")
    logger.info(f"  Jensen-Shannon Distance: {evaluation['distribution_metrics']['js_distance']:.4f}")
    logger.info(f"  Rank Correlation: {evaluation['decade_metrics']['rank_correlation']:.2f}")
    logger.info(f"  Inference Time: {inference_time:.2f} seconds")
    
    # Get over/under represented decades
    rep_analysis = evaluation["decade_metrics"]["representation_analysis"]
    if rep_analysis["over_represented"]:
        over_rep = sorted(rep_analysis["over_represented"].items(), key=lambda x: x[1], reverse=True)
        logger.info("  Over-represented decades:")
        for decade, value in over_rep[:3]:  # Top 3
            logger.info(f"    {decade}: +{value:.1%}")
            
    if rep_analysis["under_represented"]:
        under_rep = sorted(rep_analysis["under_represented"].items(), key=lambda x: x[1], reverse=True)
        logger.info("  Under-represented decades:")
        for decade, value in under_rep[:3]:  # Top 3
            logger.info(f"    {decade}: -{value:.1%}")

def create_comparison_visualizations(inferred, ground_truth, dist_name, tokenizer_name, results_dir):
    """Create visualizations comparing inferred and ground truth distributions."""
    # Sort decades chronologically
    decades = sorted(set(inferred.keys()) | set(ground_truth.keys()))
    
    # Create figure for bar chart comparison
    plt.figure(figsize=(12, 6))
    
    # Set bar width and positions
    bar_width = 0.35
    r1 = np.arange(len(decades))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    inferred_values = [inferred.get(decade, 0) for decade in decades]
    truth_values = [ground_truth.get(decade, 0) for decade in decades]
    
    plt.bar(r1, inferred_values, width=bar_width, label='Inferred', color='skyblue', alpha=0.8)
    plt.bar(r2, truth_values, width=bar_width, label='Ground Truth', color='lightcoral', alpha=0.8)
    
    # Add data labels
    for i, v in enumerate(inferred_values):
        plt.text(i, v + 0.01, f"{v:.1%}", ha='center', fontsize=9)
    for i, v in enumerate(truth_values):
        plt.text(i + bar_width, v + 0.01, f"{v:.1%}", ha='center', fontsize=9)
    
    # Add labels and title
    plt.xlabel('Decade')
    plt.ylabel('Proportion')
    plt.title(f'Inferred vs Ground Truth: {dist_name}')
    plt.xticks([r + bar_width/2 for r in r1], decades, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(results_dir / "figures" / f"{tokenizer_name}_{dist_name}_comparison.png", dpi=300)
    plt.close()
    
    # Create absolute error visualization
    plt.figure(figsize=(12, 6))
    
    # Calculate absolute errors
    errors = [abs(inferred.get(decade, 0) - ground_truth.get(decade, 0)) for decade in decades]
    
    # Create color-coded bars based on error magnitude
    colors = plt.cm.RdYlGn_r(np.array(errors) / max(errors) if max(errors) > 0 else np.zeros(len(errors)))
    plt.bar(decades, errors, color=colors)
    
    # Add data labels
    for i, v in enumerate(errors):
        plt.text(i, v + 0.005, f"{v:.1%}", ha='center')
    
    # Add labels and title
    plt.xlabel('Decade')
    plt.ylabel('Absolute Error')
    plt.title(f'Absolute Error by Decade: {dist_name}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(results_dir / "figures" / f"{tokenizer_name}_{dist_name}_error.png", dpi=300)
    plt.close()

def create_bootstrap_visualization(inferred_distribution, ground_truth_distribution, confidence_intervals, output_path=None):
    """Create visualization of bootstrap results with confidence intervals."""
    plt.figure(figsize=(14, 7))
    
    # Sort decades chronologically
    decades = sorted(inferred_distribution.keys())
    
    # Extract data
    means = [inferred_distribution.get(d, 0) for d in decades]
    lower = [confidence_intervals.get(d, {}).get('lower_ci', means[i] * 0.8) for i, d in enumerate(decades)]
    upper = [confidence_intervals.get(d, {}).get('upper_ci', means[i] * 1.2) for i, d in enumerate(decades)]
    
    # Calculate error bars AS POSITIVE DISTANCES (this is the key fix)
    errors_lower = [max(0, means[i] - lower[i]) for i in range(len(means))]  # Ensure positive
    errors_upper = [max(0, upper[i] - means[i]) for i in range(len(means))]  # Ensure positive
    
    # Plot with confidence intervals - specify yerr as a 2xN array for asymmetric errors
    plt.bar(
        decades,
        means,
        alpha=0.7,
        color='skyblue',
        yerr=[errors_lower, errors_upper],  # This format expects positive values
        capsize=5,
        label="Bootstrap Estimate"
    )
    
    # Add ground truth as points
    plt.plot(decades, [ground_truth.get(d, 0) for d in decades], 'ro', label="Ground Truth")
    
    # Add data labels
    for i, v in enumerate(means):
        plt.text(i, v + 0.01, f"{v:.1%}", ha='center')
    
    # Add title and labels
    plt.title(f'Temporal Distribution with Confidence Intervals: {dist_name}')
    plt.xlabel('Decade')
    plt.ylabel('Estimated Proportion')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(results_dir / "figures" / f"{tokenizer_name}_{dist_name}_bootstrap.png", dpi=300)
    plt.close()

def calculate_reliability_metrics(confidence_intervals):
    """Calculate metrics to assess the reliability of the statistical analysis."""
    if not confidence_intervals:
        return {"reliability_score": 0, "coefficient_of_variation": 1.0, "normalized_ci_width": 1.0}
    
    # Calculate coefficient of variation (CV) for each decade
    cv_values = []
    ci_widths = []
    
    for decade, stats in confidence_intervals.items():
        mean = stats.get("mean", 0)
        if mean > 0:
            std_dev = stats.get("std_dev", 0)
            cv = std_dev / mean if mean > 0 else 1.0
            cv_values.append(cv)
            
            # Calculate normalized confidence interval width
            lower = stats.get("lower_ci", 0)
            upper = stats.get("upper_ci", 1)
            width = (upper - lower) / mean if mean > 0 else 1.0
            ci_widths.append(width)
    
    # Calculate average metrics
    avg_cv = sum(cv_values) / len(cv_values) if cv_values else 1.0
    avg_ci_width = sum(ci_widths) / len(ci_widths) if ci_widths else 1.0
    
    # Calculate overall reliability score (higher is better)
    # Score ranges from 0-100, with penalties for high CV and wide CIs
    cv_penalty = min(50, 50 * avg_cv)
    width_penalty = min(50, 50 * avg_ci_width / 2)  # Normalize by expected width
    reliability_score = 100 - cv_penalty - width_penalty
    
    return {
        "reliability_score": reliability_score,
        "coefficient_of_variation": avg_cv,
        "normalized_ci_width": avg_ci_width
    }

def compare_all_distributions(args):
    """Run analysis on all distributions and create comparison visualizations."""
    distributions = define_distributions()
    results_by_dist = {}
    
    # Run analysis for each distribution
    for dist_name in distributions:
        # Copy args and update distribution
        dist_args = argparse.Namespace(**vars(args))
        dist_args.distribution = dist_name
        
        # Run analysis and collect results
        logger.info(f"Running analysis for {dist_name}...")
        run_analysis(dist_args)
        
        # Load results from saved files
        results_dir = setup_directories()
        timestamp = datetime.now().strftime("%Y%m%d")  # Just use today's date
        result_files = list((results_dir / "distributions").glob(f"{args.tokenizer}_{dist_name}_*_distribution.json"))
        
        if result_files:
            # Use the most recent file
            result_file = sorted(result_files)[-1]
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                results_by_dist[dist_name] = {
                    "distribution": result_data["distribution"],
                    "evaluation": result_data["evaluation"]
                }
    
    # Create comparative visualizations
    if len(results_by_dist) > 1:
        logger.info("Creating comparative visualizations...")
        create_distribution_comparison(results_by_dist, distributions, args.tokenizer, setup_directories())

def create_distribution_comparison(results_by_dist, distributions, tokenizer_name, results_dir):
    """Create visualizations comparing results across different distributions."""
    # Extract metrics for comparison
    dist_names = list(results_by_dist.keys())
    log_mse_values = [results_by_dist[d]["evaluation"]["log10_mse"] for d in dist_names]
    mae_values = [results_by_dist[d]["evaluation"]["mae"] for d in dist_names]
    js_values = [results_by_dist[d]["evaluation"]["js_distance"] for d in dist_names]
    correlation_values = [results_by_dist[d]["evaluation"]["rank_correlation"] for d in dist_names]
    
    # Create figure with 2x2 subplots for metrics comparison
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    # Plot log10(MSE)
    axs[0].bar(dist_names, log_mse_values, color='royalblue')
    axs[0].set_title('log10(MSE) by Distribution Pattern\n(lower is better)')
    axs[0].set_ylabel('log10(MSE)')
    # Add Hayase benchmark line
    axs[0].axhline(y=-7.3, color='red', linestyle='--', 
                 label='Hayase benchmark: -7.3')
    axs[0].legend()
    
    # Plot Mean Absolute Error
    axs[1].bar(dist_names, mae_values, color='royalblue')
    axs[1].set_title('Mean Absolute Error by Distribution Pattern\n(lower is better)')
    axs[1].set_ylabel('MAE')
    
    # Plot Jensen-Shannon Distance
    axs[2].bar(dist_names, js_values, color='royalblue')
    axs[2].set_title('Jensen-Shannon Distance by Distribution Pattern\n(lower is better)')
    axs[2].set_ylabel('Jensen-Shannon Distance')
    
    # Plot Rank Correlation
    axs[3].bar(dist_names, correlation_values, color='royalblue')
    axs[3].set_title('Rank Correlation by Distribution Pattern\n(higher is better)')
    axs[3].set_ylabel('Rank Correlation')
    
    # Add labels and adjust layout
    for ax in axs:
        ax.set_xticklabels(dist_names, rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / f"{tokenizer_name}_metric_comparison.png", dpi=300)
    plt.close()
    
    # Create error by decade comparison
    plt.figure(figsize=(14, 8))
    
    # Get all decades across all distributions
    all_decades = set()
    for dist_name in dist_names:
        inferred = results_by_dist[dist_name]["distribution"]
        ground_truth = distributions[dist_name]["distribution"]
        all_decades.update(set(inferred.keys()) | set(ground_truth.keys()))
    
    # Sort decades chronologically
    decades = sorted(all_decades)
    
    # Calculate errors for each distribution
    for i, dist_name in enumerate(dist_names):
        inferred = results_by_dist[dist_name]["distribution"]
        ground_truth = distributions[dist_name]["distribution"]
        
        errors = [abs(inferred.get(decade, 0) - ground_truth.get(decade, 0)) for decade in decades]
        plt.plot(decades, errors, 'o-', label=dist_name, color=plt.cm.tab10(i))
    
    # Add labels and title
    plt.xlabel('Decade')
    plt.ylabel('Absolute Error')
    plt.title('Error by Decade Across Distribution Patterns')
    plt.xticks(rotation=45)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(results_dir / "figures" / f"{tokenizer_name}_error_comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run temporal distribution inference on Maxwell")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to analyze")
    parser.add_argument("--texts_per_decade", type=int, default=1000, 
                      help="Number of texts per decade (higher = more accurate)")
    parser.add_argument("--distribution", type=str, default="uniform", 
                      choices=["uniform", "recency_bias", "historical_bias", "bimodal", "all"],
                      help="Distribution pattern to test (use 'all' to run all patterns)")
    parser.add_argument("--bootstrap", action="store_true", 
                      help="Perform bootstrap validation for confidence intervals")
    parser.add_argument("--bootstrap_iterations", type=int, default=100,
                      help="Number of bootstrap iterations to perform (higher = more reliable)")
    
    args = parser.parse_args()
    
    # Run all distributions or just the specified one
    if args.distribution == "all":
        compare_all_distributions(args)
    else:
        run_analysis(args)