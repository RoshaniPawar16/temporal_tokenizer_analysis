"""
Main analysis script for temporal distribution inference.
Designed for resource-constrained environments with enhanced evaluation capabilities.
"""

import logging
from pathlib import Path
import time
import json
import argparse

from src.data.dataset_manager import TemporalDatasetManager
from src.validation.temporal_inference import TemporalDistributionInference
from src.validation.statistical_validator import TemporalValidator
from src.validation.evaluation_metrics import TemporalEvaluationMetrics
from src.config import RESULTS_DIR, TIME_PERIODS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_analysis(tokenizer_name: str = "gpt2", 
                texts_per_decade: int = 5,
                ground_truth: dict = None,
                bootstrap_iterations: int = 20):
    """
    Run complete temporal analysis with statistical validation and evaluation.
    
    Args:
        tokenizer_name: Name of the tokenizer to analyze
        texts_per_decade: Number of texts to use per decade
        ground_truth: Optional ground truth distribution for evaluation
        bootstrap_iterations: Number of bootstrap iterations for statistical validation
    
    Returns:
        Complete analysis results
    """
    start_time = time.time()
    logger.info(f"Starting analysis for {tokenizer_name}")
    
    # Step 1: Load or create dataset
    logger.info(f"Loading temporal dataset with {texts_per_decade} texts per decade...")
    manager = TemporalDatasetManager()
    
    # Try to load existing dataset first
    dataset = manager.load_dataset()
    
    # If dataset is empty or insufficient, build a new one
    if not dataset or sum(len(texts) for texts in dataset.values()) < texts_per_decade * len(TIME_PERIODS):
        logger.info("Building new dataset...")
        combined_dataset = manager.build_temporal_dataset(
            texts_per_decade=texts_per_decade,
            save_dataset=True
        )
        
        # Extract texts from combined dataset
        dataset = {decade: [text for text, _ in texts] 
                  for decade, texts in combined_dataset.items()}
    
    # Log dataset statistics
    non_empty_decades = {decade: texts for decade, texts in dataset.items() if texts}
    logger.info(f"Dataset contains {sum(len(texts) for texts in non_empty_decades.values())} texts across {len(non_empty_decades)} decades")
    
    # Step 2: Run inference
    logger.info("Initializing inference analyzer...")
    analyzer = TemporalDistributionInference(tokenizer_name=tokenizer_name)
    
    logger.info("Running pattern analysis...")
    analysis_results = analyzer.run_analysis(dataset)
    
    # Step 3: Run statistical validation
    logger.info(f"Performing statistical validation with {bootstrap_iterations} bootstrap iterations...")
    validator = TemporalValidator(
        inference_method=lambda texts: analyzer.infer_temporal_distribution(
            analyzer.analyze_decade_patterns(texts)
        )
    )
    
    confidence_intervals = validator.bootstrap_analysis(
        decade_texts=dataset,
        n_bootstrap=bootstrap_iterations,
        sample_ratio=0.8  # Use 80% of data in each bootstrap sample
    )
    
    validator.visualize_uncertainty(confidence_intervals, 
                                   point_estimate=analysis_results["distribution"])
    
    # Calculate reliability metrics
    reliability_metrics = calculate_reliability_metrics(confidence_intervals)
    
    # Step 4: Evaluate against ground truth if available
    evaluation_results = None
    if ground_truth:
        logger.info("Evaluating against ground truth distribution...")
        evaluator = TemporalEvaluationMetrics()
        evaluation_results = evaluator.evaluate_distribution(
            analysis_results["distribution"],
            ground_truth,
            model_name=tokenizer_name
        )
        
        # Log evaluation metrics
        log10_mse = evaluation_results["distribution_metrics"]["log10_mse"]
        js_distance = evaluation_results["distribution_metrics"]["js_distance"]
        rank_corr = evaluation_results["decade_metrics"]["rank_correlation"]
        
        logger.info(f"Evaluation metrics:")
        logger.info(f"  log10(MSE): {log10_mse:.2f}")
        logger.info(f"  Jensen-Shannon Distance: {js_distance:.4f}")
        logger.info(f"  Temporal Rank Correlation: {rank_corr:.4f}")
    
    # Save complete results
    results_path = RESULTS_DIR / f"{tokenizer_name}_complete_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "tokenizer": tokenizer_name,
            "analysis": analysis_results,
            "confidence_intervals": confidence_intervals,
            "reliability_metrics": reliability_metrics,
            "evaluation": evaluation_results
        }, f, indent=2)
    
    # Log execution time
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    
    # Print detailed summary
    print_analysis_summary(
        tokenizer_name, 
        analysis_results["distribution"], 
        confidence_intervals,
        reliability_metrics,
        evaluation_results
    )
    
    return {
        "analysis": analysis_results,
        "confidence_intervals": confidence_intervals,
        "reliability_metrics": reliability_metrics,
        "evaluation": evaluation_results
    }

def calculate_reliability_metrics(confidence_intervals):
    """
    Calculate metrics to assess the reliability of the statistical analysis.
    Addresses Professor Wei's concerns about statistical validity.
    
    Args:
        confidence_intervals: Results from bootstrap analysis
        
    Returns:
        Dictionary with reliability metrics
    """
    if not confidence_intervals:
        return {"reliability_score": 0, "cv_mean": 1.0, "ci_width_normalized": 1.0}
    
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

def print_analysis_summary(tokenizer_name, distribution, confidence_intervals, 
                          reliability_metrics, evaluation_results=None):
    """
    Print a detailed summary of the analysis results.
    
    Args:
        tokenizer_name: Name of the analyzed tokenizer
        distribution: Inferred distribution
        confidence_intervals: Statistical validation results
        reliability_metrics: Metrics assessing statistical reliability
        evaluation_results: Optional evaluation against ground truth
    """
    print("\nTemporal Distribution Summary:")
    print("=" * 50)
    print(f"Tokenizer: {tokenizer_name}")
    
    # Print reliability assessment
    reliability_score = reliability_metrics.get("reliability_score", 0)
    cv = reliability_metrics.get("coefficient_of_variation", 1.0)
    ci_width = reliability_metrics.get("normalized_ci_width", 1.0)
    
    # Interpret reliability
    reliability_level = "High" if reliability_score > 75 else "Medium" if reliability_score > 50 else "Low"
    
    print(f"\nStatistical Reliability Assessment:")
    print(f"Reliability Score: {reliability_score:.1f}/100 ({reliability_level})")
    print(f"Coefficient of Variation: {cv:.2f}")
    print(f"Normalized CI Width: {ci_width:.2f}")
    
    # Sort decades chronologically
    decades = sorted(distribution.keys())
    
    print("\nInferred Temporal Distribution:")
    for decade in decades:
        proportion = distribution[decade]
        ci = confidence_intervals.get(decade, {})
        ci_str = f" (95% CI: {ci.get('lower_ci', 0):.1%} - {ci.get('upper_ci', 0):.1%})" if ci else ""
        print(f"{decade}: {proportion:.1%}{ci_str}")
    
    # Print evaluation metrics if available
    if evaluation_results:
        print("\nEvaluation Metrics (comparison to ground truth):")
        print(f"log10(MSE): {evaluation_results['distribution_metrics']['log10_mse']:.2f}")
        print(f"Mean Absolute Error: {evaluation_results['distribution_metrics']['mae']:.4f}")
        print(f"Jensen-Shannon Distance: {evaluation_results['distribution_metrics']['js_distance']:.4f}")
        print(f"Rank Correlation: {evaluation_results['decade_metrics']['rank_correlation']:.2f}")
        
        # Print decades with significant discrepancies
        rep_analysis = evaluation_results["decade_metrics"]["representation_analysis"]
        
        if rep_analysis["over_represented"]:
            print("\nOver-represented decades:")
            for decade, value in rep_analysis["over_represented"].items():
                print(f"  {decade}: +{value:.1%}")
                
        if rep_analysis["under_represented"]:
            print("\nUnder-represented decades:")
            for decade, value in rep_analysis["under_represented"].items():
                print(f"  {decade}: -{value:.1%}")
    
    print("\nMost Distinctive Decade Patterns:")
    print("=" * 50)
    
    for decade in decades:
        if decade in analysis_results["distinctive_patterns"]:
            patterns = analysis_results["distinctive_patterns"][decade][:3]
            if patterns:
                print(f"\n{decade} distinctive patterns:")
                for pattern, score in patterns:
                    print(f"  '{pattern}': {score:.2f}x more common than average")

def run_comparative_analysis(tokenizer_names=["gpt2", "gpt2-medium", "gpt2-large"]):
    """
    Run a comparative analysis of multiple tokenizers.
    
    Args:
        tokenizer_names: List of tokenizers to analyze and compare
    
    Returns:
        Comparative analysis results
    """
    logger.info(f"Starting comparative analysis of {len(tokenizer_names)} tokenizers...")
    
    all_results = []
    
    # Analyze each tokenizer
    for tokenizer_name in tokenizer_names:
        logger.info(f"\nAnalyzing {tokenizer_name}...")
        results = run_analysis(tokenizer_name=tokenizer_name)
        all_results.append((tokenizer_name, results))
    
    # Compare distributions
    distributions = [(name, results["analysis"]["distribution"]) 
                   for name, results in all_results]
    
    # Visualize comparison
    visualize_distribution_comparison(distributions)
    
    logger.info("Comparative analysis complete.")
    
    return all_results

def visualize_distribution_comparison(distributions):
    """
    Create a visualization comparing distributions from multiple tokenizers.
    
    Args:
        distributions: List of (tokenizer_name, distribution) tuples
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get all decades from all distributions
    all_decades = set()
    for _, distribution in distributions:
        all_decades.update(distribution.keys())
    decades = sorted(all_decades)
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Set bar width and positions
    bar_width = 0.8 / len(distributions)
    positions = np.arange(len(decades))
    
    # Plot each distribution
    for i, (name, distribution) in enumerate(distributions):
        # Calculate offsets for grouped bars
        offset = i - len(distributions) / 2 + 0.5
        x_positions = positions + (offset * bar_width)
        
        # Get values for each decade (0 if missing)
        values = [distribution.get(decade, 0) for decade in decades]
        
        plt.bar(x_positions, values, width=bar_width, label=name, alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Decade')
    plt.ylabel('Estimated Proportion')
    plt.title('Comparative Temporal Distribution Analysis')
    plt.xticks(positions, decades, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    results_dir = RESULTS_DIR / "comparative"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "tokenizer_comparison.png")
    plt.close()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Temporal distribution analysis for tokenizers")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to analyze")
    parser.add_argument("--texts", type=int, default=5, help="Texts per decade")
    parser.add_argument("--bootstrap", type=int, default=20, help="Bootstrap iterations")
    parser.add_argument("--comparative", action="store_true", help="Run comparative analysis")
    
    args = parser.parse_args()
    
    if args.comparative:
        # Run comparison of multiple tokenizers
        run_comparative_analysis(["gpt2", "gpt2-medium", "gpt2-large"])
    else:
        # Run analysis of a single tokenizer
        run_analysis(
            tokenizer_name=args.tokenizer,
            texts_per_decade=args.texts,
            bootstrap_iterations=args.bootstrap
        )