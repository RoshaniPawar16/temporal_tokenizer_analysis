"""
Maxwell-optimized runner for temporal distribution inference

This script runs the temporal distribution analysis with enhanced 
parameters for better performance on the Maxwell HPC environment.
"""

import os
import argparse
import logging
from datetime import datetime
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'maxwell_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
logger = logging.getLogger(__name__)

# Import project modules
from src.data.dataset_manager import TemporalDatasetManager
from src.validation.temporal_inference import TemporalDistributionInference
from src.validation.evaluation_metrics import TemporalEvaluationMetrics
from src.validation.statistical_validator import TemporalValidator
from src.config import TIME_PERIODS

def main(args):
    """Main execution function with enhanced parameters for Maxwell"""
    logger.info(f"Starting temporal distribution analysis with enhanced parameters")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Texts per decade: {args.texts_per_decade}")
    logger.info(f"Distribution pattern: {args.distribution}")
    
    # Initialize components
    dataset_manager = TemporalDatasetManager()
    inference = TemporalDistributionInference(tokenizer_name=args.tokenizer)
    evaluator = TemporalEvaluationMetrics()
    
    # Define test distributions
    test_distributions = {
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
    
    # Create controlled dataset with the selected distribution
    logger.info(f"Creating controlled dataset with {args.distribution} distribution")
    selected_distribution = test_distributions[args.distribution]["distribution"]
    
    # Create dataset with the specified distribution
    controlled_dataset = dataset_manager.create_controlled_dataset(
        distribution=selected_distribution,
        total_texts=args.texts_per_decade * 10  # Increase total texts for better statistics
    )
    
    # Extract text-only version of the dataset (removing source information)
    decade_texts = {decade: [text for text, _ in texts] for decade, texts in controlled_dataset.items()}
    
    # Calculate actual distribution based on the dataset
    total_texts = sum(len(texts) for texts in decade_texts.values())
    ground_truth = {decade: len(texts)/total_texts for decade, texts in decade_texts.items()} if total_texts else selected_distribution
    
    # Run inference with improved parameters
    logger.info("Running inference with enhanced parameters")
    start_time = time.time()
    
    # Analyze with larger sample size for better statistical reliability
    decade_patterns = inference.analyze_decade_patterns(
        decade_texts,
        sample_size=args.sample_size  # Use larger sample size for better pattern detection
    )
    
    # Infer distribution with more robust constraints
    inferred_distribution = inference.infer_temporal_distribution(
        decade_patterns,
        weight_early_merges=True,  # Give higher weight to earlier merge rules
        continuity_constraint=True  # Add constraint for temporal continuity
    )
    
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f} seconds")
    
    # Evaluate results against ground truth
    evaluation_results = evaluator.evaluate_distribution(
        inferred_distribution,
        ground_truth,
        model_name=f"{args.tokenizer}_{args.distribution}_{args.texts_per_decade}"
    )
    
    # Add timing information
    evaluation_results["inference_time"] = inference_time
    
    # Run bootstrap validation if requested
    if args.bootstrap:
        logger.info(f"Running bootstrap validation with {args.bootstrap_iterations} iterations")
        validator = TemporalValidator(
            inference_method=lambda texts: inference.infer_temporal_distribution(
                inference.analyze_decade_patterns(texts, sample_size=args.sample_size)
            )
        )
        
        confidence_intervals = validator.bootstrap_analysis(
            decade_texts=decade_texts,
            n_bootstrap=args.bootstrap_iterations,
            sample_ratio=0.8
        )
        
        # Add confidence intervals to results
        evaluation_results["confidence_intervals"] = confidence_intervals
    
    # Save results
    results_dir = Path("results_from_maxwell")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    results_file = results_dir / f"results_{args.tokenizer}_{args.distribution}_{args.texts_per_decade}.json"
    
    # Convert numpy values to native Python types for JSON serialization
    serializable_results = {}
    for key, value in evaluation_results.items():
        if key in ["inferred_distribution", "ground_truth_distribution"]:
            serializable_results[key] = {k: float(v) for k, v in value.items()}
        elif key == "confidence_intervals" and value:
            serializable_ci = {}
            for decade, ci_data in value.items():
                serializable_ci[decade] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                          for k, v in ci_data.items()}
            serializable_results[key] = serializable_ci
        elif key == "distribution_metrics":
            serializable_results[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                         for k, v in value.items()}
        else:
            serializable_results[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary metrics
    print("\n" + "="*50)
    print(f"RESULTS SUMMARY FOR {args.tokenizer} ON {args.distribution.upper()}")
    print("="*50)
    print(f"Log10(MSE): {evaluation_results['distribution_metrics']['log10_mse']:.4f}")
    print(f"MAE: {evaluation_results['distribution_metrics']['mae']:.4f}")
    print(f"Jensen-Shannon Distance: {evaluation_results['distribution_metrics']['js_distance']:.4f}")
    print(f"Inference Time: {inference_time:.2f} seconds")
    print("="*50)
    
    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Temporal Distribution Analysis")
    parser.add_argument("--tokenizer", type=str, default="gpt2", 
                        help="Tokenizer to analyze (default: gpt2)")
    parser.add_argument("--distribution", type=str, default="recency_bias", 
                        choices=["uniform", "recency_bias", "historical_bias", "bimodal"],
                        help="Distribution pattern to test (default: recency_bias)")
    parser.add_argument("--texts_per_decade", type=int, default=100,
                        help="Number of texts per decade (default: 100)")
    parser.add_argument("--sample_size", type=int, default=10000,
                        help="Sample size for merge rule analysis (default: 10000)")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Run bootstrap validation")
    parser.add_argument("--bootstrap_iterations", type=int, default=100,
                        help="Number of bootstrap iterations (default: 100)")
    
    args = parser.parse_args()
    main(args)