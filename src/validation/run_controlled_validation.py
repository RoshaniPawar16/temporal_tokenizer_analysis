"""
Controlled validation of temporal inference on known distributions.
Measures accuracy of inference with known ground truth.
"""

import logging
import random
from typing import Dict, List
import json
from pathlib import Path

from ..data.dataset_manager import TemporalDatasetManager
from ..validation.evaluation_metrics import TemporalEvaluationMetrics
from ..validation.temporal_inference import TemporalDistributionInference
from ..config import RESULTS_DIR, TIME_PERIODS

logger = logging.getLogger(__name__)

def run_controlled_validation(tokenizer_name="gpt2", n_iterations=5):
    """
    Run a controlled validation experiment with known ground truth distributions.
    
    Args:
        tokenizer_name: Name of tokenizer to test
        n_iterations: Number of validation iterations
    
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Running controlled validation for {tokenizer_name}...")
    
    # Initialize components
    dataset_manager = TemporalDatasetManager()
    inference = TemporalDistributionInference(tokenizer_name=tokenizer_name)
    evaluator = TemporalEvaluationMetrics()
    
    # Create results directory
    results_dir = RESULTS_DIR / "controlled_validation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define test distributions to evaluate
    test_distributions = [
        {
            "name": "uniform",
            "distribution": {decade: 1.0/len(TIME_PERIODS) for decade in TIME_PERIODS.keys()}
        },
        {
            "name": "recency_bias",
            "distribution": {
                "1950s": 0.05, "1960s": 0.05, "1970s": 0.10, "1980s": 0.10,
                "1990s": 0.15, "2000s": 0.20, "2010s": 0.25, "2020s": 0.10
            }
        },
        {
            "name": "historical_bias",
            "distribution": {
                "1950s": 0.25, "1960s": 0.20, "1970s": 0.15, "1980s": 0.10,
                "1990s": 0.10, "2000s": 0.10, "2010s": 0.05, "2020s": 0.05
            }
        },
        {
            "name": "bimodal",
            "distribution": {
                "1950s": 0.20, "1960s": 0.10, "1970s": 0.05, "1980s": 0.05,
                "1990s": 0.05, "2000s": 0.05, "2010s": 0.20, "2020s": 0.30
            }
        }
    ]
    
    # Run validation for each test distribution
    all_results = []
    
    for test_case in test_distributions:
        logger.info(f"\nValidating with distribution: {test_case['name']}")
        distribution_results = []
        
        for i in range(n_iterations):
            logger.info(f"Iteration {i+1}/{n_iterations}")
            
            # Create dataset with known distribution
            controlled_dataset = dataset_manager.create_controlled_dataset(
                distribution=test_case["distribution"],
                total_texts=50  # Adjusted for resource constraints
            )
            
            # Run inference
            try:
                decade_patterns = inference.analyze_decade_patterns(controlled_dataset)
                inferred_distribution = inference.infer_temporal_distribution(decade_patterns)
                
                # Calculate Hayase-style log10(MSE)
                log10_mse = evaluator.calculate_hayase_log_mse(
                    inferred_distribution, 
                    test_case["distribution"]
                )
                
                distribution_results.append({
                    "iteration": i+1,
                    "inferred_distribution": inferred_distribution,
                    "log10_mse": log10_mse
                })
                
                logger.info(f"  log10(MSE): {log10_mse:.2f}")
                
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
        
        # Calculate average results
        if distribution_results:
            avg_log10_mse = sum(r["log10_mse"] for r in distribution_results) / len(distribution_results)
            
            # Full evaluation on the last iteration as an example
            if distribution_results:
                last_result = distribution_results[-1]
                evaluation = evaluator.evaluate_distribution(
                    last_result["inferred_distribution"],
                    test_case["distribution"],
                    model_name=f"{tokenizer_name}_{test_case['name']}"
                )
            
            logger.info(f"Average log10(MSE) for {test_case['name']}: {avg_log10_mse:.2f}")
            
            # Add to overall results
            all_results.append({
                "distribution_name": test_case["name"],
                "ground_truth": test_case["distribution"],
                "iterations": distribution_results,
                "avg_log10_mse": avg_log10_mse
            })
    
    # Save all results
    with open(results_dir / f"{tokenizer_name}_validation_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nControlled validation complete. Results saved to {results_dir}")
    
    return all_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_controlled_validation()