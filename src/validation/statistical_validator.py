"""
Statistical validation for temporal distribution inference.

This module provides methods for assessing the reliability of inferred
temporal distributions through statistical techniques like bootstrapping.
"""

import logging
import random
import numpy as np
from typing import Dict, List, Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class TemporalValidator:
    """
    Implements statistical validation techniques for temporal distribution inference.
    """
    
    def __init__(self, inference_method: Callable[[Dict[str, List[str]]], Dict[str, float]]):
        """
        Initialize with an inference method.
        
        Args:
            inference_method: Function that takes texts by decade and returns a distribution
        """
        self.inference_method = inference_method
    
    def bootstrap_analysis(self, decade_texts, n_bootstrap=30, sample_ratio=0.8):
        """
        Perform bootstrap analysis to estimate confidence intervals.
        """
        logger.info(f"Running {n_bootstrap} bootstrap iterations...")
        
        # Initialize results
        bootstrap_results = defaultdict(list)
        
        # Run bootstrap iterations
        for i in range(n_bootstrap):
            logger.info(f"Bootstrap iteration {i+1}/{n_bootstrap}")
            
            # Create bootstrap sample
            bootstrap_sample = self._create_bootstrap_sample(decade_texts, sample_ratio)
            
            # Run inference
            try:
                distribution = self.inference_method(bootstrap_sample)
                
                # Record results for each decade
                for decade, proportion in distribution.items():
                    bootstrap_results[decade].append(proportion)
            except Exception as e:
                logger.error(f"Error in bootstrap iteration {i+1}: {e}")
        
        # Calculate statistics
        confidence_intervals = {}
        for decade, proportions in bootstrap_results.items():
            if proportions:
                mean = np.mean(proportions)
                median = np.median(proportions)
                std_dev = np.std(proportions, ddof=1)
                
                # 95% confidence interval
                sorted_proportions = sorted(proportions)
                lower_idx = int(0.025 * len(sorted_proportions))
                upper_idx = int(0.975 * len(sorted_proportions))
                lower_ci = sorted_proportions[max(0, lower_idx)]
                upper_ci = sorted_proportions[min(len(sorted_proportions)-1, upper_idx)]
                
                confidence_intervals[decade] = {
                    "mean": mean,
                    "median": median,
                    "std_dev": std_dev,
                    "lower_ci": lower_ci,
                    "upper_ci": upper_ci,
                    "samples": len(proportions)
                }
        
        return confidence_intervals

    def _create_bootstrap_sample(self, decade_texts, sample_ratio=0.8):
        """
        Create a bootstrap sample by randomly sampling with replacement.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            sample_ratio: Proportion of samples to use
            
        Returns:
            Dictionary mapping decades to bootstrapped text samples
        """
        bootstrap_sample = {}
        
        for decade, texts in decade_texts.items():
            if not texts:
                continue
                
            # Calculate sample size
            sample_size = max(int(len(texts) * sample_ratio), 1)
            
            # Sample with replacement
            bootstrap_sample[decade] = random.choices(texts, k=sample_size)
        
        return bootstrap_sample
    
    def ensemble_inference(self, decade_patterns: Dict[str, Dict]) -> Dict[str, float]:
        """
        Combine multiple inference methods for more robust results.
        """
        # Get results from different methods
        lp_distribution = self.inference.infer_temporal_distribution(decade_patterns)
        heuristic_distribution = self.inference._infer_distribution_heuristic(decade_patterns)
        
        # Simple averaging ensemble (equal weights)
        ensemble_distribution = {}
        all_decades = sorted(set(lp_distribution.keys()) | set(heuristic_distribution.keys()))
        
        for decade in all_decades:
            lp_value = lp_distribution.get(decade, 0.0)
            heuristic_value = heuristic_distribution.get(decade, 0.0)
            # Average the two methods
            ensemble_distribution[decade] = (lp_value + heuristic_value) / 2.0
        
        # Ensure the distribution sums to 1
        total = sum(ensemble_distribution.values())
        if total > 0:
            ensemble_distribution = {d: v/total for d, v in ensemble_distribution.items()}
        
        return ensemble_distribution

    def cross_validation(self, 
                       decade_texts: Dict[str, List[str]], 
                       k_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Perform k-fold cross-validation for assessing prediction stability.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            k_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Running {k_folds}-fold cross-validation...")
        
        # Prepare folds
        folds = []
        for _ in range(k_folds):
            fold = {}
            for decade, texts in decade_texts.items():
                if texts:
                    # Random subset (without replacement)
                    subset_size = len(texts) // k_folds
                    if subset_size > 0:
                        fold[decade] = random.sample(texts, subset_size)
            folds.append(fold)
        
        # Run inference on each fold
        fold_results = []
        for i, fold in enumerate(folds):
            logger.info(f"Processing fold {i+1}/{k_folds}")
            distribution = self.inference_method(fold)
            fold_results.append(distribution)
        
        # Calculate statistics
        all_decades = set()
        for dist in fold_results:
            all_decades.update(dist.keys())
        
        cv_results = {}
        for decade in all_decades:
            values = [dist.get(decade, 0) for dist in fold_results]
            if values:
                cv_results[decade] = {
                    "mean": np.mean(values),
                    "std_dev": np.std(values),
                    "coefficient_of_variation": np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                }
        
        return cv_results
    
    def sensitivity_analysis(self, 
                          decade_texts: Dict[str, List[str]], 
                          sample_fractions: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0]) -> Dict:
        """
        Perform sensitivity analysis to data volume.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            sample_fractions: Fractions of data to sample
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        logger.info(f"Running sensitivity analysis with fractions: {sample_fractions}")
        
        # Run inference with different data volumes
        results = {}
        for fraction in sample_fractions:
            logger.info(f"Processing {fraction:.1%} of data")
            
            # Sample data
            sampled_data = {}
            for decade, texts in decade_texts.items():
                if texts:
                    sample_size = max(1, int(len(texts) * fraction))
                    sampled_data[decade] = random.sample(texts, min(sample_size, len(texts)))
            
            # Run inference
            distribution = self.inference_method(sampled_data)
            results[fraction] = distribution
        
        # Analyze stability across different data volumes
        all_decades = set()
        for dist in results.values():
            all_decades.update(dist.keys())
            
        stability_metrics = {}
        for decade in all_decades:
            # Extract values across different fractions
            values = [dist.get(decade, 0) for dist in results.values()]
            
            # Calculate variability
            if len(values) > 1:
                stability_metrics[decade] = {
                    "values": {str(fraction): results[fraction].get(decade, 0) 
                             for fraction in sample_fractions},
                    "range": max(values) - min(values),
                    "std_dev": np.std(values),
                    "coefficient_of_variation": np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                }
        
        return {
            "distributions": results,
            "stability_metrics": stability_metrics
        }