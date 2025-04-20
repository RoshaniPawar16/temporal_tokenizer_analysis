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

import tqdm

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
        Perform bootstrap analysis to compute confidence intervals for the distribution.
        Enhanced with proper error handling and statistical validation.
        """
        logger.info(f"Performing bootstrap analysis with {n_bootstrap} iterations...")
        
        # ADDED: Ensure we complete all iterations
        successful_iterations = 0
        max_attempts = n_bootstrap * 2  # Allow up to double attempts to get n_bootstrap successes
        
        # First get baseline distribution using all data
        try:
            baseline_distribution = self.inference_method(decade_texts)
            if not baseline_distribution:
                logger.warning("Baseline inference produced empty distribution")
                baseline_distribution = {decade: 1.0/len(decade_texts) for decade in decade_texts.keys()}
        except Exception as e:
            logger.error(f"Error in baseline inference: {e}")
            baseline_distribution = {decade: 1.0/len(decade_texts) for decade in decade_texts.keys()}
        
        # Initialize containers for bootstrap results
        bootstrap_distributions = []
        decade_values = {decade: [] for decade in baseline_distribution.keys()}
        
        # Run bootstrap iterations with retry logic
        attempt = 0
        while successful_iterations < n_bootstrap and attempt < max_attempts:
            attempt += 1
            logger.info(f"Bootstrap iteration attempt {attempt}/{max_attempts} (successful: {successful_iterations}/{n_bootstrap})")
            
            try:
                # Create bootstrap sample with stratified sampling to preserve decade proportions
                bootstrap_sample = {}
                for decade, texts in decade_texts.items():
                    if not texts:
                        bootstrap_sample[decade] = []
                        continue
                    
                    # Sample with replacement
                    sample_size = max(1, int(len(texts) * sample_ratio))
                    
                    # MODIFIED: Stratified sampling for better representation
                    sampled_texts = []
                    for _ in range(sample_size):
                        idx = random.randrange(len(texts))
                        sampled_texts.append(texts[idx])
                    
                    bootstrap_sample[decade] = sampled_texts
                
                # Run inference on bootstrap sample
                try:
                    bootstrap_dist = self.inference_method(bootstrap_sample)
                    if not bootstrap_dist:
                        logger.warning(f"Empty distribution in bootstrap iteration {attempt}")
                        continue
                        
                    bootstrap_distributions.append(bootstrap_dist)
                    successful_iterations += 1
                    
                    # Record values for each decade
                    for decade, value in bootstrap_dist.items():
                        if decade in decade_values:
                            # Ensure we have numeric values
                            if isinstance(value, (int, float)):
                                decade_values[decade].append(float(value))
                            else:
                                logger.warning(f"Non-numeric value in bootstrap iteration {attempt} for {decade}: {type(value)}")
                    
                    # Log progress for successful iterations only
                    logger.info(f"Completed bootstrap iteration {successful_iterations}/{n_bootstrap}")
                except Exception as e:
                    logger.warning(f"Error in bootstrap iteration {attempt} inference: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Failed to create bootstrap sample {attempt}: {e}")
                continue
            
            # Force garbage collection every few iterations
            if attempt % 5 == 0:
                import gc
                gc.collect()
        
        # Log how many iterations were successful
        logger.info(f"Completed {successful_iterations}/{n_bootstrap} bootstrap iterations successfully")
        
        # Calculate statistics
        confidence_intervals = {}
        for decade, values in decade_values.items():
            if not values:
                confidence_intervals[decade] = {
                    "mean": baseline_distribution.get(decade, 0),
                    "std_dev": 0,
                    "lower_ci": baseline_distribution.get(decade, 0),
                    "upper_ci": baseline_distribution.get(decade, 0),
                    "samples": 0
                }
                continue
                
            # Sort values for percentile confidence intervals
            sorted_values = sorted(values)
            
            # 95% confidence interval
            lower_idx = int(0.025 * len(sorted_values))
            upper_idx = int(0.975 * len(sorted_values))
            
            confidence_intervals[decade] = {
                "mean": sum(values) / len(values),
                "std_dev": np.std(values),
                "lower_ci": sorted_values[lower_idx] if lower_idx < len(sorted_values) else sorted_values[0],
                "upper_ci": sorted_values[upper_idx] if upper_idx < len(sorted_values) else sorted_values[-1],
                "samples": len(values),
                "min": min(values),
                "max": max(values),
                "coefficient_of_variation": np.std(values) / max(1e-5, np.mean(values))
            }
        
        # Calculate reliability metrics
        reliability_score = self.calculate_reliability_metrics(confidence_intervals)["reliability_score"]
        
        # Add overall reliability score to results
        for decade in confidence_intervals:
            confidence_intervals[decade]["reliability_score"] = reliability_score
        
        return confidence_intervals

    def calculate_reliability_metrics(self, confidence_intervals):
        """
        Calculate metrics to assess the reliability of the bootstrap analysis.
        
        Args:
            confidence_intervals: Dictionary with confidence interval statistics
            
        Returns:
            Dictionary with reliability metrics
        """
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