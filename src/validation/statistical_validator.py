"""
Statistical validation for temporal distribution inference.

This module provides methods for assessing the reliability of inferred
temporal distributions through statistical techniques like bootstrapping.
"""

import logging
import random
import numpy as np
from typing import Dict, List, Callable, Any

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
    
    def bootstrap_analysis(self, 
                         decade_texts: Dict[str, List[str]], 
                         n_bootstrap: int = 100, 
                         sample_ratio: float = 0.8) -> Dict[str, Dict[str, float]]:
        """
        Perform bootstrap analysis to estimate confidence intervals.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            n_bootstrap: Number of bootstrap iterations
            sample_ratio: Proportion of data to sample in each iteration
            
        Returns:
            Dictionary with confidence intervals for each decade
        """
        logger.info(f"Running {n_bootstrap} bootstrap iterations...")
        
        bootstrap_results = []
        
        for i in range(n_bootstrap):
            logger.info(f"Bootstrap iteration {i+1}/{n_bootstrap}")
            
            # Create bootstrap sample
            bootstrap_sample = {}
            for decade, texts in decade_texts.items():
                if texts:
                    # Sample with replacement
                    sample_size = max(1, int(len(texts) * sample_ratio))
                    bootstrap_sample[decade] = random.choices(texts, k=sample_size)
            
            # Run inference on bootstrap sample
            distribution = self.inference_method(bootstrap_sample)
            
            bootstrap_results.append(distribution)
        
        # Calculate statistics
        all_decades = set()
        for dist in bootstrap_results:
            all_decades.update(dist.keys())
        
        confidence_intervals = {}
        for decade in all_decades:
            values = [dist.get(decade, 0) for dist in bootstrap_results]
            if values:
                confidence_intervals[decade] = {
                    "mean": np.mean(values),
                    "std_dev": np.std(values),
                    "lower_ci": np.percentile(values, 2.5),  # 95% confidence interval
                    "upper_ci": np.percentile(values, 97.5),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        return confidence_intervals
    
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