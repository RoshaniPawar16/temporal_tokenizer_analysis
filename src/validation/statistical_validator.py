"""
Statistical Validator for Temporal Inference

Implements bootstrapping to estimate uncertainty in temporal distribution inference.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import logging
from typing import Dict, List, Callable, Optional
import json
from pathlib import Path

from ..config import RESULTS_DIR

logger = logging.getLogger(__name__)

class TemporalValidator:
    """
    Validates temporal distribution inference with bootstrapping.
    Designed for resource-constrained environments.
    """
    
    def __init__(self, inference_method: Callable):
        """
        Initialize with inference method to validate.
        
        Args:
            inference_method: Function that performs inference
        """
        self.inference_method = inference_method
        self.results_dir = RESULTS_DIR / "validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def bootstrap_analysis(self,
                         decade_texts: Dict[str, List[str]],
                         n_bootstrap: int = 5,  # Reduced for resource constraints
                         sample_ratio: float = 0.8) -> Dict[str, Dict[str, float]]:
        """
        Perform bootstrap resampling to estimate uncertainty.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            n_bootstrap: Number of bootstrap iterations
            sample_ratio: Proportion of texts to sample in each iteration
            
        Returns:
            Dictionary with confidence intervals for each decade
        """
        # Filter to non-empty decades
        decade_texts = {decade: texts for decade, texts in decade_texts.items() if texts}
        
        if not decade_texts:
            logger.warning("No data available for bootstrapping")
            return {}
        
        # Track bootstrap results
        bootstrap_results = []
        
        # Run bootstrap iterations
        logger.info(f"Running {n_bootstrap} bootstrap iterations...")
        
        for i in range(n_bootstrap):
            logger.info(f"Bootstrap iteration {i+1}/{n_bootstrap}")
            
            # Create bootstrap sample
            bootstrap_sample = {}
            for decade, texts in decade_texts.items():
                # Calculate sample size
                sample_size = max(1, int(len(texts) * sample_ratio))
                # Sample with replacement
                bootstrap_sample[decade] = random.choices(texts, k=sample_size)
            
            # Run inference on bootstrap sample
            try:
                result = self.inference_method(bootstrap_sample)
                bootstrap_results.append(result)
            except Exception as e:
                logger.warning(f"Inference failed on bootstrap sample: {e}")
        
        # Calculate statistics
        all_decades = set()
        for result in bootstrap_results:
            all_decades.update(result.keys())
        
        confidence_intervals = {}
        for decade in all_decades:
            # Get all proportion estimates for this decade
            proportions = [result.get(decade, 0) for result in bootstrap_results]
            
            if not proportions:
                continue
                
            # Calculate statistics
            confidence_intervals[decade] = {
                "mean": float(np.mean(proportions)),
                "std_dev": float(np.std(proportions)),
                "lower_ci": float(np.percentile(proportions, 5) if len(proportions) >= 3 else 0),
                "upper_ci": float(np.percentile(proportions, 95) if len(proportions) >= 3 else 0)
            }
        
        return confidence_intervals
    
    def visualize_uncertainty(self, 
                            confidence_intervals: Dict[str, Dict[str, float]],
                            point_estimate: Optional[Dict[str, float]] = None):
        """
        Visualize bootstrap results with confidence intervals.
        
        Args:
            confidence_intervals: Results from bootstrap_analysis
            point_estimate: Single point estimate for comparison
        """
        if not confidence_intervals:
            logger.warning("No confidence intervals to visualize")
            return
        
        # Sort decades chronologically
        decades = sorted(confidence_intervals.keys())
        
        # Extract data
        means = [confidence_intervals[d]["mean"] for d in decades]
        lower = [confidence_intervals[d].get("lower_ci", means[i] * 0.8) for i, d in enumerate(decades)]
        upper = [confidence_intervals[d].get("upper_ci", means[i] * 1.2) for i, d in enumerate(decades)]
        
        # Calculate error bars
        errors_lower = [means[i] - lower[i] for i in range(len(means))]
        errors_upper = [upper[i] - means[i] for i in range(len(means))]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot confidence intervals
        plt.bar(
            decades, 
            means, 
            alpha=0.7, 
            color='skyblue', 
            yerr=[errors_lower, errors_upper],
            capsize=5
        )
        
        # Add data labels
        for i, v in enumerate(means):
            plt.text(i, v + 0.01, f"{v:.1%}", ha='center')
        
        # Add title and labels
        plt.title('Temporal Distribution with Confidence Intervals')
        plt.xlabel('Decade')
        plt.ylabel('Estimated Proportion')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.results_dir / "temporal_distribution_uncertainty.png")
        plt.close()