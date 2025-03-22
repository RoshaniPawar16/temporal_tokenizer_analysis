"""
Evaluation Metrics for Temporal Distribution Inference

Implements metrics to evaluate the accuracy of temporal distribution inference
by comparing against ground truth distributions when available.

This module provides both distribution-level metrics (comparing overall distributions)
and decade-level metrics (evaluating accuracy for specific time periods).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import logging
from collections import defaultdict

from ..config import RESULTS_DIR

logger = logging.getLogger(__name__)

class TemporalEvaluationMetrics:
    """
    Evaluates temporal distribution inference accuracy using multiple metrics.
    
    Implements metrics from Hayase et al. (log10(MSE)) and adds additional metrics
    for comprehensive evaluation:
    
    1. Distribution-level metrics:
       - Mean Squared Error (MSE) and log10(MSE)
       - Kullback-Leibler Divergence
       - Jensen-Shannon Distance
       - Mean Absolute Error
       
    2. Decade-level metrics:
       - Per-decade absolute error
       - Spearman rank correlation (temporal trend accuracy)
       - Identification of over/under-represented decades
    """
    
    def __init__(self):
        """Initialize the evaluation metrics module."""
        self.results_dir = RESULTS_DIR / "evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_distribution(self, 
                            inferred: Dict[str, float], 
                            ground_truth: Dict[str, float],
                            model_name: str = "model") -> Dict:
        """
        Comprehensively evaluate an inferred distribution against ground truth.
        
        Args:
            inferred: Dictionary mapping decades to inferred proportions
            ground_truth: Dictionary mapping decades to ground truth proportions
            model_name: Name of the model/tokenizer being evaluated
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Normalize both distributions if they don't sum to 1
        inferred_norm = self._normalize_distribution(inferred)
        ground_truth_norm = self._normalize_distribution(ground_truth)
        
        # Get a sorted list of all decades from both distributions
        all_decades = sorted(set(inferred_norm.keys()) | set(ground_truth_norm.keys()))
        
        # Create vectors with values for all decades (using 0 for missing decades)
        inferred_vec = [inferred_norm.get(decade, 0.0) for decade in all_decades]
        truth_vec = [ground_truth_norm.get(decade, 0.0) for decade in all_decades]
        
        # Calculate distribution-level metrics
        mse = self._calculate_mse(inferred_vec, truth_vec)
        log10_mse = np.log10(mse) if mse > 0 else float('-inf')
        mae = self._calculate_mae(inferred_vec, truth_vec)
        kl_div = self._calculate_kl_divergence(inferred_vec, truth_vec)
        js_dist = self._calculate_js_distance(inferred_vec, truth_vec)
        
        # Calculate decade-level metrics
        decade_errors = self._calculate_decade_errors(inferred_norm, ground_truth_norm)
        rank_correlation = self._calculate_rank_correlation(inferred_norm, ground_truth_norm)
        representation_analysis = self._analyze_decade_representation(inferred_norm, ground_truth_norm)
        
        # Compile all metrics
        evaluation_results = {
            "model_name": model_name,
            "distribution_metrics": {
                "mse": mse,
                "log10_mse": log10_mse,
                "mae": mae,
                "kl_divergence": kl_div,
                "js_distance": js_dist
            },
            "decade_metrics": {
                "decade_errors": decade_errors,
                "rank_correlation": rank_correlation,
                "representation_analysis": representation_analysis
            },
            "decades": all_decades,
            "inferred_distribution": inferred_norm,
            "ground_truth_distribution": ground_truth_norm
        }
        
        # Generate visualizations
        self._visualize_distribution_comparison(
            inferred_norm, 
            ground_truth_norm, 
            model_name
        )
        
        self._visualize_decade_errors(
            decade_errors,
            model_name
        )
        
        # Save results
        self._save_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def _normalize_distribution(self, distribution: Dict[str, float]) -> Dict[str, float]:
        """Ensure distribution sums to 1."""
        total = sum(distribution.values())
        if abs(total - 1.0) > 1e-6 and total > 0:  # Allow small floating point errors
            return {decade: value / total for decade, value in distribution.items()}
        return distribution
    
    def _calculate_mse(self, inferred: List[float], truth: List[float]) -> float:
        """Calculate Mean Squared Error."""
        return np.mean([(i - t) ** 2 for i, t in zip(inferred, truth)])
    
    def _calculate_mae(self, inferred: List[float], truth: List[float]) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean([abs(i - t) for i, t in zip(inferred, truth)])
    
    def _calculate_kl_divergence(self, inferred: List[float], truth: List[float]) -> float:
        """
        Calculate Kullback-Leibler Divergence.
        Smoothed to handle zeros.
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        inferred_smooth = [i + epsilon for i in inferred]
        truth_smooth = [t + epsilon for t in truth]
        
        # Normalize after smoothing
        inferred_sum = sum(inferred_smooth)
        truth_sum = sum(truth_smooth)
        
        inferred_norm = [i / inferred_sum for i in inferred_smooth]
        truth_norm = [t / truth_sum for t in truth_smooth]
        
        # Calculate KL divergence: sum(p_i * log(p_i / q_i))
        return sum(p * np.log(p / q) for p, q in zip(truth_norm, inferred_norm))
    
    def _calculate_js_distance(self, inferred: List[float], truth: List[float]) -> float:
        """
        Calculate Jensen-Shannon Distance.
        Symmetric and bounded between 0 and 1.
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        inferred_smooth = [i + epsilon for i in inferred]
        truth_smooth = [t + epsilon for t in truth]
        
        # Normalize after smoothing
        inferred_sum = sum(inferred_smooth)
        truth_sum = sum(truth_smooth)
        
        p = [i / inferred_sum for i in inferred_smooth]
        q = [t / truth_sum for t in truth_smooth]
        
        # Calculate midpoint distribution
        m = [(p_i + q_i) / 2 for p_i, q_i in zip(p, q)]
        
        # Calculate JS divergence: 0.5 * (KL(P||M) + KL(Q||M))
        kl_p_m = sum(p_i * np.log(p_i / m_i) for p_i, m_i in zip(p, m))
        kl_q_m = sum(q_i * np.log(q_i / m_i) for q_i, m_i in zip(q, m))
        
        js_divergence = 0.5 * (kl_p_m + kl_q_m)
        
        # Convert to distance by taking the square root
        return np.sqrt(js_divergence)
    
    def _calculate_decade_errors(self, inferred: Dict[str, float], truth: Dict[str, float]) -> Dict[str, float]:
        """Calculate absolute error for each decade."""
        all_decades = sorted(set(inferred.keys()) | set(truth.keys()))
        return {decade: abs(inferred.get(decade, 0) - truth.get(decade, 0)) for decade in all_decades}
    
    def _calculate_rank_correlation(self, inferred: Dict[str, float], truth: Dict[str, float]) -> float:
        """
        Calculate Spearman rank correlation to assess temporal trend accuracy.
        
        Returns:
            Correlation coefficient (-1 to 1)
        """
        common_decades = sorted(set(inferred.keys()) & set(truth.keys()))
        
        if len(common_decades) < 2:
            return 0.0  # Not enough data points for correlation
            
        # Extract values for common decades
        inferred_values = [inferred[decade] for decade in common_decades]
        truth_values = [truth[decade] for decade in common_decades]
        
        # Calculate ranks
        inferred_ranks = pd.Series(inferred_values).rank()
        truth_ranks = pd.Series(truth_values).rank()
        
        # Calculate Spearman correlation
        n = len(common_decades)
        rank_diff_squared = sum((i - t) ** 2 for i, t in zip(inferred_ranks, truth_ranks))
        
        # Spearman correlation formula
        correlation = 1 - (6 * rank_diff_squared) / (n * (n**2 - 1))
        
        return correlation
    
    def _analyze_decade_representation(self, 
                                    inferred: Dict[str, float], 
                                    truth: Dict[str, float], 
                                    threshold: float = 0.05) -> Dict:
        """
        Analyze which decades are over- or under-represented.
        
        Args:
            inferred: Inferred distribution
            truth: Ground truth distribution
            threshold: Minimum difference to be considered significant
            
        Returns:
            Dictionary with over/under-represented decades and magnitudes
        """
        all_decades = sorted(set(inferred.keys()) | set(truth.keys()))
        
        over_represented = {}
        under_represented = {}
        accurate = {}
        
        for decade in all_decades:
            inferred_val = inferred.get(decade, 0.0)
            truth_val = truth.get(decade, 0.0)
            diff = inferred_val - truth_val
            
            if diff > threshold:
                over_represented[decade] = diff
            elif diff < -threshold:
                under_represented[decade] = abs(diff)
            else:
                accurate[decade] = abs(diff)
        
        return {
            "over_represented": over_represented,
            "under_represented": under_represented,
            "accurate": accurate
        }
    
    def _visualize_distribution_comparison(self, 
                                         inferred: Dict[str, float], 
                                         truth: Dict[str, float],
                                         model_name: str):
        """
        Create a bar chart comparing inferred and ground truth distributions.
        
        Args:
            inferred: Inferred distribution
            truth: Ground truth distribution
            model_name: Name of model being evaluated
        """
        # Sort decades chronologically
        all_decades = sorted(set(inferred.keys()) | set(truth.keys()))
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Set bar width and positions
        bar_width = 0.35
        r1 = np.arange(len(all_decades))
        r2 = [x + bar_width for x in r1]
        
        # Create bars
        plt.bar(r1, [inferred.get(decade, 0) for decade in all_decades], 
                width=bar_width, label='Inferred', color='skyblue', alpha=0.8)
        plt.bar(r2, [truth.get(decade, 0) for decade in all_decades], 
                width=bar_width, label='Ground Truth', color='lightcoral', alpha=0.8)
        
        # Add labels and title
        plt.xlabel('Decade')
        plt.ylabel('Proportion')
        plt.title(f'Inferred vs Ground Truth Distribution: {model_name}')
        plt.xticks([r + bar_width/2 for r in r1], all_decades, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.results_dir / f"{model_name}_distribution_comparison.png")
        plt.close()
    
    def _visualize_decade_errors(self, decade_errors: Dict[str, float], model_name: str):
        """
        Create a visualization of per-decade errors.
        
        Args:
            decade_errors: Dictionary mapping decades to absolute errors
            model_name: Name of model being evaluated
        """
        # Sort decades chronologically
        decades = sorted(decade_errors.keys())
        errors = [decade_errors[decade] for decade in decades]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create color-coded bars based on error magnitude
        colors = [plt.cm.RdYlGn_r(err / max(errors)) if max(errors) > 0 else 'skyblue' for err in errors]
        plt.bar(decades, errors, color=colors)
        
        # Add labels and title
        plt.xlabel('Decade')
        plt.ylabel('Absolute Error')
        plt.title(f'Absolute Error by Decade: {model_name}')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.results_dir / f"{model_name}_decade_errors.png")
        plt.close()
    
    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results to disk."""
        model_name = results["model_name"]
        
        # Format decimal values for JSON serialization
        def format_values(obj):
            if isinstance(obj, dict):
                return {k: format_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [format_values(i) for i in obj]
            elif isinstance(obj, float):
                # Keep full precision for metrics, format percentages for distributions
                if abs(obj) < 1:
                    return obj  # Keep full precision for metrics
                else:
                    return float(f"{obj:.6f}")  # Format larger numbers
            else:
                return obj
        
        formatted_results = format_values(results)
        
        # Save as JSON
        with open(self.results_dir / f"{model_name}_evaluation.json", 'w') as f:
            json.dump(formatted_results, f, indent=2)
    
    def evaluate_multiple_models(self, 
                               results: List[Tuple[str, Dict[str, float], Dict[str, float]]]
                              ) -> Dict:
        """
        Evaluate and compare multiple models.
        
        Args:
            results: List of tuples containing (model_name, inferred_distribution, ground_truth)
            
        Returns:
            Dictionary with comparative evaluation metrics
        """
        all_evaluations = {}
        summary = {
            "models": [],
            "log10_mse": [],
            "mae": [],
            "js_distance": [],
            "rank_correlation": []
        }
        
        # Evaluate each model
        for model_name, inferred, ground_truth in results:
            evaluation = self.evaluate_distribution(inferred, ground_truth, model_name)
            all_evaluations[model_name] = evaluation
            
            # Extract key metrics for comparison
            summary["models"].append(model_name)
            summary["log10_mse"].append(evaluation["distribution_metrics"]["log10_mse"])
            summary["mae"].append(evaluation["distribution_metrics"]["mae"])
            summary["js_distance"].append(evaluation["distribution_metrics"]["js_distance"])
            summary["rank_correlation"].append(evaluation["decade_metrics"]["rank_correlation"])
        
        # Create comparative visualization
        self._visualize_model_comparison(summary)
        
        return {
            "individual_evaluations": all_evaluations,
            "summary": summary
        }
    
    def _visualize_model_comparison(self, summary: Dict):
        """
        Create a visualization comparing multiple models.
        
        Args:
            summary: Dictionary with model comparison metrics
        """
        if not summary["models"]:
            return
            
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot log10(MSE)
        axs[0, 0].bar(summary["models"], summary["log10_mse"], color='skyblue')
        axs[0, 0].set_title('log10(MSE) - Lower is Better')
        axs[0, 0].set_xticklabels(summary["models"], rotation=45)
        axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot MAE
        axs[0, 1].bar(summary["models"], summary["mae"], color='lightcoral')
        axs[0, 1].set_title('Mean Absolute Error - Lower is Better')
        axs[0, 1].set_xticklabels(summary["models"], rotation=45)
        axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot JS Distance
        axs[1, 0].bar(summary["models"], summary["js_distance"], color='lightgreen')
        axs[1, 0].set_title('Jensen-Shannon Distance - Lower is Better')
        axs[1, 0].set_xticklabels(summary["models"], rotation=45)
        axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot Rank Correlation
        axs[1, 1].bar(summary["models"], summary["rank_correlation"], color='plum')
        axs[1, 1].set_title('Rank Correlation - Higher is Better')
        axs[1, 1].set_xticklabels(summary["models"], rotation=45)
        axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "model_comparison.png")
        plt.close()

    def calculate_hayase_log_mse(self, inferred: Dict[str, float], ground_truth: Dict[str, float]) -> float:
        """
        Calculate log10(MSE) as used in Hayase et al. for direct comparison.
        
        Args:
            inferred: Dictionary mapping decades to inferred proportions
            ground_truth: Dictionary mapping decades to ground truth proportions
            
        Returns:
            log10(MSE) value
        """
        # Normalize distributions
        inferred_norm = self._normalize_distribution(inferred)
        ground_truth_norm = self._normalize_distribution(ground_truth)
        
        # Get all decades
        all_decades = sorted(set(inferred_norm.keys()) | set(ground_truth_norm.keys()))
        
        # Calculate MSE
        squared_errors = []
        for decade in all_decades:
            pred_val = inferred_norm.get(decade, 0.0)
            true_val = ground_truth_norm.get(decade, 0.0)
            squared_errors.append((pred_val - true_val) ** 2)
        
        mse = sum(squared_errors) / len(squared_errors)
        log10_mse = np.log10(mse) if mse > 0 else -float('inf')
        
        return log10_mse