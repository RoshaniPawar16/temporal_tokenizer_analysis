"""
Evaluation metrics for temporal distribution inference.

This module provides functions for evaluating the accuracy of inferred 
temporal distributions against known ground truth distributions.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon

from ..config import RESULTS_DIR

logger = logging.getLogger(__name__)

class TemporalEvaluationMetrics:
    """
    Implements metrics for evaluating temporal distribution inference accuracy.
    """
    
    def __init__(self):
        """Initialize with default settings."""
        # Set up results directory
        self.results_dir = RESULTS_DIR / "evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_distribution(self, 
                           inferred: Dict[str, float], 
                           ground_truth: Dict[str, float],
                           model_name: str = "model") -> Dict:
        """
        Evaluate inferred distribution against ground truth with multiple metrics.
        
        Args:
            inferred: Inferred temporal distribution
            ground_truth: Ground truth distribution
            model_name: Name of the model or method being evaluated
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Collect all decades from both distributions
        all_decades = sorted(set(inferred.keys()) | set(ground_truth.keys()))
        
        # Calculate distribution-level metrics
        mse, log10_mse = self._calculate_mse(inferred, ground_truth)
        mae = self._calculate_mae(inferred, ground_truth)
        js_distance = self._calculate_js_distance(inferred, ground_truth)
        
        # Calculate decade-level metrics
        decade_errors = self._calculate_decade_errors(inferred, ground_truth)
        rank_correlation = self._calculate_rank_correlation(inferred, ground_truth)
        representation_analysis = self._identify_representation_issues(inferred, ground_truth)
        
        # Collect metrics
        distribution_metrics = {
            "mse": mse,
            "log10_mse": log10_mse,
            "mae": mae,
            "js_distance": js_distance
        }
        
        decade_metrics = {
            "errors": decade_errors,
            "rank_correlation": rank_correlation,
            "representation_analysis": representation_analysis
        }
        
        # Visualize results
        self._create_evaluation_visualizations(
            inferred, ground_truth, decade_errors, 
            distribution_metrics, decade_metrics, model_name
        )
        
        return {
            "model_name": model_name,
            "distribution_metrics": distribution_metrics,
            "decade_metrics": decade_metrics,
            "inferred_distribution": inferred,
            "ground_truth_distribution": ground_truth
        }
    
    def evaluate_multiple_models(self, 
                              model_results: List[Tuple[str, Dict[str, float], Dict[str, float]]]) -> Dict:
        """
        Compare evaluations across multiple models.
        
        Args:
            model_results: List of tuples (model_name, inferred_distribution, ground_truth)
            
        Returns:
            Dictionary with comparative evaluation
        """
        # Evaluate each model
        evaluations = []
        for model_name, inferred, ground_truth in model_results:
            evaluation = self.evaluate_distribution(inferred, ground_truth, model_name)
            evaluations.append(evaluation)
        
        # Extract metric summaries
        summary = {
            "models": [e["model_name"] for e in evaluations],
            "log10_mse": [e["distribution_metrics"]["log10_mse"] for e in evaluations],
            "mae": [e["distribution_metrics"]["mae"] for e in evaluations],
            "js_distance": [e["distribution_metrics"]["js_distance"] for e in evaluations],
            "rank_correlation": [e["decade_metrics"]["rank_correlation"] for e in evaluations]
        }
        
        # Create comparative visualizations
        self._create_comparative_visualizations(evaluations)
        
        return {
            "evaluations": evaluations,
            "summary": summary
        }
    
    def _calculate_mse(self, 
                     inferred: Dict[str, float], 
                     ground_truth: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate Mean Squared Error and log10(MSE).
        
        Args:
            inferred: Inferred temporal distribution
            ground_truth: Ground truth distribution
            
        Returns:
            Tuple of (MSE, log10(MSE))
        """
        # Collect all decades
        all_decades = set(inferred.keys()) | set(ground_truth.keys())
        
        # Calculate squared errors
        squared_errors = []
        for decade in all_decades:
            inf_val = inferred.get(decade, 0.0)
            gt_val = ground_truth.get(decade, 0.0)
            squared_errors.append((inf_val - gt_val) ** 2)
        
        # Calculate MSE
        mse = sum(squared_errors) / len(squared_errors) if squared_errors else 0
        
        # Calculate log10(MSE) similar to Hayase et al.
        log10_mse = np.log10(mse) if mse > 0 else float('-inf')
        
        return mse, log10_mse
    
    def _calculate_mae(self, 
                     inferred: Dict[str, float], 
                     ground_truth: Dict[str, float]) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            inferred: Inferred temporal distribution
            ground_truth: Ground truth distribution
            
        Returns:
            Mean Absolute Error
        """
        # Collect all decades
        all_decades = set(inferred.keys()) | set(ground_truth.keys())
        
        # Calculate absolute errors
        abs_errors = []
        for decade in all_decades:
            inf_val = inferred.get(decade, 0.0)
            gt_val = ground_truth.get(decade, 0.0)
            abs_errors.append(abs(inf_val - gt_val))
        
        # Calculate MAE
        mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0
        
        return mae
    
    def _calculate_js_distance(self, 
                            inferred: Dict[str, float], 
                            ground_truth: Dict[str, float]) -> float:
        """
        Calculate Jensen-Shannon distance between distributions.
        
        Args:
            inferred: Inferred temporal distribution
            ground_truth: Ground truth distribution
            
        Returns:
            Jensen-Shannon distance
        """
        # Collect all decades
        all_decades = sorted(set(inferred.keys()) | set(ground_truth.keys()))
        
        # Create probability vectors
        p = np.array([inferred.get(decade, 0.0) for decade in all_decades])
        q = np.array([ground_truth.get(decade, 0.0) for decade in all_decades])
        
        # Ensure vectors sum to 1
        if np.sum(p) > 0:
            p = p / np.sum(p)
        if np.sum(q) > 0:
            q = q / np.sum(q)
        
        # Calculate Jensen-Shannon distance
        js_dist = jensenshannon(p, q)
        
        return js_dist
    
    def _calculate_decade_errors(self, 
                              inferred: Dict[str, float], 
                              ground_truth: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate errors for each decade.
        
        Args:
            inferred: Inferred temporal distribution
            ground_truth: Ground truth distribution
            
        Returns:
            Dictionary mapping decades to their absolute errors
        """
        # Collect all decades
        all_decades = set(inferred.keys()) | set(ground_truth.keys())
        
        # Calculate errors
        errors = {}
        for decade in all_decades:
            inf_val = inferred.get(decade, 0.0)
            gt_val = ground_truth.get(decade, 0.0)
            errors[decade] = abs(inf_val - gt_val)
        
        return errors
    
    def _calculate_rank_correlation(self, 
                                 inferred: Dict[str, float], 
                                 ground_truth: Dict[str, float]) -> float:
        """
        Calculate Spearman rank correlation between distributions.
        
        Args:
            inferred: Inferred temporal distribution
            ground_truth: Ground truth distribution
            
        Returns:
            Spearman rank correlation coefficient
        """
        # Collect all decades
        all_decades = sorted(set(inferred.keys()) | set(ground_truth.keys()))
        
        # Create value arrays
        inferred_values = [inferred.get(decade, 0.0) for decade in all_decades]
        truth_values = [ground_truth.get(decade, 0.0) for decade in all_decades]
        
        # Check if we have enough non-zero values
        if len(all_decades) <= 2 or sum(1 for v in inferred_values if v > 0) < 2 or sum(1 for v in truth_values if v > 0) < 2:
            return 0.0
        
        # Calculate Spearman correlation
        corr, _ = spearmanr(inferred_values, truth_values)
        
        # Handle NaN values
        if np.isnan(corr):
            return 0.0
            
        return corr
    
    def _identify_representation_issues(self, 
                                     inferred: Dict[str, float], 
                                     ground_truth: Dict[str, float]) -> Dict:
        """
        Identify over- and under-represented decades.
        
        Args:
            inferred: Inferred temporal distribution
            ground_truth: Ground truth distribution
            
        Returns:
            Dictionary with representation analysis
        """
        # Collect all decades
        all_decades = set(inferred.keys()) | set(ground_truth.keys())
        
        # Calculate representation issues
        over_represented = {}
        under_represented = {}
        
        for decade in all_decades:
            inf_val = inferred.get(decade, 0.0)
            gt_val = ground_truth.get(decade, 0.0)
            
            diff = inf_val - gt_val
            if diff > 0:
                over_represented[decade] = diff
            elif diff < 0:
                under_represented[decade] = abs(diff)
        
        return {
            "over_represented": over_represented,
            "under_represented": under_represented
        }
    
    def _create_evaluation_visualizations(self, 
                                       inferred: Dict[str, float], 
                                       ground_truth: Dict[str, float],
                                       decade_errors: Dict[str, float],
                                       distribution_metrics: Dict[str, float],
                                       decade_metrics: Dict,
                                       model_name: str):
        """
        Create evaluation visualizations.
        
        Args:
            inferred: Inferred temporal distribution
            ground_truth: Ground truth distribution
            decade_errors: Errors by decade
            distribution_metrics: Distribution-level metrics
            decade_metrics: Decade-level metrics
            model_name: Name of the model being evaluated
        """
        # Sort decades chronologically
        decades = sorted(set(inferred.keys()) | set(ground_truth.keys()))
        
        # Create figure for comparing distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Inferred vs Ground Truth
        inferred_values = [inferred.get(decade, 0) for decade in decades]
        truth_values = [ground_truth.get(decade, 0) for decade in decades]
        
        # Set bar positions
        bar_width = 0.35
        r1 = np.arange(len(decades))
        r2 = [x + bar_width for x in r1]
        
        # Create bars
        ax1.bar(r1, inferred_values, bar_width, label='Inferred', color='skyblue')
        ax1.bar(r2, truth_values, bar_width, label='Ground Truth', color='lightcoral')
        
        # Add labels
        for i, v in enumerate(inferred_values):
            ax1.text(i, v + 0.01, f"{v:.1%}", ha='center', fontsize=8)
        for i, v in enumerate(truth_values):
            ax1.text(i + bar_width, v + 0.01, f"{v:.1%}", ha='center', fontsize=8)
        
        ax1.set_xlabel('Decade')
        ax1.set_ylabel('Proportion')
        ax1.set_title(f'Inferred vs Ground Truth Distribution\n{model_name}')
        ax1.set_xticks([r + bar_width/2 for r in r1])
        ax1.set_xticklabels(decades, rotation=45)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Errors by decade
        error_values = [decade_errors.get(decade, 0) for decade in decades]
        colors = plt.cm.RdYlGn_r(np.array(error_values) / max(error_values) if max(error_values) > 0 else np.zeros(len(error_values)))
        
        ax2.bar(decades, error_values, color=colors)
        ax2.set_xlabel('Decade')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title(f'Errors by Decade\n{model_name}')
        ax2.set_xticks(decades)
        ax2.set_xticklabels(decades, rotation=45)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add data labels
        for i, v in enumerate(error_values):
            ax2.text(i, v + 0.005, f"{v:.1%}", ha='center', fontsize=8)
        
        # Add metrics as text annotation
        metrics_text = (
            f"log10(MSE): {distribution_metrics['log10_mse']:.2f}\n"
            f"MAE: {distribution_metrics['mae']:.4f}\n"
            f"JS Distance: {distribution_metrics['js_distance']:.4f}\n"
            f"Rank Correlation: {decade_metrics['rank_correlation']:.2f}"
        )
        
        # Add a text box for metrics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.95, 0.95, metrics_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.results_dir / f"{model_name}_evaluation.png", dpi=300)
        plt.close()
        
        # Create a second figure for representation analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot over-represented decades
        over_rep = decade_metrics["representation_analysis"]["over_represented"]
        if over_rep:
            over_decades = sorted(over_rep.keys())
            over_values = [over_rep[d] for d in over_decades]
            ax1.bar(over_decades, over_values, color='salmon')
            ax1.set_title('Over-represented Decades')
            ax1.set_xlabel('Decade')
            ax1.set_ylabel('Over-representation')
            ax1.set_xticks(over_decades)
            ax1.set_xticklabels(over_decades, rotation=45)
            
            # Add data labels
            for i, v in enumerate(over_values):
                ax1.text(i, v + 0.005, f"+{v:.1%}", ha='center')
        else:
            ax1.text(0.5, 0.5, "No over-represented decades", ha='center', va='center')
            ax1.set_title('Over-represented Decades')
        
        # Plot under-represented decades
        under_rep = decade_metrics["representation_analysis"]["under_represented"]
        if under_rep:
            under_decades = sorted(under_rep.keys())
            under_values = [under_rep[d] for d in under_decades]
            ax2.bar(under_decades, under_values, color='skyblue')
            ax2.set_title('Under-represented Decades')
            ax2.set_xlabel('Decade')
            ax2.set_ylabel('Under-representation')
            ax2.set_xticks(under_decades)
            ax2.set_xticklabels(under_decades, rotation=45)
            
            # Add data labels
            for i, v in enumerate(under_values):
                ax2.text(i, v + 0.005, f"-{v:.1%}", ha='center')
        else:
            ax2.text(0.5, 0.5, "No under-represented decades", ha='center', va='center')
            ax2.set_title('Under-represented Decades')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.results_dir / f"{model_name}_representation_analysis.png", dpi=300)
        plt.close()
    
    def _create_comparative_visualizations(self, evaluations: List[Dict]):
        """
        Create visualizations comparing different models.
        
        Args:
            evaluations: List of evaluation results for different models
        """
        # Extract model names
        model_names = [e["model_name"] for e in evaluations]
        
        # Create 2x2 grid of metric comparisons
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        
        # Plot log10(MSE)
        log10_mse_values = [e["distribution_metrics"]["log10_mse"] for e in evaluations]
        axs[0].bar(model_names, log10_mse_values, color='royalblue')
        axs[0].set_title('log10(MSE) Comparison\n(lower is better)')
        axs[0].set_ylabel('log10(MSE)')
        axs[0].set_xticklabels(model_names, rotation=45)
        
        # Plot MAE
        mae_values = [e["distribution_metrics"]["mae"] for e in evaluations]
        axs[1].bar(model_names, mae_values, color='royalblue')
        axs[1].set_title('Mean Absolute Error Comparison\n(lower is better)')
        axs[1].set_ylabel('MAE')
        axs[1].set_xticklabels(model_names, rotation=45)
        
        # Plot Jensen-Shannon Distance
        js_values = [e["distribution_metrics"]["js_distance"] for e in evaluations]
        axs[2].bar(model_names, js_values, color='royalblue')
        axs[2].set_title('Jensen-Shannon Distance Comparison\n(lower is better)')
        axs[2].set_ylabel('Jensen-Shannon Distance')
        axs[2].set_xticklabels(model_names, rotation=45)
        
        # Plot Rank Correlation
        corr_values = [e["decade_metrics"]["rank_correlation"] for e in evaluations]
        axs[3].bar(model_names, corr_values, color='royalblue')
        axs[3].set_title('Rank Correlation Comparison\n(higher is better)')
        axs[3].set_ylabel('Rank Correlation')
        axs[3].set_xticklabels(model_names, rotation=45)
        
        # Add grid to each subplot
        for ax in axs:
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.results_dir / "model_comparison.png", dpi=300)
        plt.close()