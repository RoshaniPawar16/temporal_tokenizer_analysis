# src/visualization/temporal_distribution_visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TemporalDistributionVisualizer:
    """
    Visualize inferred temporal distributions and comparison with ground truth.
    """
    
    def __init__(self, results_dir=None):
        """
        Initialize the visualizer with output directory.
        
        Args:
            results_dir: Directory to save visualizations (default: results/visualizations)
        """
        if results_dir is None:
            results_dir = Path("results/visualizations")
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_distribution(self, 
                           distribution, 
                           title=None,
                           save_path=None):
        """
        Visualize a temporal distribution.
        
        Args:
            distribution: Dictionary mapping decades to proportions
            title: Optional title for the plot
            save_path: Path to save the visualization
        """
        # Sort decades chronologically
        decades = sorted(distribution.keys())
        proportions = [distribution[decade] for decade in decades]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot bar chart
        bars = plt.bar(decades, proportions, color='skyblue')
        
        # Add data labels
        for i, v in enumerate(proportions):
            plt.text(i, v + 0.01, f"{v:.1%}", ha='center')
    
        # Add title and labels
        plt.title(title or 'Temporal Distribution')
        plt.xlabel('Decade')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add reference line for uniform distribution
        uniform_value = 1.0/len(decades)
        plt.axhline(y=uniform_value, color='red', linestyle='--', 
                label=f'Uniform Distribution ({uniform_value:.1%})')
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution visualization saved to {save_path}")
        
        # Return the figure for display
        return plt.gcf()
    
    def compare_distributions(self,
                          inferred,
                          ground_truth,
                          title=None,
                          save_path=None):
        """
        Compare inferred and ground truth distributions.
        
        Args:
            inferred: Dictionary mapping decades to inferred proportions
            ground_truth: Dictionary mapping decades to ground truth proportions
            title: Optional title for the plot
            save_path: Path to save the visualization
        """
        # Sort decades chronologically
        decades = sorted(set(inferred.keys()) | set(ground_truth.keys()))
        
        # Get values for each decade
        inferred_values = [inferred.get(decade, 0) for decade in decades]
        truth_values = [ground_truth.get(decade, 0) for decade in decades]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Set bar positions
        x = np.arange(len(decades))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, inferred_values, width, label='Inferred', color='skyblue')
        plt.bar(x + width/2, truth_values, width, label='Ground Truth', color='lightcoral')
        
        # Add labels
        for i, v in enumerate(inferred_values):
            plt.text(i - width/2, v + 0.01, f"{v:.1%}", ha='center')
        for i, v in enumerate(truth_values):
            plt.text(i + width/2, v + 0.01, f"{v:.1%}", ha='center')
        
        # Add title and labels
        plt.title(title or 'Inferred vs Ground Truth Distribution')
        plt.xlabel('Decade')
        plt.ylabel('Proportion')
        plt.xticks(x, decades, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution comparison saved to {save_path}")
        
        # Return the figure for display
        return plt.gcf()
    
    def visualize_error_analysis(self,
                             inferred,
                             ground_truth,
                             metrics=None,
                             save_path=None):
        """
        Visualize error analysis for inferred vs ground truth distribution.
        
        Args:
            inferred: Dictionary mapping decades to inferred proportions
            ground_truth: Dictionary mapping decades to ground truth proportions
            metrics: Optional dictionary of evaluation metrics
            save_path: Path to save the visualization
        """
        # Sort decades chronologically
        decades = sorted(set(inferred.keys()) | set(ground_truth.keys()))
        
        # Calculate errors
        errors = {decade: abs(inferred.get(decade, 0) - ground_truth.get(decade, 0)) 
                for decade in decades}
        error_values = [errors[decade] for decade in decades]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot errors
        bars = plt.bar(decades, error_values, 
                     color=plt.cm.RdYlGn_r(np.array(error_values) / max(error_values) 
                                         if max(error_values) > 0 
                                         else np.zeros(len(error_values))))
        
        # Add data labels
        for i, v in enumerate(error_values):
            plt.text(i, v + 0.005, f"{v:.1%}", ha='center')
        
        # Add title and labels
        plt.title('Absolute Error by Decade')
        plt.xlabel('Decade')
        plt.ylabel('Absolute Error')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add metrics as text annotation if provided
        if metrics:
            metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.95, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error analysis saved to {save_path}")
        
        # Return the figure for display
        return plt.gcf()