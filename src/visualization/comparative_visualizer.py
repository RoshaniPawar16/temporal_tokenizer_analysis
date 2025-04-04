# src/visualization/comparative_visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ComparativeVisualizer:
    """
    Create comparison visualizations across different models or methods.
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
    
    def compare_model_metrics(self, 
                          model_metrics, 
                          save_path=None):
        """
        Compare metrics across multiple models.
        
        Args:
            model_metrics: Dict mapping model names to dicts of metrics
            save_path: Path to save the visualization
        """
        # Extract model names and metrics
        models = list(model_metrics.keys())
        metrics = list(set().union(*[set(m.keys()) for m in model_metrics.values()]))
        
        # Filter metrics to common ones
        common_metrics = [m for m in metrics 
                        if all(m in model_metrics[model] for model in models)]
        
        if not common_metrics:
            logger.warning("No common metrics found across models")
            return None
        
        # Create multi-panel figure
        fig, axes = plt.subplots(len(common_metrics), 1, 
                              figsize=(10, 5 * len(common_metrics)))
        
        # Convert to list for consistent indexing
        if len(common_metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(common_metrics):
            ax = axes[i]
            values = [model_metrics[model][metric] for model in models]
            
            # Plot bar chart
            ax.bar(models, values, color='steelblue')
            
            # Add data labels
            for j, v in enumerate(values):
                ax.text(j, v, f"{v:.4f}", ha='center', va='bottom')
            
            # Add title and labels
            ax.set_title(f'{metric} Comparison')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison saved to {save_path}")
        
        # Return the figure for display
        return fig
    
    def compare_distributions(self, 
                          model_distributions, 
                          ground_truth=None,
                          save_path=None):
        """
        Compare distributions across multiple models.
        
        Args:
            model_distributions: Dict mapping model names to distribution dicts
            ground_truth: Optional ground truth distribution for reference
            save_path: Path to save the visualization
        """
        # Get all decades from all distributions
        decades = sorted(set().union(*[set(dist.keys()) for dist in model_distributions.values()]))
        
        if ground_truth:
            decades = sorted(set(decades) | set(ground_truth.keys()))
        
        # Create a dataframe for easier plotting
        import pandas as pd
        df = pd.DataFrame(index=decades)
        
        # Add each model's distribution
        for model, dist in model_distributions.items():
            df[model] = [dist.get(decade, 0) for decade in decades]
        
        # Add ground truth if provided
        if ground_truth:
            df['Ground Truth'] = [ground_truth.get(decade, 0) for decade in decades]
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Plot grouped bar chart
        df.plot(kind='bar', figsize=(14, 8), ax=plt.gca())
        
        # Add title and labels
        plt.title('Distribution Comparison Across Models')
        plt.xlabel('Decade')
        plt.ylabel('Proportion')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Model')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution comparison saved to {save_path}")
        
        # Return the figure for display
        return plt.gcf()