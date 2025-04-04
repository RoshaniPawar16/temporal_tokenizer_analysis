# src/visualization/token_visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TokenVisualizer:
    """
    Visualize token usage patterns and distributions across time periods.
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
        
    def visualize_token_distinctiveness(self, 
                                     distinctive_tokens, 
                                     token_to_text=None,
                                     save_path=None):
        """
        Visualize distinctive tokens for each period.
        
        Args:
            distinctive_tokens: Dict mapping periods to lists of (token_id, distinctiveness)
            token_to_text: Optional dict mapping token IDs to readable text
            save_path: Path to save the visualization
        """
        # Sort periods chronologically
        periods = sorted(distinctive_tokens.keys())
        
        # Create figure
        plt.figure(figsize=(10, len(periods) * 0.7))
        
        # Plot data
        current_pos = 0
        labels = []
        values = []
        colors = []
        
        for i, period in enumerate(periods):
            # Get top distinctive patterns (limit to 5)
            top_tokens = distinctive_tokens[period][:5]
            
            if not top_tokens:
                continue
                
            # Add patterns to plot data
            for j, (token, token_text, score) in enumerate(top_tokens):
                # Get readable text if needed
                if token_to_text is not None and token in token_to_text:
                    token_text = token_to_text[token]
                    
                labels.append(f"{period}: '{token_text}'")
                values.append(score)
                colors.append(plt.cm.viridis(i / len(periods)))
                current_pos += 1
        
        # Create horizontal bar chart
        plt.barh(labels, values, color=colors)
        
        # Add labels and title
        plt.xlabel('Distinctiveness Score (higher = more period-specific)')
        plt.title('Most Distinctive Tokens by Period')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        # Return the figure for display
        return plt.gcf()
    
    def visualize_temporal_heatmap(self,
                                period_token_stats,
                                top_tokens=None,
                                token_to_text=None,
                                save_path=None):
        """
        Create a heatmap visualization of token usage across time periods.
        
        Args:
            period_token_stats: Dict of token statistics by period
            top_tokens: Optional list of tokens to include (default: auto-selected)
            token_to_text: Optional dict mapping token IDs to readable text
            save_path: Path to save the visualization
        """
        periods = list(period_token_stats.keys())
        n_periods = len(periods)
        
        if n_periods == 0:
            logger.warning("No periods with data to visualize")
            return None
        
        # If no top tokens provided, select them automatically
        if top_tokens is None:
            # Extract top tokens across all periods
            all_top_tokens = []
            for period in periods:
                # Get top tokens by frequency
                if 'token_frequencies' in period_token_stats[period]:
                    freqs = period_token_stats[period]['token_frequencies']
                    top_period_tokens = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:5]
                    all_top_tokens.extend([t[0] for t in top_period_tokens])
            
            # Remove duplicates while preserving order
            unique_top_tokens = []
            for token in all_top_tokens:
                if token not in unique_top_tokens:
                    unique_top_tokens.append(token)
            
            # Limit to top 20 tokens overall
            top_tokens = unique_top_tokens[:20]
        
        # Create heatmap data
        heatmap_data = np.zeros((len(top_tokens), len(periods)))
        for i, token in enumerate(top_tokens):
            for j, period in enumerate(periods):
                # Get normalized frequency
                if 'token_frequencies' in period_token_stats[period]:
                    frequency = period_token_stats[period]['token_frequencies'].get(token, 0)
                    heatmap_data[i, j] = frequency
        
        # Get readable token texts
        token_texts = []
        for token in top_tokens:
            if token_to_text is not None and token in token_to_text:
                text = token_to_text[token]
            else:
                text = str(token)
            token_texts.append(text)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", 
                   xticklabels=periods, yticklabels=token_texts)
        plt.title('Token Usage Frequency Across Time Periods', fontsize=14)
        plt.xlabel('Time Period')
        plt.ylabel('Token')
        plt.tight_layout()
        
        # Save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap visualization saved to {save_path}")
        
        # Return the figure for display
        return plt.gcf()