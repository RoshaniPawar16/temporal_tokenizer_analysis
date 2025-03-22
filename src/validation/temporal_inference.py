"""
Temporal Distribution Inference

Implements methods for inferring the temporal distribution of language model 
training data by analyzing tokenizer merge rules.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import random
import logging
from pathlib import Path
import json
import gc

from transformers import AutoTokenizer

from ..config import (
    RESULTS_DIR,
    TIME_PERIODS
)

logger = logging.getLogger(__name__)

class TemporalDistributionInference:
    """
    Analyzes tokenizer patterns to infer temporal distribution.
    Uses a simplified approach suitable for resource-constrained environments.
    """
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        """Initialize with tokenizer."""
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set up results directory
        self.results_dir = RESULTS_DIR / "temporal_inference"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_decade_patterns(self, decade_texts: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Analyze character and token patterns for each decade.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            
        Returns:
            Dictionary with pattern statistics by decade
        """
        decade_patterns = {}
        
        # Process each decade
        for decade, texts in decade_texts.items():
            if not texts:
                continue
                
            # Initialize pattern counters
            char_pair_counts = Counter()
            token_counts = Counter()
            total_chars = 0
            total_tokens = 0
            
            # Process each text
            for text in texts:
                # Limit text length for processing efficiency
                text = text[:5000]
                
                # Count character pairs
                for i in range(len(text) - 1):
                    char_pair = text[i:i+2]
                    char_pair_counts[char_pair] += 1
                    total_chars += 1
                
                # Analyze tokenization patterns
                tokens = self.tokenizer.tokenize(text)
                token_counts.update(tokens)
                total_tokens += len(tokens)
            
            # Calculate statistics
            if total_chars > 0:
                # Normalize character pair counts
                normalized_pairs = {pair: count / total_chars 
                                 for pair, count in char_pair_counts.items()}
                
                # Store decade statistics
                decade_patterns[decade] = {
                    'char_pairs': dict(normalized_pairs),
                    'token_counts': dict(token_counts),
                    'total_chars': total_chars,
                    'total_tokens': total_tokens
                }
            
            # Clean up to free memory
            gc.collect()
        
        return decade_patterns
    
    def find_distinctive_patterns(self, 
                                decade_patterns: Dict[str, Dict],
                                threshold: float = 1.5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Identify patterns that are distinctively common in specific decades.
        
        Args:
            decade_patterns: Results from analyze_decade_patterns
            threshold: How much more common a pattern must be
            
        Returns:
            Dictionary mapping decades to lists of distinctive patterns
        """
        distinctive_patterns = {}
        
        # Get all decades
        decades = list(decade_patterns.keys())
        
        # For each decade, find distinctive character pairs
        for decade in decades:
            decade_distinctive = []
            
            # Get character pairs for this decade
            char_pairs = decade_patterns[decade]['char_pairs']
            
            for pair, freq in char_pairs.items():
                # Calculate average frequency in other decades
                other_freqs = []
                for other_decade in decades:
                    if other_decade != decade and other_decade in decade_patterns:
                        other_pairs = decade_patterns[other_decade]['char_pairs']
                        if pair in other_pairs:
                            other_freqs.append(other_pairs[pair])
                
                # If pair exists in other decades, check if it's distinctive
                if other_freqs:
                    avg_other_freq = sum(other_freqs) / len(other_freqs)
                    if avg_other_freq > 0:
                        ratio = freq / avg_other_freq
                        if ratio > threshold:
                            decade_distinctive.append((pair, ratio))
            
            # Sort by distinctiveness ratio
            distinctive_patterns[decade] = sorted(decade_distinctive, 
                                              key=lambda x: x[1], 
                                              reverse=True)
        
        return distinctive_patterns
    
    def infer_temporal_distribution(self, decade_patterns: Dict[str, Dict]) -> Dict[str, float]:
        """
        Infer the temporal distribution in training data.
        Uses a simplified heuristic approach suitable for resource-constrained environments.
        
        Args:
            decade_patterns: Results from analyze_decade_patterns
            
        Returns:
            Dictionary mapping decades to their estimated proportion
        """
        # Extract decades
        decades = sorted(list(decade_patterns.keys()))
        
        if not decades:
            return {}
            
        # Calculate distinctive pattern scores for each decade
        distinctive_patterns = self.find_distinctive_patterns(decade_patterns)
        distinctive_scores = {}
        
        for decade, patterns in distinctive_patterns.items():
            if patterns:
                # Use sum of top pattern distinctiveness scores
                top_patterns = patterns[:min(10, len(patterns))]
                score = sum(score for _, score in top_patterns)
                distinctive_scores[decade] = score
            else:
                distinctive_scores[decade] = 1.0  # Default if no distinctive patterns
        
        # Normalize scores to get proportions
        total_score = sum(distinctive_scores.values())
        if total_score > 0:
            proportions = {decade: score / total_score for decade, score in distinctive_scores.items()}
        else:
            # Fallback to uniform distribution
            proportions = {decade: 1.0 / len(decades) for decade in decades}
        
        return proportions
    
    def visualize_results(self, 
                        distinctive_patterns: Dict[str, List[Tuple[str, float]]],
                        distribution: Dict[str, float]):
        """
        Visualize the analysis results.
        
        Args:
            distinctive_patterns: Results from find_distinctive_patterns
            distribution: Inferred temporal distribution
        """
        self._visualize_distinctive_patterns(distinctive_patterns)
        self._visualize_temporal_distribution(distribution)
    
    def _visualize_distinctive_patterns(self, distinctive_patterns: Dict[str, List[Tuple[str, float]]]):
        """Visualize distinctive patterns for each decade."""
        # Sort decades chronologically
        decades = sorted(distinctive_patterns.keys())
        
        # Create figure
        plt.figure(figsize=(10, len(decades) * 0.7))
        
        # Plot data
        current_pos = 0
        labels = []
        values = []
        colors = []
        
        for i, decade in enumerate(decades):
            # Get top distinctive patterns (limit to 5)
            top_patterns = distinctive_patterns[decade][:5]
            
            if not top_patterns:
                continue
                
            # Add patterns to plot data
            for j, (pattern, score) in enumerate(top_patterns):
                labels.append(f"{decade}: '{pattern}'")
                values.append(score)
                colors.append(plt.cm.viridis(i / len(decades)))
                current_pos += 1
        
        # Create horizontal bar chart
        plt.barh(labels, values, color=colors)
        
        # Add labels and title
        plt.xlabel('Distinctiveness Score (higher = more decade-specific)')
        plt.title(f'Most Distinctive Patterns by Decade ({self.tokenizer_name})')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.results_dir / f"{self.tokenizer_name}_distinctive_patterns.png")
        plt.close()
    
    def _visualize_temporal_distribution(self, distribution: Dict[str, float]):
        """Visualize inferred temporal distribution."""
        # Sort decades chronologically
        decades = sorted(distribution.keys())
        proportions = [distribution[decade] for decade in decades]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot bar chart
        plt.bar(decades, proportions, color='skyblue')
        
        # Add data labels
        for i, v in enumerate(proportions):
            plt.text(i, v + 0.01, f"{v:.1%}", ha='center')
        
        # Add title and labels
        plt.title(f'Inferred Temporal Distribution in {self.tokenizer_name} Training Data')
        plt.xlabel('Decade')
        plt.ylabel('Estimated Proportion')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.results_dir / f"{self.tokenizer_name}_temporal_distribution.png")
        plt.close()
    
    def run_analysis(self, decade_texts: Dict[str, List[str]]) -> Dict:
        """
        Run complete analysis pipeline.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            
        Returns:
            Complete analysis results
        """
        # Filter to non-empty decades
        decade_texts = {decade: texts for decade, texts in decade_texts.items() if texts}
        
        if not decade_texts:
            logger.warning("No data available for analysis")
            return {}
        
        # Step 1: Analyze decade patterns
        logger.info("Analyzing decade patterns...")
        decade_patterns = self.analyze_decade_patterns(decade_texts)
        
        # Step 2: Find distinctive patterns
        logger.info("Finding distinctive patterns...")
        distinctive_patterns = self.find_distinctive_patterns(decade_patterns)
        
        # Step 3: Infer temporal distribution
        logger.info("Inferring temporal distribution...")
        distribution = self.infer_temporal_distribution(decade_patterns)
        
        # Step 4: Visualize results
        logger.info("Generating visualizations...")
        self.visualize_results(distinctive_patterns, distribution)
        
        # Return complete results
        return {
            "tokenizer": self.tokenizer_name,
            "distinctive_patterns": distinctive_patterns,
            "distribution": distribution
        }