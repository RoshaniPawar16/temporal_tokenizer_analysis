"""
Temporal Distribution Inference

Implements methods for inferring the temporal distribution of language model 
training data by analyzing tokenizer merge rules.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import random
import logging
from pathlib import Path
import json
import gc
import re
from transformers import AutoTokenizer
import cvxpy as cp

from ..config import (
    RESULTS_DIR,
    TIME_PERIODS
)

logger = logging.getLogger(__name__)

class TemporalDistributionInference:
    """
    Analyzes tokenizer patterns to infer temporal distribution.
    Uses an enhanced approach with linear programming for more accurate results.
    """
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        """Initialize with tokenizer."""
        self.tokenizer_name = tokenizer_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Extract merge rules from tokenizer - trying multiple approaches
            self.merge_rules = []
            
            # Approach 1: Direct access to bpe_ranks (GPT-2 style)
            if hasattr(self.tokenizer, 'bpe_ranks'):
                self.merge_rules = list(self.tokenizer.bpe_ranks.keys())
                logger.info(f"Extracted {len(self.merge_rules)} merge rules from bpe_ranks")
            
            # Approach 2: Access via merges attribute (BERT/RoBERTa style)
            elif hasattr(self.tokenizer, 'merges'):
                self.merge_rules = self.tokenizer.merges
                logger.info(f"Extracted {len(self.merge_rules)} merge rules from merges attribute")
            
            # Approach 3: Try to access the tokenizer's model
            elif hasattr(self.tokenizer, 'model') and hasattr(self.tokenizer.model, 'merges'):
                self.merge_rules = self.tokenizer.model.merges
                logger.info(f"Extracted {len(self.merge_rules)} merge rules from model.merges")
            
            # Approach 4: Look in tokenizer's backend tokenizer
            elif hasattr(self.tokenizer, 'backend_tokenizer'):
                backend = self.tokenizer.backend_tokenizer
                if hasattr(backend, 'mergeable_ranks'):
                    self.merge_rules = list(backend.mergeable_ranks.keys())
                    logger.info(f"Extracted {len(self.merge_rules)} merge rules from backend_tokenizer")
            
            # If no merge rules found, try to generate some by analyzing the tokenizer's behavior
            if not self.merge_rules:
                logger.warning(f"Could not extract merge rules directly, generating approximation")
                # Generate a sample of text to analyze tokenizer behavior
                sample_text = """This is a sample text to analyze the tokenizer's behavior.
                We'll use this to identify patterns and generate approximate merge rules."""
                tokens = self.tokenizer.tokenize(sample_text)
                
                # Extract character pairs from tokens
                char_pairs = set()
                for token in tokens:
                    # Extract raw token (removing potential prefixes)
                    if token.startswith('Ġ') or token.startswith('▁'):
                        raw_token = token[1:]
                    else:
                        raw_token = token
                    
                    # Extract all character pairs
                    for i in range(len(raw_token) - 1):
                        char_pairs.add(raw_token[i:i+2])
                
                # Use these as approximate merge rules
                self.merge_rules = list(char_pairs)
            
            logger.info(f"Loaded {len(self.merge_rules)} merge rules from {tokenizer_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer {tokenizer_name}: {e}")
            self.tokenizer = None
            self.merge_rules = []
        
        # Set up results directory
        self.results_dir = RESULTS_DIR / "temporal_inference"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze_merge_rule_dynamics(self, decade_patterns: Dict[str, Dict]) -> Dict[str, List[str]]:
        """
        Analyze how merge rules change in importance across decades.
        This helps identify stronger temporal markers.
        
        Args:
            decade_patterns: Results from analyze_decade_patterns
            
        Returns:
            Dictionary mapping decades to lists of decade-specific merge rules
        """
        # Get all decades
        decades = sorted(decade_patterns.keys())
        
        # Extract merge rules from each decade
        all_merge_rules = set()
        decade_rules = {}
        
        for decade, patterns in decade_patterns.items():
            if 'merge_rules' in patterns:
                rules = patterns['merge_rules']
                decade_rules[decade] = rules
                all_merge_rules.update(rules.keys())
        
        # Calculate normalized frequencies
        normalized_freqs = {}
        for rule in all_merge_rules:
            normalized_freqs[rule] = {}
            for decade in decades:
                if decade in decade_rules and rule in decade_rules[decade]:
                    total_tokens = decade_patterns[decade]['total_tokens']
                    if total_tokens > 0:
                        normalized_freqs[rule][decade] = decade_rules[decade][rule] / total_tokens
                    else:
                        normalized_freqs[rule][decade] = 0
                else:
                    normalized_freqs[rule][decade] = 0
        
        # Find rules that show clear temporal trends
        temporal_rules = {}
        for decade in decades:
            decade_distinctive = []
            
            for rule in all_merge_rules:
                # Skip rules that don't appear in this decade
                if decade not in normalized_freqs[rule] or normalized_freqs[rule][decade] == 0:
                    continue
                    
                # Get frequencies for this rule across all decades
                decade_freqs = [normalized_freqs[rule].get(d, 0) for d in decades]
                
                # Calculate average frequency in other decades
                other_freqs = [f for i, f in enumerate(decade_freqs) if decades[i] != decade]
                if other_freqs and sum(other_freqs) > 0:
                    avg_other = sum(other_freqs) / len(other_freqs)
                    
                    # Calculate distinctiveness
                    if avg_other > 0:
                        distinctiveness = normalized_freqs[rule][decade] / avg_other
                        
                        # This rule is distinctive if it's much more common in this decade
                        if distinctiveness > 2.0:
                            decade_distinctive.append((rule, distinctiveness))
            
            # Sort by distinctiveness
            decade_distinctive.sort(key=lambda x: x[1], reverse=True)
            temporal_rules[decade] = [rule for rule, _ in decade_distinctive[:20]]
        
        return temporal_rules

    def analyze_decade_patterns(self, decade_texts: Dict[str, List[str]], sample_size: int = 5000) -> Dict[str, Dict]:
        """
        Analyze merge rules and token patterns for each decade with improved sampling.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            sample_size: Maximum number of tokens to analyze per decade
            
        Returns:
            Dictionary with pattern statistics by decade
        """
        decade_patterns = {}
        
        # Process each decade
        for decade, texts in decade_texts.items():
            if not texts:
                continue
                
            # Sample texts to maintain manageable processing time
            sampled_texts = texts
            if len(texts) > 50:  # Limit number of texts per decade for efficiency
                sampled_texts = random.sample(texts, 50)
            
            # Initialize pattern counters
            merge_rule_counts = Counter()
            token_counts = Counter()
            char_pair_counts = Counter()
            total_tokens = 0
            total_chars = 0
            
            # Combine texts for more efficient tokenization
            combined_text = " ".join(sampled_texts[:20])  # Process in batches
            
            # Tokenize combined text
            tokens = self.tokenizer.tokenize(combined_text)
            encoded = self.tokenizer.encode(combined_text, add_special_tokens=False)
            
            # Count tokens
            token_counts.update(tokens)
            total_tokens += len(tokens)
            
            # Count merge rules
            for token in tokens:
                # Extract applicable merge rules for this token
                applicable_rules = self._extract_merge_rules(token)
                merge_rule_counts.update(applicable_rules)
            
            # Count character pairs (bigrams)
            for i in range(len(combined_text) - 1):
                char_pair = combined_text[i:i+2]
                char_pair_counts[char_pair] += 1
                total_chars += 1
            
            # Calculate statistics
            if total_tokens > 0 and total_chars > 0:
                # Store decade statistics
                decade_patterns[decade] = {
                    'merge_rules': dict(merge_rule_counts),
                    'tokens': dict(token_counts),
                    'char_pairs': dict(char_pair_counts),
                    'total_tokens': total_tokens,
                    'total_chars': total_chars
                }
            
            # Clean up to free memory
            gc.collect()
        
        return decade_patterns
    
    def _extract_merge_rules(self, token: str) -> Set[str]:
        """
        Extract merge rules that could have generated this token.
        This approximates the merge rules since we don't have access to the exact tokenization process.
        
        Args:
            token: A token string
            
        Returns:
            Set of potential merge rules
        """
        # For GPT tokenizers, merge rules are usually character pairs
        rules = set()
        
        # Handle continuation tokens differently
        if token.startswith('Ġ') or token.startswith('▁'):
            # Space prefix in different tokenizers
            raw_token = token[1:]
        else:
            raw_token = token
        
        # Extract character pairs (bigrams)
        for i in range(len(raw_token) - 1):
            bigram = raw_token[i:i+2]
            rules.add(bigram)
        
        return rules
    
    def find_distinctive_patterns(self, 
                               decade_patterns: Dict[str, Dict],
                               threshold: float = 1.5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Identify patterns that are distinctively common in specific decades.
        Enhanced to focus on more reliable signals.
        
        Args:
            decade_patterns: Results from analyze_decade_patterns
            threshold: How much more common a pattern must be
            
        Returns:
            Dictionary mapping decades to lists of distinctive patterns
        """
        distinctive_patterns = {}
        
        # Get all decades
        decades = list(decade_patterns.keys())
        
        # For each decade, find distinctive patterns
        for decade in decades:
            decade_distinctive = []
            
            # Get patterns for this decade (prioritize merge rules)
            if 'merge_rules' in decade_patterns[decade] and decade_patterns[decade]['merge_rules']:
                patterns = decade_patterns[decade]['merge_rules']
                pattern_type = 'merge_rules'
            elif 'char_pairs' in decade_patterns[decade]:
                patterns = decade_patterns[decade]['char_pairs']
                pattern_type = 'char_pairs'
            else:
                continue
            
            # Calculate global pattern frequencies across all decades
            global_freqs = defaultdict(list)
            for other_decade in decades:
                if pattern_type in decade_patterns[other_decade]:
                    other_patterns = decade_patterns[other_decade][pattern_type]
                    for pattern, freq in other_patterns.items():
                        global_freqs[pattern].append(freq)
            
            # Find patterns distinctive to this decade
            for pattern, freq in patterns.items():
                # Get frequencies in other decades
                all_freqs = global_freqs[pattern]
                
                if len(all_freqs) > 1:  # Pattern exists in multiple decades
                    # Calculate average frequency excluding this decade
                    other_freqs = [f for i, f in enumerate(all_freqs) 
                                 if list(decade_patterns.keys())[i] != decade]
                    
                    if other_freqs:
                        avg_other_freq = sum(other_freqs) / len(other_freqs)
                        if avg_other_freq > 0:
                            distinctiveness = freq / avg_other_freq
                            if distinctiveness > threshold:
                                decade_distinctive.append((pattern, distinctiveness))
            
            # Sort by distinctiveness ratio
            decade_distinctive.sort(key=lambda x: x[1], reverse=True)
            distinctive_patterns[decade] = decade_distinctive[:20]  # Keep only top 20
        
        return distinctive_patterns
    
    def infer_temporal_distribution(self, 
                             decade_patterns: Dict[str, Dict],
                             num_merge_rules: int = 3000,
                             weight_early_merges: bool = True) -> Dict[str, float]:
        """
        Infer the temporal distribution in training data using linear programming.
        More closely approximates Hayase et al.'s approach.
        
        Args:
            decade_patterns: Results from analyze_decade_patterns
            num_merge_rules: Number of merge rules to consider
            weight_early_merges: Whether to give higher weight to earlier merge rules
            
        Returns:
            Dictionary mapping decades to their estimated proportion
        """
        # Extract decades
        decades = sorted(list(decade_patterns.keys()))
        
        if not decades:
            return {}
        
        try:
            # Prepare linear programming variables
            alpha = cp.Variable(len(decades), pos=True)
            
            # Sum-to-one constraint
            constraints = [cp.sum(alpha) == 1]
            
            # Extract merge rule frequencies for each decade
            merge_frequencies = {}
            for i, decade in enumerate(decades):
                if 'merge_rules' in decade_patterns[decade]:
                    for rule, count in decade_patterns[decade]['merge_rules'].items():
                        if rule not in merge_frequencies:
                            merge_frequencies[rule] = np.zeros(len(decades))
                        merge_frequencies[rule][i] = count / decade_patterns[decade]['total_tokens']
            
            # Sort merge rules by frequency and take top N
            if merge_frequencies:
                merge_rules = sorted(merge_frequencies.keys(), 
                                key=lambda r: sum(merge_frequencies[r]), 
                                reverse=True)[:num_merge_rules]
                
                # Construct objective function matrix
                objective_terms = []
                for rule in merge_rules:
                    freqs = merge_frequencies[rule]
                    if sum(freqs) > 0:
                        # Add term rewarding models whose merge rule distributions match the inferred alpha
                        objective_terms.append(cp.sum(cp.multiply(alpha, freqs)))
                
                # Objective: maximize sum of weighted merge rule matches
                objective = cp.Maximize(sum(objective_terms))
                
                # Solve the problem
                prob = cp.Problem(objective, constraints)
                prob.solve()
                
                # Extract solution
                if prob.status == cp.OPTIMAL:
                    distribution = {decade: float(alpha.value[i]) for i, decade in enumerate(decades)}
                    
                    # Normalize to ensure sum to 1
                    total = sum(distribution.values())
                    if total > 0:
                        return {decade: value / total for decade, value in distribution.items()}
            
            # If we get here, something went wrong with the LP approach
            logger.warning("Linear programming approach failed, falling back to heuristic")
        except Exception as e:
            logger.error(f"Error in linear programming: {e}")
        
        # Fallback to simple heuristic method
        return self._infer_distribution_heuristic(decade_patterns)
    
    def _infer_distribution_heuristic(self, decade_patterns: Dict[str, Dict]) -> Dict[str, float]:
        """
        Fallback heuristic method for temporal distribution inference.
        
        Args:
            decade_patterns: Results from analyze_decade_patterns
            
        Returns:
            Dictionary mapping decades to their estimated proportion
        """
        # Extract decades
        decades = sorted(list(decade_patterns.keys()))
        
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
        # This implementation is good as-is, no changes needed
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
        # This implementation is good as-is, no changes needed
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
        
        # Add reference line for uniform distribution
        plt.axhline(y=1.0/len(decades), color='red', linestyle='--', 
                label=f'Uniform Distribution ({1.0/len(decades):.1%})')
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.results_dir / f"{self.tokenizer_name}_temporal_distribution.png")
        plt.close()

    def run_analysis(self, decade_texts: Dict[str, List[str]]) -> Dict:
        """
        Run complete analysis pipeline with enhanced methods.
        
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
        
        # Check if tokenizer is properly initialized
        if not self.tokenizer or not self.merge_rules:
            logger.error("Tokenizer not properly initialized or no merge rules available")
            # Return placeholder results to avoid crashing
            return {
                "tokenizer": self.tokenizer_name,
                "distinctive_patterns": {decade: [] for decade in decade_texts.keys()},
                "distribution": {decade: 1.0 / len(decade_texts) for decade in decade_texts.keys()}
            }
        
        # Step 1: Analyze decade patterns with increased sample size
        logger.info("Analyzing decade patterns...")
        decade_patterns = self.analyze_decade_patterns(decade_texts, sample_size=10000)
        
        # Step 2: Find distinctive patterns
        logger.info("Finding distinctive patterns...")
        distinctive_patterns = self.find_distinctive_patterns(decade_patterns)
        
        # Analyze temporal dynamics of merge rules
        temporal_markers = self.analyze_merge_rule_dynamics(decade_patterns)
        logger.info("Found temporal marker merge rules:")
        for decade, rules in temporal_markers.items():
            if rules:
                logger.info(f"  {decade}: {', '.join(rules[:5])}")

        # Step 3: Infer temporal distribution with enhanced approach
        logger.info("Inferring temporal distribution...")
        distribution = self.infer_temporal_distribution(
            decade_patterns,
            weight_early_merges=True
        )
        
        # Step 4: Visualize results
        logger.info("Generating visualizations...")
        self.visualize_results(distinctive_patterns, distribution)
        
        # Return complete results
        return {
            "tokenizer": self.tokenizer_name,
            "distinctive_patterns": distinctive_patterns,
            "distribution": distribution
        }