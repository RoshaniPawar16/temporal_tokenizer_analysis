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
import os

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
            # Load the tokenizer with additional options
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            
            # Extract merge rules - trying multiple approaches
            self.merge_rules = []
            
            # Method 1: GPT-2 style tokenizers often have bpe_ranks
            if hasattr(self.tokenizer, 'bpe_ranks'):
                self.merge_rules = list(self.tokenizer.bpe_ranks.keys())
                logger.info(f"Extracted {len(self.merge_rules)} merge rules from bpe_ranks")
            
            # Method 2: Access via backend tokenizer for HuggingFace's fast tokenizers
            elif hasattr(self.tokenizer, 'backend_tokenizer'):
                backend = self.tokenizer.backend_tokenizer
                if hasattr(backend, 'mergeable_ranks'):
                    self.merge_rules = list(backend.mergeable_ranks.keys())
                    logger.info(f"Extracted {len(self.merge_rules)} merge rules from backend_tokenizer")
                elif hasattr(backend, 'model') and hasattr(backend.model, 'merges'):
                    self.merge_rules = backend.model.merges
                    logger.info(f"Extracted {len(self.merge_rules)} merge rules from backend model")
            
            # Method 3: Some tokenizers store merges directly
            elif hasattr(self.tokenizer, 'merges'):
                self.merge_rules = self.tokenizer.merges
                logger.info(f"Extracted {len(self.merge_rules)} merge rules from merges attribute")
            
            # Method 4: Try to directly access the vocab file and parse merge rules
            if not self.merge_rules:
                try:
                    # For GPT-2, try to load the merges file directly
                    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
                    # Import our custom cached_path function
                    from fix_transformers import get_cached_path
                    cached_path = get_cached_path()
                    
                    # First, explicitly download the tokenizer files if needed
                    from huggingface_hub import hf_hub_download
                    try:
                        merges_file = hf_hub_download(repo_id=tokenizer_name, filename="merges.txt")
                        logger.info(f"Successfully downloaded merges.txt from HuggingFace Hub")
                    except Exception as e:
                        logger.warning(f"Could not download from Hub: {e}")
                        # Try to find the file in standard location
                        vocab_files = self.tokenizer.vocab_files_names
                        merges_file = vocab_files.get('merges_file', f"https://huggingface.co/{tokenizer_name}/resolve/main/merges.txt")
                        merges_file = cached_path(merges_file) if callable(cached_path) else merges_file
                    
                    if os.path.exists(merges_file):
                        with open(merges_file, encoding='utf-8') as f:
                            bpe_merges = f.read().split('\n')[1:-1]
                            bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
                            self.merge_rules = bpe_merges
                            logger.info(f"Extracted {len(self.merge_rules)} merge rules from merges file")
                    else:
                        logger.warning(f"Merges file {merges_file} does not exist")
                except Exception as e:
                    logger.warning(f"Could not load merges file: {e}")
            
            # If still no merge rules, create synthetic ones by analyzing tokenizer behavior
            if not self.merge_rules:
                logger.warning(f"Could not extract merge rules directly, generating approximation")
                # Create a diverse sample of text to find patterns
                sample_texts = [
                    "The quick brown fox jumps over the lazy dog.",
                    "Programming languages like Python, Java, and C++ are widely used.",
                    "In 1956, artificial intelligence research began in earnest.",
                    "The human genome contains approximately 3 billion DNA base pairs.",
                    "Blockchain technology enables secure, decentralized transactions.",
                    "Quantum computers leverage superposition and entanglement principles."
                ]
                
                # Analyze tokenization patterns
                all_tokens = []
                for text in sample_texts:
                    all_tokens.extend(self.tokenizer.tokenize(text))
                
                # Generate approximate merge rules from tokens
                char_pairs = set()
                for token in all_tokens:
                    # Handle different tokenizer prefixes
                    if token.startswith('Ġ') or token.startswith('▁'):
                        raw_token = token[1:]
                    else:
                        raw_token = token
                    
                    # Extract character pairs
                    for i in range(len(raw_token) - 1):
                        char_pairs.add(raw_token[i:i+2])
                
                # Use these character pairs as approximate merge rules
                self.merge_rules = list(char_pairs)
                logger.warning(f"Created {len(self.merge_rules)} synthetic merge rules")
            
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
            if 'merge_rules' in decade_patterns[decade]:
                patterns = decade_patterns[decade]['merge_rules']
                pattern_type = 'merge_rules'
            elif 'char_pairs' in decade_patterns[decade]:
                patterns = decade_patterns[decade]['char_pairs']
                pattern_type = 'char_pairs'
            else:
                continue
            
            # Calculate global pattern frequencies across all decades
            global_freqs = {}
            for other_decade in decades:
                if pattern_type in decade_patterns[other_decade]:
                    other_patterns = decade_patterns[other_decade][pattern_type]
                    total_tokens = decade_patterns[other_decade]['total_tokens']
                    for pattern, freq in other_patterns.items():
                        if pattern not in global_freqs:
                            global_freqs[pattern] = []
                        # Store normalized frequency (by total tokens)
                        if total_tokens > 0:
                            norm_freq = freq / total_tokens
                        else:
                            norm_freq = 0
                        global_freqs[pattern].append((other_decade, norm_freq))
            
            # Find patterns distinctive to this decade
            for pattern, freq in patterns.items():
                # Skip patterns with too few occurrences
                if freq < 3:  # Require at least 3 occurrences
                    continue
                    
                # Get this decade's normalized frequency
                this_norm_freq = freq / decade_patterns[decade]['total_tokens'] if decade_patterns[decade]['total_tokens'] > 0 else 0
                
                # Get normalized frequencies in other decades
                other_decades_norm_freqs = [f for d, f in global_freqs.get(pattern, []) if d != decade]
                
                if other_decades_norm_freqs:
                    avg_other_freq = sum(other_decades_norm_freqs) / len(other_decades_norm_freqs)
                    
                    # Calculate distinctiveness (ratio to average in other decades)
                    if avg_other_freq > 0:
                        distinctiveness = this_norm_freq / avg_other_freq
                        
                        # This rule is distinctive if it's much more common in this decade
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
            
            # Add minimum probability constraint to prevent zeros
            min_prob = 0.01  # Minimum 1% probability for any decade
            constraints.extend([alpha[i] >= min_prob for i in range(len(decades))])
            
            # Extract merge rule frequencies for each decade
            merge_frequencies = {}
            for i, decade in enumerate(decades):
                if 'merge_rules' in decade_patterns[decade]:
                    for rule, count in decade_patterns[decade]['merge_rules'].items():
                        if rule not in merge_frequencies:
                            merge_frequencies[rule] = np.zeros(len(decades))
                        # Normalize by total tokens in this decade
                        if decade_patterns[decade]['total_tokens'] > 0:
                            merge_frequencies[rule][i] = count / decade_patterns[decade]['total_tokens']
            
            # Calculate distinctiveness for each rule (how much it varies across decades)
            distinctiveness = {}
            for rule, freqs in merge_frequencies.items():
                if np.sum(freqs) > 0:
                    max_val = np.max(freqs)
                    max_idx = np.argmax(freqs)
                    other_vals = np.delete(freqs, max_idx)
                    mean_others = np.mean(other_vals) if len(other_vals) > 0 else 0.0001
                    # Distinctiveness is ratio of max to mean of others (capped to avoid extreme values)
                    distinctiveness[rule] = min(max_val / mean_others if mean_others > 0 else 1.0, 5.0)
            
            # Sort merge rules by a combination of frequency and distinctiveness
            rule_scores = {}
            for rule in merge_frequencies.keys():
                freqs = merge_frequencies[rule]
                overall_freq = np.sum(freqs)
                distinct_score = distinctiveness.get(rule, 1.0)
                # Balance between frequency and distinctiveness
                rule_scores[rule] = overall_freq * np.log1p(distinct_score)
            
            # Select top rules by this combined score
            merge_rules = sorted(rule_scores.keys(), 
                            key=lambda r: rule_scores[r], 
                            reverse=True)[:num_merge_rules]
            
            # Construct objective function with weighted terms
            objective_terms = []
            for idx, rule in enumerate(merge_rules):
                freqs = merge_frequencies[rule]
                distinct_factor = distinctiveness.get(rule, 1.0)
                
                # Apply early merge weight if enabled
                position_weight = 1.0
                if weight_early_merges:
                    position_weight = 1.0 - (idx / (2 * num_merge_rules))  # Linear decay to 0.5
                
                # Weight by distinctiveness and position
                weighted_freqs = freqs * distinct_factor * position_weight
                
                # Add term
                objective_terms.append(cp.sum(cp.multiply(alpha, weighted_freqs)))
            
            # Add entropy regularization to prevent distribution collapse
            regularization_strength = 0.05
            entropy_term = -regularization_strength * cp.sum(cp.entr(alpha))
            
            # Modified section - converted maximization to minimization by negating
            # Objective: minimize negative sum of weighted terms with regularization
            objective = cp.Minimize(-(sum(objective_terms) + entropy_term))

            
            # Solve the problem
            prob = cp.Problem(objective, constraints)
            try:
                # Try with default solver first
                prob.solve()
            except Exception as e:
                try:
                    # Try with SCS solver if default fails
                    logger.warning(f"Default solver failed: {e}, trying SCS solver")
                    prob.solve(solver=cp.SCS)
                except Exception as e2:
                    logger.error(f"SCS solver also failed: {e2}")
                    raise
            
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
        
        # Fallback to heuristic method
        return self._infer_distribution_heuristic(decade_patterns)
    
    def _extract_normalized_frequencies(self, decade_patterns, decades):
        """
        Extract normalized frequencies and calculate distinctiveness scores.
        
        Args:
            decade_patterns: Patterns detected for each decade
            decades: List of decades to analyze
            
        Returns:
            Tuple of (normalized frequencies dict, distinctiveness scores dict)
        """
        merge_frequencies = {}
        all_counts = defaultdict(list)
        
        # Collect all rule frequencies across decades
        for i, decade in enumerate(decades):
            if 'merge_rules' in decade_patterns[decade]:
                total_tokens = decade_patterns[decade]['total_tokens']
                if total_tokens > 0:
                    for rule, count in decade_patterns[decade]['merge_rules'].items():
                        if rule not in merge_frequencies:
                            merge_frequencies[rule] = np.zeros(len(decades))
                        normalized_freq = count / total_tokens
                        merge_frequencies[rule][i] = normalized_freq
                        all_counts[rule].append(normalized_freq)
        
        # Calculate distinctiveness scores (ratio of max frequency to mean of others)
        distinctive_scores = {}
        for rule, freqs in merge_frequencies.items():
            if len(all_counts[rule]) > 1:
                max_freq = np.max(freqs)
                max_decade_idx = np.argmax(freqs)
                other_freqs = [f for i, f in enumerate(freqs) if i != max_decade_idx and f > 0]
                mean_others = np.mean(other_freqs) if other_freqs else 0.0001
                distinctive_scores[rule] = max_freq / mean_others if mean_others > 0 else 1.0
            else:
                distinctive_scores[rule] = 1.0
        
        return merge_frequencies, distinctive_scores

    def _select_distinctive_rules(self, merge_frequencies, distinctive_scores, num_rules=1000):
        """
        Select rules based on a combination of frequency and distinctiveness.
        
        Args:
            merge_frequencies: Dict of rule frequencies by decade
            distinctive_scores: Dict of distinctiveness scores by rule
            num_rules: Number of rules to select
            
        Returns:
            List of selected rule identifiers
        """
        # Combine frequency and distinctiveness for scoring
        rule_scores = {}
        for rule, freqs in merge_frequencies.items():
            overall_freq = np.sum(freqs)
            distinctiveness = distinctive_scores[rule]
            # Balance between frequency and distinctiveness
            rule_scores[rule] = overall_freq * np.log1p(distinctiveness)
        
        # Select top rules by this combined score
        return sorted(rule_scores.keys(), key=lambda r: rule_scores[r], reverse=True)[:num_rules]
    
    def infer_distribution_ensemble(self, decade_patterns, methods=None, weights=None):
        """
        Use an ensemble of methods to produce a more robust distribution estimate.
        
        Args:
            decade_patterns: Patterns detected for each decade
            methods: List of (method_function, params_dict) tuples
            weights: List of weights for each method
            
        Returns:
            Dictionary mapping decades to their estimated proportion
        """
        if methods is None:
            methods = [
                (self.infer_temporal_distribution, {'num_merge_rules': 500, 'regularization_strength': 0.05}),
                (self.infer_temporal_distribution, {'num_merge_rules': 1000, 'regularization_strength': 0.1}),
                (self.infer_temporal_distribution, {'num_merge_rules': 2000, 'regularization_strength': 0.2}),
                (self._infer_distribution_bayesian, {}),
                (self._infer_distribution_heuristic, {})
            ]
        
        if weights is None:
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Weights for each method
        
        # Apply each method
        distributions = []
        for (method, params) in methods:
            try:
                distribution = method(decade_patterns, **params)
                distributions.append(distribution)
            except Exception as e:
                logger.warning(f"Method {method.__name__} failed: {e}")
                # Add uniform distribution as fallback
                decades = sorted(decade_patterns.keys())
                distributions.append({d: 1.0/len(decades) for d in decades})
        
        # Combine distributions using weighted average
        decades = sorted(list(set().union(*[d.keys() for d in distributions])))
        ensemble_distribution = {}
        
        for decade in decades:
            ensemble_distribution[decade] = sum(
                weights[i] * dist.get(decade, 0) 
                for i, dist in enumerate(distributions)
            )
        
        # Normalize to ensure sum to 1
        total = sum(ensemble_distribution.values())
        if total > 0:
            ensemble_distribution = {d: v/total for d, v in ensemble_distribution.items()}
        
        return ensemble_distribution

    def _infer_distribution_bayesian(self, decade_patterns):
        """
        Bayesian approach to infer distribution using rule probabilities.
        
        Args:
            decade_patterns: Patterns detected for each decade
            
        Returns:
            Dictionary mapping decades to their estimated proportion
        """
        decades = sorted(list(decade_patterns.keys()))
        if not decades:
            return {}
        
        # Calculate P(rule|decade) for each rule and decade
        rule_likelihoods = {}
        for decade in decades:
            if 'merge_rules' in decade_patterns[decade]:
                total_tokens = decade_patterns[decade]['total_tokens']
                if total_tokens > 0:
                    for rule, count in decade_patterns[decade]['merge_rules'].items():
                        if rule not in rule_likelihoods:
                            rule_likelihoods[rule] = {}
                        rule_likelihoods[rule][decade] = count / total_tokens
        
        # Apply Bayes' rule with uniform prior
        prior = {decade: 1.0/len(decades) for decade in decades}
        posterior = prior.copy()
        
        # Consider only the most distinctive rules for more reliable inference
        distinctive_rules = []
        for rule, likelihoods in rule_likelihoods.items():
            if len(likelihoods) > 1:
                values = list(likelihoods.values())
                max_val = max(values)
                max_decade = max(likelihoods.keys(), key=lambda d: likelihoods[d])
                other_values = [v for d, v in likelihoods.items() if d != max_decade]
                avg_others = sum(other_values) / len(other_values) if other_values else 0.0001
                distinctiveness = max_val / avg_others if avg_others > 0 else 1.0
                if distinctiveness > 1.5:
                    distinctive_rules.append((rule, distinctiveness, max_decade))
        
        # Sort by distinctiveness
        distinctive_rules.sort(key=lambda x: x[1], reverse=True)
        
        # Use top N distinctive rules
        for rule, _, _ in distinctive_rules[:100]:
            if rule in rule_likelihoods:
                likelihoods = rule_likelihoods[rule]
                
                # Update posterior using Bayes' rule
                for decade in decades:
                    # Add small epsilon to avoid zeros
                    likelihood = likelihoods.get(decade, 0.0001) 
                    posterior[decade] *= likelihood
        
        # Normalize posterior
        total = sum(posterior.values())
        if total > 0:
            return {decade: prob/total for decade, prob in posterior.items()}
        else:
            return {decade: 1.0/len(decades) for decade in decades}

    def _infer_distribution_heuristic(self, decade_patterns: Dict[str, Dict]) -> Dict[str, float]:
        """
        Improved heuristic method for temporal distribution inference.
        Uses a combination of distinctive patterns and normalized frequencies.
        
        Args:
            decade_patterns: Results from analyze_decade_patterns
        """
        # Extract decades
        decades = sorted(list(decade_patterns.keys()))
        
        # Calculate distinctive pattern scores for each decade
        distinctive_patterns = self.find_distinctive_patterns(decade_patterns, threshold=1.2)
        
        # Initial scores based on distinctive patterns
        decade_scores = {}
        for decade, patterns in distinctive_patterns.items():
            # Take top 10 patterns, weight by distinctiveness
            if patterns:
                # Calculate weighted score using both distinctiveness and frequency
                weighted_score = 0
                for pattern, score in patterns[:min(10, len(patterns))]:
                    # Get normalized frequency for this pattern in this decade
                    freq = 0
                    if 'merge_rules' in decade_patterns[decade]:
                        total_tokens = decade_patterns[decade]['total_tokens']
                        if total_tokens > 0 and pattern in decade_patterns[decade]['merge_rules']:
                            freq = decade_patterns[decade]['merge_rules'][pattern] / total_tokens
                    
                    # Weight score by both distinctiveness and frequency
                    weighted_score += score * freq * 100  # Scale up for numerical stability
                
                decade_scores[decade] = weighted_score
            else:
                decade_scores[decade] = 0.1  # Small non-zero default
        
        # Add token distribution similarity scores
        token_similarity_scores = self._calculate_token_similarity_scores(decade_patterns)
        
        # Combine scores (70% distinctive patterns, 30% token similarity)
        combined_scores = {}
        for decade in decades:
            distinctive_weight = 0.7
            similarity_weight = 0.3
            
            combined_scores[decade] = (
                distinctive_weight * decade_scores.get(decade, 0) +
                similarity_weight * token_similarity_scores.get(decade, 0)
            )
        
        # Handle case where all scores are zero
        if sum(combined_scores.values()) <= 0:
            return {decade: 1.0 / len(decades) for decade in decades}
        
        # Normalize scores to get proportions
        total_score = sum(combined_scores.values())
        proportions = {decade: score / total_score for decade, score in combined_scores.items()}
        
        return proportions

    def _calculate_token_similarity_scores(self, decade_patterns: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate scores based on overall token distribution similarity.
        This provides a complementary signal to distinctive patterns.
        """
        scores = {}
        decades = list(decade_patterns.keys())
        
        # First, get a vector representation for each decade's token distribution
        decade_vectors = {}
        all_tokens = set()
        
        for decade, patterns in decade_patterns.items():
            if 'tokens' in patterns:
                decade_vectors[decade] = patterns['tokens']
                all_tokens.update(patterns['tokens'].keys())
        
        # Create normalized frequency vectors for each decade
        normalized_vectors = {}
        for decade, token_counts in decade_vectors.items():
            total_count = sum(token_counts.values())
            if total_count > 0:
                normalized_vectors[decade] = {token: count/total_count for token, count in token_counts.items()}
        
        # Calculate similarity of each decade to the overall distribution
        all_decade_vector = {}
        for token in all_tokens:
            all_decade_vector[token] = sum(vectors.get(token, 0) for vectors in normalized_vectors.values()) / len(normalized_vectors)
        
        # Score each decade by similarity to the pattern found in the tokenizer
        for decade in decades:
            if decade in normalized_vectors:
                # Calculate cosine similarity
                vec1 = [normalized_vectors[decade].get(token, 0) for token in all_tokens]
                vec2 = [all_decade_vector.get(token, 0) for token in all_tokens]
                
                # Simple similarity calculation
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                magnitude1 = sum(a * a for a in vec1) ** 0.5
                magnitude2 = sum(b * b for b in vec2) ** 0.5
                
                if magnitude1 > 0 and magnitude2 > 0:
                    similarity = dot_product / (magnitude1 * magnitude2)
                    scores[decade] = similarity
                else:
                    scores[decade] = 0.1
            else:
                scores[decade] = 0.1
        
        return scores
    
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
        
        # Check if we have enough merge rules for meaningful analysis
        if len(self.merge_rules) < 100:
            logger.warning(f"Only {len(self.merge_rules)} merge rules available - analysis may be less accurate")
        
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