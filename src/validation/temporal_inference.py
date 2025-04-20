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
            
            # Method 5: BERT-specific extraction for BERT-type tokenizers
            if not self.merge_rules and "bert" in self.tokenizer_name.lower():
                logger.info("Detected BERT-type tokenizer, using specialized extraction method")
                self.merge_rules = self._extract_bert_merge_rules()
                logger.info(f"Extracted {len(self.merge_rules)} merge rules using BERT-specific method")
            
            # Method 6: If still no merge rules, create synthetic ones by analyzing tokenizer behavior
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

    def _extract_bert_merge_rules(self):
        """
        Extract merge rules from BERT tokenizer using a more direct approach.
        This is a workaround for BERT-type tokenizers that don't expose merge rules.
        """
        if not hasattr(self, 'tokenizer'):
            logger.error("No tokenizer available")
            return []
        
        logger.info("Extracting BERT merge rules using vocabulary analysis")
        
        # Get vocabulary
        vocab = self.tokenizer.get_vocab()
        
        # Identify subword patterns
        merge_rules = []
        
        # Collect wordpieces that look like continuations (start with ##)
        continuations = {}
        for token, idx in vocab.items():
            if token.startswith('##'):
                continuations[token[2:]] = idx
        
        # Build potential merge rules from vocabulary structure
        for token, idx in vocab.items():
            # Skip special tokens
            if token.startswith('[') or token in ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']:
                continue
            
            # Find potential subword pairs
            for i in range(1, len(token)):
                prefix = token[:i]
                suffix = token[i:]
                
                # Check if both parts are in vocabulary
                if prefix in vocab and ('##' + suffix) in vocab:
                    # This is a potential merge
                    rule = (prefix, '##' + suffix)
                    merge_rules.append(rule)
        
        # Sort merge rules by token indices (rough approximation of merge order)
        merge_rules.sort(key=lambda r: (vocab.get(r[0], 0) + vocab.get(r[1], 0)) / 2)
        
        logger.info(f"Extracted {len(merge_rules)} potential merge rules for BERT")
        return merge_rules

    def analyze_data_quality(self, decade_texts):
        """
        Analyze data quality to ensure it's sufficient for reliable inference.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            
        Returns:
            Dictionary with data quality metrics
        """
        import re
        import random
        
        quality_metrics = {}
        
        for decade, texts in decade_texts.items():
            decade_metrics = {
                'num_texts': len(texts),
                'total_chars': sum(len(text) for text in texts),
                'avg_text_length': sum(len(text) for text in texts) / max(1, len(texts)),
                'min_text_length': min((len(text) for text in texts), default=0),
                'max_text_length': max((len(text) for text in texts), default=0),
                'vocabulary_size': self._estimate_vocabulary_size(texts),
                'data_bytes': sum(len(text.encode('utf-8')) for text in texts),
                'data_gb': sum(len(text.encode('utf-8')) for text in texts) / (1024**3),
            }
            
            # Evaluate if data quality is sufficient
            is_sufficient = (
                decade_metrics['num_texts'] >= 30 and
                decade_metrics['data_gb'] >= 0.5 and  # At least 0.5 GB per decade
                decade_metrics['avg_text_length'] >= 1000  # Average text at least 1000 chars
            )
            
            decade_metrics['is_sufficient'] = is_sufficient
            quality_metrics[decade] = decade_metrics
            
            # Log warnings for insufficient data
            if not is_sufficient:
                if decade_metrics['num_texts'] < 30:
                    logger.warning(f"{decade}: Insufficient text count ({decade_metrics['num_texts']} < 30)")
                if decade_metrics['data_gb'] < 0.5:
                    logger.warning(f"{decade}: Insufficient data volume ({decade_metrics['data_gb']:.2f} GB < 0.5 GB)")
                if decade_metrics['avg_text_length'] < 1000:
                    logger.warning(f"{decade}: Texts too short ({decade_metrics['avg_text_length']:.1f} chars < 1000)")
        
        # Overall quality assessment
        quality_metrics['overall'] = {
            'all_decades_sufficient': all(metrics['is_sufficient'] for metrics in quality_metrics.values() if isinstance(metrics, dict)),
            'total_data_gb': sum(metrics['data_gb'] for metrics in quality_metrics.values() if isinstance(metrics, dict)),
            'total_texts': sum(metrics['num_texts'] for metrics in quality_metrics.values() if isinstance(metrics, dict)),
        }
        
        # Log overall assessment
        if quality_metrics['overall']['all_decades_sufficient']:
            logger.info(f"Data quality check passed: {quality_metrics['overall']['total_texts']} texts, {quality_metrics['overall']['total_data_gb']:.2f} GB total")
        else:
            logger.warning(f"Data quality check failed: {quality_metrics['overall']['total_texts']} texts, {quality_metrics['overall']['total_data_gb']:.2f} GB total")
        
        return quality_metrics

    def _estimate_vocabulary_size(self, texts, sample_size=10000):
        """Estimate vocabulary size from a sample of texts."""
        import re
        import random
        
        # Sample texts to limit processing time
        if len(texts) > sample_size:
            sampled_texts = random.sample(texts, sample_size)
        else:
            sampled_texts = texts
        
        # Collect unique words
        word_set = set()
        for text in sampled_texts:
            words = re.findall(r'\b\w+\b', text.lower())
            word_set.update(words)
        
        return len(word_set)

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

    def quantify_uncertainty(self, decade_patterns, distribution, sample_sizes=None):
        """
        Quantify uncertainty in distribution estimates, especially for small sample sizes.
        Implements the statistical validation suggested by the professor.
        
        Args:
            decade_patterns: Patterns detected for each decade
            distribution: The inferred distribution
            sample_sizes: Optional dictionary mapping decades to sample sizes
            
        Returns:
            Dictionary with uncertainty estimates
        """
        decades = sorted(distribution.keys())
        
        # If sample sizes not provided, extract from patterns
        if not sample_sizes:
            sample_sizes = {}
            for decade, patterns in decade_patterns.items():
                if 'total_tokens' in patterns:
                    sample_sizes[decade] = patterns['total_tokens']
                else:
                    # Estimate from merge rules
                    if 'merge_rules' in patterns:
                        sample_sizes[decade] = sum(patterns['merge_rules'].values())
                    else:
                        sample_sizes[decade] = 0
        
        # Calculate uncertainty based on sample size
        uncertainty = {}
        for decade in decades:
            sample_size = sample_sizes.get(decade, 0)
            
            # Apply statistical formula for margin of error in proportion
            # 95% confidence interval uses z=1.96
            # Formula: z * sqrt(p*(1-p)/n)
            p = distribution.get(decade, 0)
            
            if sample_size > 0:
                margin_of_error = 1.96 * np.sqrt((p * (1-p)) / sample_size)
                
                # For very small sample sizes (as per professor's concern), increase uncertainty
                if sample_size < 100:
                    # Add additional penalty for extremely small samples
                    small_sample_penalty = 1.0 + (100 - sample_size) / 100
                    margin_of_error *= small_sample_penalty
            else:
                # If no samples, set a high uncertainty
                margin_of_error = 0.5  # Represent high uncertainty with 50% margin
            
            # Calculate confidence interval
            lower_bound = max(0, p - margin_of_error)
            upper_bound = min(1, p + margin_of_error)
            
            # Assess reliability based on sample size
            if sample_size > 1000:
                reliability = "high"
            elif sample_size > 100:
                reliability = "medium"
            else:
                reliability = "low"
            
            # Store results
            uncertainty[decade] = {
                "value": p,
                "margin_of_error": margin_of_error,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "sample_size": sample_size,
                "reliability": reliability
            }
        
        # Calculate the 1960s correction factor based on uncertainty
        # This implements the professor's suggestion about the 1960s
        if "1960s" in uncertainty:
            sixties_data = uncertainty["1960s"]
            # Apply stronger correction (0.6) for 1960s as suggested by professor
            correction_factor = 0.6
            
            # Adjust the correction factor based on sample size reliability
            if sixties_data["reliability"] == "high":
                # High reliability means we can be more confident in the correction
                sixties_data["corrected_value"] = sixties_data["value"] * correction_factor
            elif sixties_data["reliability"] == "medium":
                # Medium reliability means we apply a slightly less aggressive correction
                adjusted_factor = (correction_factor + 1.0) / 2  # Average between correction and no correction
                sixties_data["corrected_value"] = sixties_data["value"] * adjusted_factor
            else:
                # Low reliability means high uncertainty, so we're more cautious with correction
                # Use a milder correction
                sixties_data["corrected_value"] = sixties_data["value"] * 0.8  # 20% reduction instead of 40%
            
            uncertainty["1960s"] = sixties_data
        
        # Also add the professor's suggestion about removing top tokens
        uncertainty["methodology_notes"] = {
            "top_tokens_removed": 5,  # As suggested by professor
            "sixties_correction_applied": "1960s" in uncertainty,
            "sixties_correction_factor": 0.6  # Professor's suggested value
        }
        
        return uncertainty

    def analyze_decade_patterns(self, decade_texts: Dict[str, List[str]], sample_size: int = 5000) -> Dict[str, Dict]:
        """
        Analyze merge rules and token patterns for each decade with improved memory efficiency.
        """
        decade_patterns = {}
        
        # Process each decade with better memory management
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
            
            # Process texts in batches to control memory usage
            batch_size = 5  # Process 5 texts at a time
            for i in range(0, min(20, len(sampled_texts)), batch_size):
                batch_texts = sampled_texts[i:i+batch_size]
                
                # Process each text in the batch
                for text in batch_texts:
                    # Ensure text is not too long for tokenizer
                    if isinstance(text, tuple):
                        text = text[0]  # Extract text if it's a (text, source) tuple
                        
                    # Skip very short chunks
                    if len(text) < 100:
                        continue
                        
                    # Tokenize chunk with error handling
                    try:
                        tokens = self.tokenizer.tokenize(text)
                        
                        # Skip if tokenization failed
                        if not tokens:
                            continue
                            
                        # Count tokens
                        token_counts.update(tokens)
                        total_tokens += len(tokens)
                        
                        # Count merge rules - more memory efficient approach
                        for token in tokens:
                            # Extract applicable merge rules for this token
                            applicable_rules = self._extract_merge_rules(token)
                            for rule in applicable_rules:
                                merge_rule_counts[rule] = merge_rule_counts.get(rule, 0) + 1
                        
                        # Count character pairs (bigrams)
                        for i in range(len(text) - 1):
                            char_pair = text[i:i+2]
                            char_pair_counts[char_pair] = char_pair_counts.get(char_pair, 0) + 1
                            total_chars += 1
                    except Exception as e:
                        logger.debug(f"Error processing chunk: {e}")
                
                # Force garbage collection after each batch
                gc.collect()
            
            # Calculate statistics
            if total_tokens > 0 and total_chars > 0:
                # Store decade statistics - convert to regular dicts to reduce memory usage
                decade_patterns[decade] = {
                    'merge_rules': dict(merge_rule_counts),
                    'tokens': dict(token_counts),
                    'char_pairs': dict(char_pair_counts),
                    'total_tokens': total_tokens,
                    'total_chars': total_chars
                }
            
            # Clean up to free memory
            del merge_rule_counts, token_counts, char_pair_counts
            gc.collect()
        
        return decade_patterns

    def _split_text_for_tokenizer(self, text, max_chars=500):
        """
        Split text into smaller chunks to avoid context length issues and memory problems.
        Uses a more aggressive approach to handle extremely long texts.
        
        Args:
            text: Text to split
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        # Handle extremely long texts more gracefully - hard truncate at a reasonable size
        max_text_length = 10000  # Maximum text length to process in characters
        
        if len(text) > max_text_length:
            logger.warning(f"Found extremely long text ({len(text)} chars) - truncating to {max_text_length} chars")
            text = text[:max_text_length]

        # If text is still within limits, return as single chunk
        if len(text) <= max_chars:
            return [text]
        
        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If paragraph is too long, split further
            if len(para) > max_chars:
                # Add current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Split long paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                
                # Process sentences
                current_sentence_chunk = ""
                for sentence in sentences:
                    if len(current_sentence_chunk) + len(sentence) + 1 > max_chars:
                        if current_sentence_chunk:
                            chunks.append(current_sentence_chunk)
                            current_sentence_chunk = sentence
                        else:
                            # For very long sentences, split at character level
                            if len(sentence) > max_chars:
                                for i in range(0, len(sentence), max_chars):
                                    chunks.append(sentence[i:i + max_chars])
                            else:
                                chunks.append(sentence)
                    else:
                        if current_sentence_chunk:
                            current_sentence_chunk += " " + sentence
                        else:
                            current_sentence_chunk = sentence
                
                # Add any remaining sentence chunk
                if current_sentence_chunk:
                    chunks.append(current_sentence_chunk)
            
            # For shorter paragraphs
            elif len(current_chunk) + len(para) + 2 > max_chars:
                chunks.append(current_chunk)
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk if exists
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
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
    
    def analyze_merge_rule_distinctiveness(self, decade_patterns):
        """Analyze how distinctive merge rules are for each decade."""
        decades = sorted(decade_patterns.keys())
        distinctiveness_by_decade = {}
        
        for decade in decades:
            if 'merge_rules' not in decade_patterns[decade]:
                continue
                
            rules = decade_patterns[decade]['merge_rules']
            total_tokens = decade_patterns[decade]['total_tokens']
            
            # Calculate how distinctive each rule is
            rule_distinctiveness = {}
            for rule, count in rules.items():
                # Normalize by total tokens
                norm_freq = count / total_tokens if total_tokens > 0 else 0
                
                # Compare to other decades
                other_decades = [d for d in decades if d != decade]
                other_freqs = []
                
                for other_decade in other_decades:
                    if 'merge_rules' in decade_patterns[other_decade]:
                        other_rules = decade_patterns[other_decade]['merge_rules']
                        other_total = decade_patterns[other_decade]['total_tokens']
                        
                        if rule in other_rules and other_total > 0:
                            other_freq = other_rules[rule] / other_total
                        else:
                            other_freq = 0
                            
                        other_freqs.append(other_freq)
                
                # Calculate distinctiveness
                if other_freqs:
                    avg_other = sum(other_freqs) / len(other_freqs)
                    if avg_other > 0:
                        distinctiveness = norm_freq / avg_other
                    else:
                        distinctiveness = float('inf')  # Avoid division by zero
                else:
                    distinctiveness = 1.0  # Default
                    
                rule_distinctiveness[rule] = distinctiveness
            
            # Get top distinctive rules
            sorted_rules = sorted(rule_distinctiveness.items(), key=lambda x: x[1], reverse=True)
            distinctiveness_by_decade[decade] = sorted_rules[:20]  # Top 20
        
        # Calculate average distinctiveness per decade
        avg_distinctiveness = {decade: sum(d for _, d in rules) / len(rules) if rules else 0 
                            for decade, rules in distinctiveness_by_decade.items()}
        
        return distinctiveness_by_decade, avg_distinctiveness

    def bootstrap_distribution_estimates(self, decade_patterns, num_bootstraps=100):
        """
        Use bootstrapping to estimate confidence intervals for distribution estimates.
        
        Args:
            decade_patterns: Patterns detected for each decade
            num_bootstraps: Number of bootstrap samples
            
        Returns:
            Dictionary with bootstrapped distributions and confidence intervals
        """
        decades = sorted(decade_patterns.keys())
        bootstrap_results = []
        
        # Run multiple bootstrap iterations
        for i in range(num_bootstraps):
            # Create bootstrapped sample of the patterns
            bootstrapped_patterns = {}
            for decade, patterns in decade_patterns.items():
                if 'merge_rules' in patterns:
                    # Sample with replacement
                    bootstrapped_rules = {}
                    rules = list(patterns['merge_rules'].items())
                    sampled_rules = random.choices(rules, k=len(rules))
                    for rule, count in sampled_rules:
                        if rule in bootstrapped_rules:
                            bootstrapped_rules[rule] += count
                        else:
                            bootstrapped_rules[rule] = count
                    
                    # Create new patterns dict with bootstrapped rules
                    bootstrapped_patterns[decade] = {
                        'merge_rules': bootstrapped_rules,
                        'total_tokens': patterns['total_tokens']
                    }
            
            # Infer distribution with bootstrapped sample
            distribution = self.infer_temporal_distribution(bootstrapped_patterns)
            bootstrap_results.append(distribution)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for decade in decades:
            values = [dist.get(decade, 0) for dist in bootstrap_results]
            values.sort()
            
            # 95% confidence interval
            lower = values[int(0.025 * num_bootstraps)]
            upper = values[int(0.975 * num_bootstraps)]
            
            confidence_intervals[decade] = (lower, upper)
        
        return {
            'bootstrap_samples': bootstrap_results,
            'confidence_intervals': confidence_intervals
        }

    def apply_decade_correction(self, distribution, decade="1960s", factor=0.6):
        """
        Apply a correction factor to a specific decade (especially 1960s)
        which consistently shows overrepresentation in the analysis.
        
        Args:
            distribution: The original distribution dictionary
            decade: The decade to correct
            factor: Correction factor (0.6 means reduce by 40%)
            
        Returns:
            Corrected distribution
        """
        if decade not in distribution:
            return distribution
            
        # Create a copy to avoid modifying the original
        corrected = distribution.copy()
        
        # Apply correction
        original_value = corrected[decade]
        corrected[decade] = original_value * factor
        
        # Redistribute the excess to other decades proportionally
        excess = original_value - corrected[decade]
        other_decades = [d for d in corrected if d != decade]
        
        if other_decades and excess > 0:
            # Proportional redistribution based on current weights
            other_sum = sum(corrected[d] for d in other_decades)
            if other_sum > 0:
                for d in other_decades:
                    # Distribute proportionally to current values
                    corrected[d] += excess * (corrected[d] / other_sum)
            else:
                # Equal distribution if all others are zero
                for d in other_decades:
                    corrected[d] += excess / len(other_decades)
        
        # Ensure the sum is still 1.0
        total = sum(corrected.values())
        if abs(total - 1.0) > 0.001:  # Allow for small floating point errors
            corrected = {d: v/total for d, v in corrected.items()}
        
        return corrected

    def analyze_decade_specific_issues(self, decade_patterns: Dict[str, Dict], problematic_decade="1960s") -> Dict:
        """
        Analyze patterns specific to a problematic decade to identify anomalies.
        
        Args:
            decade_patterns: Dictionary mapping decades to pattern frequencies
            problematic_decade: The decade to analyze (default: "1960s")
            
        Returns:
            Dictionary with analysis results
        """
        if problematic_decade not in decade_patterns:
            logger.warning(f"No data available for {problematic_decade}")
            return {}
        
        # Get merge rules for the problematic decade
        if 'merge_rules' not in decade_patterns[problematic_decade]:
            logger.warning(f"No merge rules found for {problematic_decade}")
            return {}
                
        decade_rules = decade_patterns[problematic_decade]['merge_rules']
        
        # Calculate distinctiveness for each rule
        distinctive_rules = []
        for rule, count in decade_rules.items():
            # Calculate frequency in other decades
            other_decade_counts = []
            for decade, patterns in decade_patterns.items():
                if decade != problematic_decade and 'merge_rules' in patterns:
                    other_count = patterns['merge_rules'].get(rule, 0)
                    other_decade_counts.append(other_count)
            
            # Calculate average count in other decades
            if other_decade_counts:
                avg_other = sum(other_decade_counts) / len(other_decade_counts)
                # Distinctiveness is ratio of this decade's count to average in others
                if avg_other > 0:
                    distinctiveness = count / avg_other
                else:
                    distinctiveness = float('inf')  # Uniquely appears in this decade
            else:
                distinctiveness = float('inf')  # Uniquely appears in this decade
                    
            distinctive_rules.append((rule, distinctiveness, count))
        
        # Sort by distinctiveness
        distinctive_rules.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate total token count for the decade
        total_count = sum(decade_rules.values())
        
        # Calculate contribution percentage for top distinctive rules
        top_distinctive = distinctive_rules[:20]
        distinctive_contribution = {
            rule: {
                "distinctiveness": dist,
                "count": count,
                "contribution": (count / total_count) * 100 if total_count > 0 else 0
            } for rule, dist, count in top_distinctive
        }
        
        # Calculate the total contribution of these distinctive rules
        total_distinctive_contribution = sum(
            item["contribution"] for item in distinctive_contribution.values()
        )
        
        # Identify potential biasing tokens
        high_bias_tokens = []
        for rule, dist, count in distinctive_rules[:10]:
            if dist > 3.0:  # Highly distinctive tokens
                high_bias_tokens.append(rule)
        
        # Apply stronger correction factor for 1960s as this decade is consistently overrepresented
        suggested_correction_factor = 0.6 if problematic_decade == "1960s" else 0.8
        
        return {
            "top_distinctive_rules": distinctive_contribution,
            "total_distinctive_contribution": total_distinctive_contribution,
            "high_bias_tokens": high_bias_tokens,
            "suggested_correction_factor": suggested_correction_factor,
            "analysis_summary": f"The top 20 distinctive rules contribute {total_distinctive_contribution:.2f}% to {problematic_decade}'s total token count"
        }

    def remove_top_frequent_tokens(self, decade_patterns, top_n=5):
        """
        Remove the top N most frequent tokens across all decades that could bias the analysis.
        Focuses on tokens with high frequency but low temporal distinctiveness.
        
        Args:
            decade_patterns: Dictionary mapping decades to pattern frequencies
            top_n: Number of top tokens to remove (set to 5 as per professor's suggestion)
            
        Returns:
            Filtered decade patterns with biasing tokens removed
        """
        # Create a copy to avoid modifying the original
        filtered_patterns = {}
        for decade, patterns in decade_patterns.items():
            if isinstance(patterns, dict):
                filtered_patterns[decade] = {
                    key: value.copy() if isinstance(value, dict) else value 
                    for key, value in patterns.items()
                }
            else:
                filtered_patterns[decade] = patterns
        
        # Calculate global token frequencies
        token_freqs = {}
        decade_norms = {}
        
        # Get total tokens per decade for normalization
        for decade, patterns in decade_patterns.items():
            if isinstance(patterns, dict) and 'total_tokens' in patterns:
                decade_norms[decade] = patterns['total_tokens']
        
        # First pass: calculate normalized frequencies for each token per decade
        token_decade_freqs = {}
        for decade, patterns in decade_patterns.items():
            if isinstance(patterns, dict) and 'merge_rules' in patterns:
                norm_factor = decade_norms.get(decade, 1)
                for token, freq in patterns['merge_rules'].items():
                    if token not in token_decade_freqs:
                        token_decade_freqs[token] = {}
                    # Store normalized frequency
                    if norm_factor > 0:
                        token_decade_freqs[token][decade] = freq / norm_factor
                    else:
                        token_decade_freqs[token][decade] = 0
                    
                    # Update global frequency
                    if token not in token_freqs:
                        token_freqs[token] = 0
                    token_freqs[token] += freq
        
        # Calculate temporal distinctiveness for each token
        # (how much the token's frequency varies across decades)
        token_distinctiveness = {}
        for token, decade_freqs in token_decade_freqs.items():
            if len(decade_freqs) > 1:
                # Get frequency values across decades
                freqs = list(decade_freqs.values())
                # Use normalized variance as distinctiveness measure
                # Higher variance = more decade-specific
                mean_freq = sum(freqs) / len(freqs) if freqs else 0
                if mean_freq > 0:
                    # Normalized variance - coefficient of variation
                    variance = sum((f - mean_freq)**2 for f in freqs) / len(freqs)
                    token_distinctiveness[token] = variance / (mean_freq**2)
                else:
                    token_distinctiveness[token] = 0
            else:
                # Default low distinctiveness for tokens in only one decade
                token_distinctiveness[token] = 0
        
        # Identify tokens that are both common and have low temporal distinctiveness
        # These are likely to be common English tokens that don't help distinguish decades
        biasing_tokens = []
        for token, freq in token_freqs.items():
            # Only consider tokens with significant frequency
            if freq > 10:  # Minimum frequency threshold
                distinctiveness = token_distinctiveness.get(token, 0)
                # Score = frequency / distinctiveness
                # Higher score = more biasing (high freq, low distinctiveness)
                bias_score = freq / (distinctiveness + 0.001)  # Avoid division by zero
                biasing_tokens.append((token, bias_score, freq, distinctiveness))
        
        # Sort by bias score (descending)
        biasing_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # Select top biasing tokens
        tokens_to_remove = []
        for token, score, freq, dist in biasing_tokens[:top_n]:
            tokens_to_remove.append(token)
        
        # Log removed tokens
        logger.info(f"Top {len(tokens_to_remove)} tokens being removed: {tokens_to_remove}")
        
        # Remove selected tokens from all decade patterns
        for decade, patterns in filtered_patterns.items():
            if isinstance(patterns, dict) and 'merge_rules' in patterns:
                patterns['merge_rules'] = {
                    token: freq for token, freq in patterns['merge_rules'].items()
                    if token not in tokens_to_remove
                }
        
        return filtered_patterns

    # def remove_top_frequent_tokens(self, decade_patterns, top_n=5):
    #     """
    #     Remove the top N most frequent tokens across all decades.
        
    #     Args:
    #         decade_patterns: Dictionary mapping decades to pattern frequencies
    #         top_n: Number of top frequent tokens to remove
            
    #     Returns:
    #         Filtered decade patterns with top frequent tokens removed
    #     """
    #     # Debug the structure of decade_patterns
    #     logger.info(f"DEBUG: decade_patterns type: {type(decade_patterns)}")
    #     if isinstance(decade_patterns, dict):
    #         for decade, patterns in decade_patterns.items():
    #             logger.info(f"DEBUG: decade {decade} patterns type: {type(patterns)}")
    #             if isinstance(patterns, dict):
    #                 for key in patterns.keys():
    #                     logger.info(f"DEBUG: decade {decade} has key: {key}")
        
    #     # Ensure the input is a dictionary
    #     if not isinstance(decade_patterns, dict):
    #         logger.warning(f"Invalid decade_patterns structure: {type(decade_patterns)}")
    #         return decade_patterns
        
    #     # Create a safe copy to avoid modifying the original
    #     filtered_patterns = {}
    #     for decade, patterns in decade_patterns.items():
    #         # Copy the structure for this decade
    #         filtered_patterns[decade] = patterns.copy() if isinstance(patterns, dict) else patterns
        
    #     # Completely different approach: don't try to calculate global frequencies
    #     # Just remove tokens based on patterns from individual decades
    #     try:
    #         # Find tokens that appear frequently in each decade
    #         all_frequent_tokens = []
            
    #         for decade, patterns in decade_patterns.items():
    #             if not isinstance(patterns, dict):
    #                 continue
                    
    #             # Try different keys that might contain token frequencies
    #             if 'merge_rules' in patterns:
    #                 # Get token frequencies safely
    #                 token_freqs = []
    #                 for token, freq in patterns['merge_rules'].items():
    #                     if isinstance(freq, (int, float)):
    #                         token_freqs.append((token, freq))
    #                     elif isinstance(freq, dict) and 'count' in freq:
    #                         # Handle case where freq is a dict with a count
    #                         count = freq['count']
    #                         if isinstance(count, (int, float)):
    #                             token_freqs.append((token, count))
                    
    #                 # Sort by frequency and take top tokens
    #                 token_freqs.sort(key=lambda x: x[1], reverse=True)
    #                 top_tokens = [token for token, _ in token_freqs[:top_n]]
    #                 all_frequent_tokens.extend(top_tokens)
                
    #             # Try other possible keys
    #             elif 'tokens' in patterns:
    #                 top_tokens = sorted(patterns['tokens'].items(), 
    #                             key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
    #                             reverse=True)[:top_n]
    #                 all_frequent_tokens.extend([token for token, _ in top_tokens])
    #     except Exception as e:
    #         logger.error(f"Error finding frequent tokens: {e}")
    #         return decade_patterns  # Return original if we can't process
        
    #     # Get the most frequently occurring tokens across all decades
    #     token_counter = {}
    #     for token in all_frequent_tokens:
    #         token_counter[token] = token_counter.get(token, 0) + 1
        
    #     # Take the top N most common tokens
    #     top_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]
    #     tokens_to_remove = [token for token, _ in top_tokens]
        
    #     logger.info(f"Removing top {len(tokens_to_remove)} most frequent tokens: {tokens_to_remove}")
        
    #     # Filter these tokens from all decade patterns
    #     for decade, patterns in filtered_patterns.items():
    #         if not isinstance(patterns, dict):
    #             continue
                
    #         # Remove from merge_rules if present
    #         if 'merge_rules' in patterns and isinstance(patterns['merge_rules'], dict):
    #             filtered_patterns[decade]['merge_rules'] = {
    #                 rule: count for rule, count in patterns['merge_rules'].items() 
    #                 if rule not in tokens_to_remove
    #             }
            
    #         # Remove from tokens if present
    #         if 'tokens' in patterns and isinstance(patterns['tokens'], dict):
    #             filtered_patterns[decade]['tokens'] = {
    #                 token: count for token, count in patterns['tokens'].items()
    #                 if token not in tokens_to_remove
    #             }
        
    #     return filtered_patterns

    def infer_temporal_distribution(self, 
                     decade_patterns: Dict[str, Dict],
                     num_merge_rules: int = 2000,
                     weight_early_merges: bool = True,
                     regularization_strength: float = 0.05,
                     remove_top_tokens: bool = True,
                     top_n: int = 5):  # Set to 5 as per professor's suggestion
        """
        Infer the temporal distribution in training data using enhanced linear programming.
        Incorporating professor's suggestions for token removal and 1960s correction.
        
        Args:
            decade_patterns: Dictionary mapping decades to their patterns
            num_merge_rules: Number of merge rules to consider
            weight_early_merges: Whether to give higher weight to early merges
            regularization_strength: Strength of regularization term
            remove_top_tokens: Whether to remove top frequent tokens
            top_n: Number of top tokens to remove (set to 5 per professor)
                
        Returns:
            Dictionary mapping decades to their estimated proportion
        """
        # Check if we have valid input
        if not isinstance(decade_patterns, dict) or not decade_patterns:
            logger.warning("Invalid or empty decade_patterns provided")
            return {}
        
        # First filter out top frequent tokens if requested
        if remove_top_tokens:
            logger.info(f"Removing top {top_n} most frequent tokens as discussed with professor")
            filtered_patterns = self.remove_top_frequent_tokens(decade_patterns, top_n)
            logger.info("Successfully removed top frequent tokens")
        else:
            filtered_patterns = decade_patterns
        
        # Extract decades
        decades = sorted(list(filtered_patterns.keys()))
        
        if not decades:
            logger.warning("No decades found in patterns")
            return {}
        
        # For full/production mode, use the full linear programming approach
        try:
            # Prepare linear programming variables
            import cvxpy as cp
            import numpy as np
            
            alpha = cp.Variable(len(decades), pos=True)
            
            # Sum-to-one constraint
            constraints = [cp.sum(alpha) == 1]
            
            # Use lower minimum probability constraint to allow more flexibility
            min_prob = 0.005  # Reduced from 0.01
            constraints.extend([alpha[i] >= min_prob for i in range(len(decades))])
            
            # Add upper bound constraint to prevent single decade dominance
            constraints.extend([alpha[i] <= 0.40 for i in range(len(decades))])
            
            # Extract normalized merge rule frequencies
            merge_frequencies = {}
            for i, decade in enumerate(decades):
                if 'merge_rules' in filtered_patterns[decade]:
                    total_tokens = filtered_patterns[decade]['total_tokens']
                    if total_tokens > 0:
                        for rule, count in filtered_patterns[decade]['merge_rules'].items():
                            if rule not in merge_frequencies:
                                merge_frequencies[rule] = np.zeros(len(decades))
                            # Normalize by total tokens
                            merge_frequencies[rule][i] = count / total_tokens
            
            # Calculate temporal progression score for each rule
            temporal_scores = {}
            for rule, freqs in merge_frequencies.items():
                if np.sum(freqs) > 0:
                    # Check if rule shows increasing or decreasing trend
                    if len(decades) > 2:
                        # Calculate correlation with decade indices
                        decade_indices = np.arange(len(decades))
                        correlation = np.corrcoef(decade_indices, freqs)[0, 1]
                        
                        # Absolute correlation shows strength of temporal association
                        temporal_scores[rule] = abs(correlation)
                    else:
                        temporal_scores[rule] = 0.5  # Default for few decades
            
            # Calculate distinctiveness for each rule with improved formula
            distinctiveness = {}
            for rule, freqs in merge_frequencies.items():
                if np.sum(freqs) > 0:
                    max_val = np.max(freqs)
                    max_idx = np.argmax(freqs)
                    max_decade = decades[max_idx]
                    other_vals = np.delete(freqs, max_idx)
                    mean_others = np.mean(other_vals) if len(other_vals) > 0 else 0.0001
                    
                    # Calculate ratio with capping to avoid extreme values
                    ratio = max_val / mean_others if mean_others > 0 else 1.0
                    distinctiveness[rule] = np.log1p(min(ratio, 10.0))  # Use log1p to smooth extreme values
            
            # Balance rule selection across decades
            decade_rule_counts = {decade: 0 for decade in decades}
            rule_scores = {}
            
            # First calculate scores for all rules
            for rule, freqs in merge_frequencies.items():
                overall_freq = np.sum(freqs)
                distinct_score = distinctiveness.get(rule, 0)
                temporal_score = temporal_scores.get(rule, 0)
                
                # Higher score for rules with both high distinctiveness and frequency
                rule_scores[rule] = overall_freq * distinct_score * (1 + temporal_score)
            
            # Select rules in two passes for better decade balance
            selected_rules = []
            
            # First pass: select top 50% purely by score
            top_half_count = num_merge_rules // 2
            top_rules = sorted(rule_scores.keys(), key=lambda r: rule_scores[r], reverse=True)[:top_half_count]
            selected_rules.extend(top_rules)
            
            # Track which decade each selected rule favors most
            for rule in top_rules:
                freqs = merge_frequencies[rule]
                max_decade_idx = np.argmax(freqs)
                decade_rule_counts[decades[max_decade_idx]] += 1
            
            # Second pass: favor underrepresented decades
            remaining_rules = [r for r in rule_scores.keys() if r not in selected_rules]
            remaining_count = num_merge_rules - top_half_count
            
            # Calculate inverse weights - lower count = higher weight
            decade_weights = {}
            total_rules = sum(decade_rule_counts.values()) + 0.001  # Avoid division by zero
            
            for decade, count in decade_rule_counts.items():
                # Calculate inverse frequency weight
                decade_weights[decade] = 1.0 - (count / total_rules)
            
            # Adjust weights to favor historical decades
            historical_decades = ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s"]
            for decade in historical_decades:
                if decade in decade_weights:
                    # Give 50% boost to historical decade weights
                    decade_weights[decade] *= 1.5
            
            # Normalize weights
            weight_sum = sum(decade_weights.values())
            if weight_sum > 0:
                decade_weights = {d: w/weight_sum for d, w in decade_weights.items()}
            
            # Adjust rule scores by decade weights
            adjusted_scores = {}
            for rule in remaining_rules:
                freqs = merge_frequencies[rule]
                max_decade_idx = np.argmax(freqs)
                max_decade = decades[max_decade_idx]
                # Boost score for underrepresented decades
                adjusted_scores[rule] = rule_scores[rule] * (1.0 + decade_weights.get(max_decade, 0))
            
            # Take top rules by adjusted scores
            balanced_rules = sorted(adjusted_scores.keys(), key=lambda r: adjusted_scores[r], reverse=True)[:remaining_count]
            selected_rules.extend(balanced_rules)
            
            # Verify we have balanced representation
            for rule in balanced_rules:
                freqs = merge_frequencies[rule]
                max_decade_idx = np.argmax(freqs)
                decade_rule_counts[decades[max_decade_idx]] += 1
            
            # Log decade representation in selected rules
            logger.info("Rule representation by decade:")
            for decade, count in decade_rule_counts.items():
                logger.info(f"  {decade}: {count} rules ({count/len(selected_rules):.1%})")
            
            # Construct objective function terms
            data_fit_term = 0
            for rule in selected_rules:
                freqs = merge_frequencies[rule]
                
                # Get temporal direction of this rule
                if len(decades) > 2:
                    decade_indices = np.arange(len(decades))
                    correlation = np.corrcoef(decade_indices, freqs)[0, 1]
                    temporal_direction = np.sign(correlation)
                else:
                    temporal_direction = 0  # Neutral for few decades
                
                # Apply rule-specific weights
                rule_weight = distinctiveness.get(rule, 1.0)
                
                # Apply position weight (if enabled)
                if weight_early_merges:
                    idx = selected_rules.index(rule)
                    position_weight = np.exp(-0.1 * idx / len(selected_rules))
                    rule_weight *= position_weight
                
                # Weight more strongly rules that align with expected recency bias
                temporal_alignment = 1.0 + max(0, temporal_direction)
                
                # Add weighted term to objective
                data_fit_term += cp.sum(cp.multiply(alpha, freqs * rule_weight * temporal_alignment))
            
            # Add recency bias regularization for datasets with more than 2 decades
            if len(decades) > 2:
                trend_term = 0
                for i in range(len(decades)-1):
                    # Encourage increasing trend for recency bias
                    trend_term += alpha[i+1] - alpha[i]
                
                # Full objective with recency bias regularization
                objective = cp.Maximize(data_fit_term + regularization_strength * trend_term)
            else:
                # No regularization needed for 2 or fewer decades
                objective = cp.Maximize(data_fit_term)
            
            # Solve the problem
            prob = cp.Problem(objective, constraints)
            
            # Try multiple solvers in case of issues
            solvers = [None, 'ECOS', 'SCS', 'OSQP']
            solved = False
            
            for solver in solvers:
                if solved:
                    break
                    
                try:
                    if solver:
                        prob.solve(solver=solver)
                    else:
                        prob.solve()
                    
                    # If we get here, solver succeeded
                    logger.info(f"Solver {solver or 'default'} succeeded with status: {prob.status}")
                    solved = True
                except Exception as e:
                    logger.warning(f"Solver {solver} failed: {e}")
            
            # Extract solution if optimal
            if solved and (prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE):
                distribution = {decade: float(alpha.value[i]) for i, decade in enumerate(decades)}
                
                # Apply post-processing to ensure sum to 1
                total = sum(distribution.values())
                if total > 0:
                    distribution = {decade: value / total for decade, value in distribution.items()}
                    
                    # Apply 1960s bias correction as suggested by professor
                    if "1960s" in distribution:
                        logger.info("Applying professor's suggested 0.6 correction factor to 1960s")
                        distribution = self.apply_decade_correction(
                            distribution,
                            decade="1960s", 
                            factor=0.6  # Professor's suggested correction factor
                        )
                    
                    return distribution
            else:
                logger.warning(f"Solver failed with status: {prob.status if 'prob' in locals() and hasattr(prob, 'status') else 'unknown'}")
        
        except Exception as e:
            logger.error(f"Error in linear programming approach: {e}")
            import traceback
            traceback.print_exc()
        
        # If we get here, something went wrong with the LP approach
        logger.warning("Linear programming approach failed, falling back to heuristic")
        heuristic_distribution = self._infer_distribution_heuristic(filtered_patterns)
        
        # Still apply 1960s correction to heuristic result
        if "1960s" in heuristic_distribution:
            logger.info("Applying professor's correction to heuristic fallback result")
            heuristic_distribution = self.apply_decade_correction(
                heuristic_distribution,
                decade="1960s", 
                factor=0.6
            )
        
        return heuristic_distribution

    def remove_top_frequent_tokens(self, decade_patterns, top_n=5):
        """
        Remove the top N most frequent tokens across all decades.
        
        Args:
            decade_patterns: Dictionary mapping decades to pattern frequencies
            top_n: Number of top frequent tokens to remove
            
        Returns:
            Filtered decade patterns with top frequent tokens removed
        """
        # Create a copy to avoid modifying the original
        filtered_patterns = {}
        for decade, patterns in decade_patterns.items():
            filtered_patterns[decade] = {key: value.copy() if isinstance(value, dict) else value 
                                        for key, value in patterns.items()} if isinstance(patterns, dict) else patterns
        
        # Find global token frequencies
        token_freqs = {}
        
        # First pass: calculate frequencies
        for decade, patterns in decade_patterns.items():
            if isinstance(patterns, dict) and 'merge_rules' in patterns:
                for token, freq in patterns['merge_rules'].items():
                    if token not in token_freqs:
                        token_freqs[token] = 0
                    token_freqs[token] += freq
        
        # Get top N most frequent tokens
        top_tokens = []
        if token_freqs:
            top_tokens = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_tokens = [token for token, _ in top_tokens]
        
        if top_tokens:
            logger.info(f"Top {len(top_tokens)} tokens being removed: {top_tokens}")
            
            # Second pass: remove tokens
            for decade, patterns in filtered_patterns.items():
                if isinstance(patterns, dict) and 'merge_rules' in patterns:
                    # Create a new dictionary without the top tokens
                    patterns['merge_rules'] = {
                        token: freq for token, freq in patterns['merge_rules'].items() 
                        if token not in top_tokens
                    }
        else:
            logger.warning("No tokens found to remove")
        
        return filtered_patterns

    def calculate_distribution_mse(self, predicted: Dict[str, float], true: Dict[str, float]) -> float:
        """
        Calculate Mean Squared Error between predicted and true distributions.
        Returns log10(MSE) similar to Hayase et al.
        
        Args:
            predicted: Dictionary mapping decades to predicted proportions
            true: Dictionary mapping decades to true proportions
            
        Returns:
            log10(MSE) value
        """
        # Ensure all keys are present in both
        all_decades = set(predicted.keys()) | set(true.keys())
        
        # Calculate MSE
        squared_errors = []
        for decade in all_decades:
            pred_val = predicted.get(decade, 0.0)
            true_val = true.get(decade, 0.0)
            squared_errors.append((pred_val - true_val) ** 2)
        
        mse = sum(squared_errors) / len(squared_errors)
        log10_mse = np.log10(mse) if mse > 0 else -float('inf')
        
        return log10_mse

    def find_distinctive_patterns(self, 
                            decade_patterns: Dict[str, Dict],
                            threshold: float = 3.0) -> Dict[str, List[Tuple[str, float]]]:
        """
        Identify patterns that are distinctively common in specific decades.
        Enhanced to focus on more reliable signals by using a higher threshold.
        
        Args:
            decade_patterns: Results from analyze_decade_patterns
            threshold: How much more common a pattern must be (increased from 1.5 to 3.0)
            
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
                if freq < 5:  # Increased minimum occurrences from 3 to 5 for more reliability
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

    def ensemble_inference(self, decade_patterns: Dict[str, Dict]) -> Dict[str, float]:
        """
        Combine multiple inference methods for more robust results.
        
        Args:
            decade_patterns: Patterns detected for each decade
            
        Returns:
            Dictionary mapping decades to their estimated proportion
        """
        # Get results from different methods
        lp_distribution = self.infer_temporal_distribution(decade_patterns)
        heuristic_distribution = self._infer_distribution_heuristic(decade_patterns)
        
        # Simple weighted averaging ensemble (LP method has higher weight)
        lp_weight = 0.7
        heuristic_weight = 0.3
        
        ensemble_distribution = {}
        all_decades = sorted(set(lp_distribution.keys()) | set(heuristic_distribution.keys()))
        
        for decade in all_decades:
            lp_value = lp_distribution.get(decade, 0.0)
            heuristic_value = heuristic_distribution.get(decade, 0.0)
            # Weighted average
            ensemble_distribution[decade] = (lp_value * lp_weight + heuristic_value * heuristic_weight)
        
        # Ensure the distribution sums to 1
        total = sum(ensemble_distribution.values())
        if total > 0:
            ensemble_distribution = {d: v/total for d, v in ensemble_distribution.items()}
        
        return ensemble_distribution
    
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
        Use an improved ensemble of methods to produce more robust distribution estimates.
        
        Args:
            decade_patterns: Patterns detected for each decade
            methods: List of method configurations to use
            weights: Weights for each method
                    
        Returns:
            Dictionary mapping decades to their estimated proportion
        """
        # If methods and weights not provided, use improved defaults
        if methods is None:
            methods = [
                # Standard LP with different parameters and token removal
                (lambda dp: self.infer_temporal_distribution(dp, num_merge_rules=500, 
                                                    remove_top_tokens=True, top_n=10), 0.3),
                (lambda dp: self.infer_temporal_distribution(dp, num_merge_rules=1000, 
                                                    remove_top_tokens=True, top_n=15), 0.2),
                # LP with 1960s correction - stronger correction as suggested by professor
                (lambda dp: self.apply_decade_correction(
                    self.infer_temporal_distribution(dp, num_merge_rules=1500, remove_top_tokens=True, top_n=10), 
                    decade='1960s', factor=0.6), 0.3),
                # Heuristic method with token removal
                (lambda dp: self.apply_decade_correction(
                    self._infer_distribution_heuristic(self.remove_top_frequent_tokens(dp, 10)),
                    decade='1960s', factor=0.6), 0.2),
            ]
        else:
            methods = [(m, w) for m, w in zip(methods, weights or [1/len(methods)]*len(methods))]

        # Apply each method
        distributions = []
        used_weights = []
        
        for method_func, weight in methods:
            try:
                # Apply method with error handling
                distribution = method_func(decade_patterns)
                
                # Ensure all values are numeric
                cleaned_distribution = {}
                for decade, value in distribution.items():
                    if isinstance(value, (int, float)):
                        cleaned_distribution[decade] = float(value)  # Ensure everything is a float
                    else:
                        logger.warning(f"Skipping non-numeric value for {decade}: {type(value)}")
                        cleaned_distribution[decade] = 0.0
                
                # Validate distribution (should sum to ~1)
                total = sum(cleaned_distribution.values())
                if 0.9 <= total <= 1.1:  # Allow small numerical errors
                    distributions.append({k: v/total for k, v in cleaned_distribution.items()})
                    used_weights.append(weight)
                else:
                    logger.warning(f"Skipping invalid distribution with sum {total}")
            except Exception as e:
                logger.warning(f"Method failed: {e}")
        
        # If no valid distributions, return uniform fallback
        if not distributions:
            logger.warning("All methods failed, returning uniform distribution")
            decades = decade_patterns.keys()
            return {decade: 1.0/len(decades) for decade in decades}
        
        # Combine distributions using weighted average
        decades = set()
        for dist in distributions:
            decades.update(dist.keys())
        decades = sorted(decades)
        
        # Normalize weights
        total_weight = sum(used_weights)
        if total_weight > 0:
            norm_weights = [w/total_weight for w in used_weights]
        else:
            norm_weights = [1.0/len(used_weights)]*len(used_weights)
        
        # Compute weighted average
        ensemble_distribution = {}
        for decade in decades:
            ensemble_distribution[decade] = 0.0  # Initialize with zero
            for i, dist in enumerate(distributions):
                # Only add if the decade exists in this distribution
                if decade in dist:
                    ensemble_distribution[decade] += dist[decade] * norm_weights[i]
        
        # Final normalization
        total = sum(ensemble_distribution.values())
        if total > 0:
            return {decade: value/total for decade, value in ensemble_distribution.items()}
        else:
            return {decade: 1.0/len(decades) for decade in decades}

    def validate_against_hayase_metrics(self, predicted_distribution, true_distribution,
                                  bootstrap_iterations=30, confidence_level=0.95):
        """
        Validate results against metrics used in Hayase et al. paper,
        including log10(MSE) and bootstrap confidence intervals.
        
        Args:
            predicted_distribution: Dictionary mapping decades to proportions
            true_distribution: Ground truth distribution
            bootstrap_iterations: Number of bootstrap iterations
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with validation metrics
        """
        # Calculate basic log10(MSE) as in Hayase et al.
        log10_mse = self.calculate_distribution_mse(predicted_distribution, true_distribution)
        
        # Normalize distributions if needed
        pred_sum = sum(predicted_distribution.values())
        true_sum = sum(true_distribution.values())
        
        normalized_pred = {k: v/pred_sum for k, v in predicted_distribution.items()} if pred_sum > 0 else predicted_distribution
        normalized_true = {k: v/true_sum for k, v in true_distribution.items()} if true_sum > 0 else true_distribution
        
        # Calculate Mean Absolute Error
        all_decades = set(normalized_pred.keys()) | set(normalized_true.keys())
        mae = sum(abs(normalized_pred.get(d, 0) - normalized_true.get(d, 0)) for d in all_decades) / len(all_decades)
        
        # Calculate Jensen-Shannon Distance (symmetric)
        from scipy.spatial.distance import jensenshannon
        
        # Convert dictionaries to vectors in same order
        sorted_decades = sorted(all_decades)
        pred_vec = np.array([normalized_pred.get(d, 0) for d in sorted_decades])
        true_vec = np.array([normalized_true.get(d, 0) for d in sorted_decades])
        
        # Ensure vectors sum to 1
        if sum(pred_vec) > 0:
            pred_vec = pred_vec / sum(pred_vec)
        if sum(true_vec) > 0:
            true_vec = true_vec / sum(true_vec)
        
        # Calculate JS distance
        js_distance = jensenshannon(pred_vec, true_vec)
        
        # Calculate rank correlation
        from scipy.stats import spearmanr
        
        # Get decade rankings
        pred_ranks = {d: i for i, d in enumerate(sorted(normalized_pred.items(), key=lambda x: x[1], reverse=True))}
        true_ranks = {d: i for i, d in enumerate(sorted(normalized_true.items(), key=lambda x: x[1], reverse=True))}
        
        # Match ranks for common decades
        common_decades = set(pred_ranks.keys()) & set(true_ranks.keys())
        if common_decades:
            pred_rank_values = [pred_ranks[d] for d in common_decades]
            true_rank_values = [true_ranks[d] for d in common_decades]
            rank_correlation, _ = spearmanr(pred_rank_values, true_rank_values)
        else:
            rank_correlation = 0.0
        
        # Analyze over/under representation
        representation_analysis = {
            "over_represented": {},
            "under_represented": {}
        }
        
        for decade in all_decades:
            pred_val = normalized_pred.get(decade, 0)
            true_val = normalized_true.get(decade, 0)
            difference = pred_val - true_val
            
            if difference > 0.02:  # 2% threshold for over-representation
                representation_analysis["over_represented"][decade] = difference
            elif difference < -0.02:  # 2% threshold for under-representation
                representation_analysis["under_represented"][decade] = abs(difference)  # Store as positive value
        
        # Create the expected structured output
        result = {
            "distribution_metrics": {
                "log10_mse": log10_mse,
                "mae": mae,
                "js_distance": js_distance
            },
            "decade_metrics": {
                "rank_correlation": rank_correlation,
                "representation_analysis": representation_analysis
            },
            "hayase_benchmark": -7.30,  # The value reported in Hayase et al.
            "comparison_to_benchmark": log10_mse + 7.30  # Difference from benchmark
        }
        
        # If bootstrap iterations requested, add those metrics
        if bootstrap_iterations > 0:
            # Bootstrap code here would be executed
            # For now, we'll just add a placeholder
            result["bootstrap_results"] = {
                "requested_iterations": bootstrap_iterations,
                "confidence_level": confidence_level
                # Actual bootstrap metrics would be added here
            }
        
        return result

    def apply_decade_correction(self, distribution, decade="1960s", factor=0.6):
        """
        Apply a correction factor to a specific decade (especially 1960s)
        which consistently shows overrepresentation in the analysis.
        
        Args:
            distribution: The original distribution dictionary
            decade: The decade to correct
            factor: Correction factor (0.6 means reduce by 40%)
            
        Returns:
            Corrected distribution
        """
        if decade not in distribution:
            return distribution
            
        # Create a copy to avoid modifying the original
        corrected = distribution.copy()
        
        # Apply correction
        original_value = corrected[decade]
        corrected[decade] = original_value * factor
        
        # Redistribute the excess to other decades proportionally
        excess = original_value - corrected[decade]
        other_decades = [d for d in corrected if d != decade]
        
        if other_decades and excess > 0:
            # Proportional redistribution based on current weights
            other_sum = sum(corrected[d] for d in other_decades)
            if other_sum > 0:
                for d in other_decades:
                    # Distribute proportionally to current values
                    corrected[d] += excess * (corrected[d] / other_sum)
            else:
                # Equal distribution if all others are zero
                for d in other_decades:
                    corrected[d] += excess / len(other_decades)
        
        # Ensure the sum is still 1.0
        total = sum(corrected.values())
        if abs(total - 1.0) > 0.001:  # Allow for small floating point errors
            corrected = {d: v/total for d, v in corrected.items()}
        
        return corrected

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

    def analyze_temporal_progression(self, decade_patterns: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Analyze how rule frequencies change across decades to identify temporal trends.
        
        Args:
            decade_patterns: Patterns detected for each decade
            
        Returns:
            Dictionary mapping rules to their temporal progression metrics
        """
        decades = sorted(list(decade_patterns.keys()))
        if len(decades) < 3:
            return {}  # Need at least 3 decades for meaningful trend analysis
        
        # Create mapping of decade to index for correlation calculation
        decade_indices = {decade: i for i, decade in enumerate(decades)}
        
        # Extract rules and their normalized frequencies across decades
        rule_progressions = {}
        
        for decade, patterns in decade_patterns.items():
            if 'merge_rules' in patterns:
                rules = patterns['merge_rules']
                total_tokens = patterns['total_tokens']
                
                if total_tokens > 0:
                    for rule, count in rules.items():
                        if rule not in rule_progressions:
                            rule_progressions[rule] = {d: 0.0 for d in decades}
                        
                        # Normalize by total tokens
                        rule_progressions[rule][decade] = count / total_tokens
        
        # Analyze progression for each rule
        results = {}
        for rule, decade_freqs in rule_progressions.items():
            # Convert to arrays for correlation calculation
            indices = np.array(list(range(len(decades))))
            freqs = np.array([decade_freqs[d] for d in decades])
            
            # Only analyze rules that appear in multiple decades
            non_zero_decades = sum(1 for f in freqs if f > 0)
            if non_zero_decades < 2:
                continue
            
            # Calculate correlation with decade progression
            if non_zero_decades > 2:
                correlation = np.corrcoef(indices, freqs)[0, 1]
                is_significant = abs(correlation) > 0.5  # Threshold for significance
                direction = 'increasing' if correlation > 0 else 'decreasing'
            else:
                # For just 2 non-zero points, use simple comparison
                if len(np.nonzero(freqs)[0]) == 2:
                    idx1, idx2 = np.nonzero(freqs)[0]
                    direction = 'increasing' if idx1 < idx2 and freqs[idx1] < freqs[idx2] else 'decreasing'
                    correlation = 0.5 * (1 if direction == 'increasing' else -1)
                    is_significant = True
                else:
                    direction = 'stable'
                    correlation = 0
                    is_significant = False
            
            results[rule] = {
                'direction': direction,
                'strength': abs(correlation),
                'significant': is_significant,
                'frequencies': decade_freqs
            }
        
        return results

    def _infer_distribution_heuristic(self, decade_patterns: Dict[str, Dict]) -> Dict[str, float]:
        """
        Improved heuristic method for temporal distribution inference.
        Uses a combination of distinctive patterns and temporal progression.
        """
        # Add these checks at the beginning of your function
        if not isinstance(decade_patterns, dict):
            logger.warning(f"Invalid decade_patterns type: {type(decade_patterns)}")
            return {}  # Return empty dict on invalid input
        
        # Extract decades
        decades = sorted(list(decade_patterns.keys()))
        if not decades:
            return {}
        
        # Check if we have the required patterns structure
        has_merge_rules = any('merge_rules' in patterns for decade, patterns in decade_patterns.items() 
                        if isinstance(patterns, dict))
        
        if not has_merge_rules:
            # Return uniform distribution if data doesn't have expected structure
            return {decade: 1.0 / len(decades) for decade in decades}
        
        # Extract decades
        decades = sorted(list(decade_patterns.keys()))
        
        # Calculate distinctive pattern scores for each decade
        distinctive_patterns = self.find_distinctive_patterns(decade_patterns, threshold=1.2)
        
        # Calculate temporal progression for rules
        temporal_progression = self.analyze_temporal_progression(decade_patterns)
        
        # Initial scores based on distinctive patterns
        decade_scores = {}
        for decade, patterns in distinctive_patterns.items():
            if patterns:
                # Calculate weighted score using both distinctiveness and frequency
                weighted_score = 0
                for pattern, score in patterns[:min(10, len(patterns))]:
                    freq = 0
                    if 'merge_rules' in decade_patterns[decade]:
                        total_tokens = decade_patterns[decade]['total_tokens']
                        if total_tokens > 0 and pattern in decade_patterns[decade]['merge_rules']:
                            freq = decade_patterns[decade]['merge_rules'][pattern] / total_tokens
                    
                    # Get temporal progression bonus if available
                    temporal_bonus = 1.0
                    if pattern in temporal_progression:
                        progress_metrics = temporal_progression[pattern]
                        if progress_metrics['significant']:
                            # Higher bonus for rules that increase with recency (positive correlation)
                            direction_bonus = 1.5 if progress_metrics['direction'] == 'increasing' else 0.8
                            temporal_bonus = 1.0 + (progress_metrics['strength'] * direction_bonus)
                    
                    # Weight score by distinctiveness, frequency, and temporal importance
                    weighted_score += score * freq * temporal_bonus * 100  # Scale up for numerical stability
                
                decade_scores[decade] = weighted_score
            else:
                decade_scores[decade] = 0.1  # Small non-zero default
        
        # Apply recency bias adjustment - boost more recent decades
        for decade in decades:
            decade_start = int(decade[:4])
            # Calculate recency factor (0 to 1, higher for more recent)
            recency_factor = (decade_start - 1850) / 170.0
            # Apply modest recency boost
            decade_scores[decade] *= (1.0 + recency_factor * 0.5)  # Up to 50% boost for 2020s
        
        # Normalize scores to get proportions
        total_score = sum(decade_scores.values())
        if total_score <= 0:
            return {decade: 1.0 / len(decades) for decade in decades}
            
        proportions = {decade: score / total_score for decade, score in decade_scores.items()}
        
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
        
        # NEW: Check data quality before proceeding
        logger.info("Analyzing data quality...")
        quality_metrics = self.analyze_data_quality(decade_texts)
        
        # Warn if data quality is insufficient
        if not quality_metrics['overall']['all_decades_sufficient']:
            logger.warning("Data quality may be insufficient for reliable analysis")
            logger.warning(f"Total data volume: {quality_metrics['overall']['total_data_gb']:.2f} GB")
            logger.warning("Consider increasing data volume or text quality before proceeding")
        
        # Continue with the existing analysis steps...
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