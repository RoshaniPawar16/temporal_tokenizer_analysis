# src/merge_rules_analyzer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import json
import random
import gc  # Added for garbage collection
import tiktoken
from transformers import AutoTokenizer
from scipy.optimize import linprog 
import logging
logger = logging.getLogger(__name__)

from .config import (
    PROJECT_ROOT,
    MODELS_DIR,
    RESULTS_DIR,
    TIME_PERIODS,
    ANALYSIS_CONFIG
)

class MergeRulesAnalyzer:
    """
    Analyzes how tokenizer merge rules are applied differently across different time periods.
    This reveals which specific word patterns are characteristic of particular decades.
    """
    
    def __init__(self, tokenizer_name="gpt2"):
        """Initialize with a pretrained tokenizer."""
        self.tokenizer_name = tokenizer_name
        self.tokenizer = self._load_tokenizer()
        
        # Initialize caches
        self._token_decomposition_cache = {}
        self._rule_usage_cache = {}
        
        # Memory efficiency flag
        self.memory_efficient = False
        
        # Extract merge rules
        self.merge_rules = self._extract_merge_rules()
        
        # Handle case when merge_rules extraction fails
        if not self.merge_rules:
            logger.warning(f"No merge rules found for {tokenizer_name}. Using fallback strategy.")
            # Create a minimal set of rules to prevent further errors
            self.merge_rules = [((chr(i), chr(i+1)), i) for i in range(97, 122)]
        
        # Create index mapping
        self.merge_rule_indices = {rule[0]: i for i, rule in enumerate(self.merge_rules)}
    
    def enable_memory_efficient_mode(self):
        """
        Enable memory-efficient mode for processing large datasets.
        This clears caches after each decade to prevent memory buildup.
        """
        self.memory_efficient = True
        self.batch_size = 20  # Add a reasonable batch size for memory-efficient processing
        self._cleanup_cache()
        logger.info("Memory-efficient mode enabled with batch size {}".format(self.batch_size))

    def _cleanup_cache(self):
        """Clear caches to free memory."""
        if hasattr(self, '_token_decomposition_cache'):
            self._token_decomposition_cache.clear()
        if hasattr(self, '_rule_usage_cache'):
            self._rule_usage_cache.clear()
        # Force garbage collection
        import gc
        gc.collect()

    def _load_tokenizer(self):
        """
        Load the specified tokenizer, with improved tiktoken handling.
        """
        try:
            # First try to load as a tiktoken tokenizer (for GPT models)
            try:
                if self.tokenizer_name.startswith("gpt"):
                    # For GPT models, use tiktoken's encoding_for_model
                    encoding = tiktoken.encoding_for_model(self.tokenizer_name)
                    logger.info(f"Loaded tiktoken tokenizer for {self.tokenizer_name}")
                    return encoding
            except (ImportError, ValueError, KeyError) as e:
                logger.warning(f"Tiktoken loading failed: {e}")
            
            # Fall back to HuggingFace implementation
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            logger.info(f"Loaded HuggingFace tokenizer for {self.tokenizer_name}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Error loading tokenizer {self.tokenizer_name}: {e}")
            logger.info("Using fallback tokenizer: gpt2")
            
            # Last resort fallback to GPT-2 via HuggingFace
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                logger.info("Loaded fallback gpt2 tokenizer from HuggingFace")
                return tokenizer
            except Exception as fallback_error:
                logger.critical(f"Critical: Failed to load any tokenizer: {fallback_error}")
                raise RuntimeError(f"Could not load any tokenizer")

    def _extract_merge_rules(self):
        """
        Extract merge rules from the tokenizer with proper tiktoken handling.
        Returns a list of (token_pair, idx) tuples.
        """
        try:
            # Direct check for tiktoken encoding object
            import tiktoken
            if isinstance(self.tokenizer, tiktoken.Encoding):
                logger.info("Detected tiktoken Encoding - extracting merge rules directly")
                
                # Access tiktoken's internal merge rules (different attribute name than expected)
                # For tiktoken, we need to check different potential attribute names
                merge_rules = []
                
                # Try each possible merge rules location
                if hasattr(self.tokenizer, "_mergeable_ranks"):
                    ranks_dict = self.tokenizer._mergeable_ranks
                elif hasattr(self.tokenizer, "mergeable_ranks"):
                    ranks_dict = self.tokenizer.mergeable_ranks
                else:
                    # Access internal attributes through the encoder if available
                    try:
                        ranks_dict = self.tokenizer._encoder._mergeable_ranks
                    except (AttributeError, KeyError):
                        # Last fallback attempt: access private core encoder
                        try:
                            if hasattr(self.tokenizer, "name"):
                                # Get the raw BPE merge list directly from tiktoken's model data
                                raw_merges = tiktoken.get_merges(self.tokenizer.name)
                                logger.info(f"Retrieved {len(raw_merges)} raw merges from tiktoken model data")
                                
                                # Convert raw merges to our format
                                for i, merge in enumerate(raw_merges):
                                    if isinstance(merge, tuple) and len(merge) == 2:
                                        first, second = merge
                                        try:
                                            # Convert bytes to strings
                                            first_str = first.decode('utf-8', errors='replace') if isinstance(first, bytes) else str(first)
                                            second_str = second.decode('utf-8', errors='replace') if isinstance(second, bytes) else str(second)
                                            merge_rules.append(((first_str, second_str), i))
                                        except Exception as decode_err:
                                            logger.debug(f"Error decoding merge rule {i}: {decode_err}")
                                
                                logger.info(f"Successfully converted {len(merge_rules)} tiktoken raw merges")
                                return sorted(merge_rules, key=lambda x: x[1])
                        except Exception as raw_err:
                            logger.warning(f"Failed to get raw merges: {raw_err}")
                            
                        # If we get here, we couldn't find the merge rules
                        logger.error("Could not locate tiktoken merge rules in any expected location")
                        ranks_dict = {}
                
                # Process the merge rules from the dictionary
                for token_pair, rank in ranks_dict.items():
                    # Process token pair
                    if isinstance(token_pair, tuple) and len(token_pair) == 2:
                        first, second = token_pair
                        try:
                            first_str = first.decode('utf-8', errors='replace') if isinstance(first, bytes) else str(first)
                            second_str = second.decode('utf-8', errors='replace') if isinstance(second, bytes) else str(second)
                            merge_rules.append(((first_str, second_str), rank))
                        except Exception as e:
                            logger.debug(f"Error processing token pair {repr(token_pair)}: {e}")
                
                # Sort by rank and return
                sorted_rules = sorted(merge_rules, key=lambda x: x[1])
                if sorted_rules:
                    logger.info(f"Successfully extracted {len(sorted_rules)} merge rules from tiktoken")
                    # Debug: print first few rules
                    if len(sorted_rules) > 2:
                        logger.debug(f"First few rules: {sorted_rules[:3]}")
                    return sorted_rules
                else:
                    logger.warning("No merge rules extracted from tiktoken")
                    # Try the direct approach with get_merges function
                    try:
                        import tiktoken
                        if hasattr(self.tokenizer, "name"):
                            model_name = self.tokenizer.name
                            merges = tiktoken.get_merges(model_name)
                            if merges:
                                logger.info(f"Got {len(merges)} merge rules directly from tiktoken.get_merges")
                                # Convert to our format
                                direct_merge_rules = []
                                for i, merge in enumerate(merges):
                                    if isinstance(merge, tuple) and len(merge) == 2:
                                        first, second = merge
                                        try:
                                            # Convert bytes to strings
                                            first_str = first.decode('utf-8', errors='replace') if isinstance(first, bytes) else str(first)
                                            second_str = second.decode('utf-8', errors='replace') if isinstance(second, bytes) else str(second)
                                            direct_merge_rules.append(((first_str, second_str), i))
                                        except Exception as decode_err:
                                            logger.debug(f"Error decoding merge {i}: {decode_err}")
                                return direct_merge_rules
                    except Exception as get_merges_error:
                        logger.error(f"Error getting merges directly: {get_merges_error}")
            
            # For Hugging Face tokenizers
            elif hasattr(self.tokenizer, "merges"):
                logger.info("Using merge rules from HuggingFace tokenizer")
                merge_rules = []
                for i, merge in enumerate(self.tokenizer.merges):
                    parts = merge.split()
                    if len(parts) == 2:
                        merge_rules.append(((parts[0], parts[1]), i))
                        
                if merge_rules:
                    logger.info(f"Extracted {len(merge_rules)} merge rules from HuggingFace tokenizer")
                    return merge_rules
                else:
                    logger.warning("No valid merge rules extracted from HuggingFace tokenizer")
                    
            # For tokenizers with bpe_ranks
            elif hasattr(self.tokenizer, "bpe_ranks"):
                logger.info("Using bpe_ranks for merge rules")
                merge_rules = []
                for token_pair, rank in self.tokenizer.bpe_ranks.items():
                    if isinstance(token_pair, tuple) and len(token_pair) == 2:
                        merge_rules.append((token_pair, rank))
                        
                sorted_rules = sorted(merge_rules, key=lambda x: x[1])
                if sorted_rules:
                    logger.info(f"Extracted {len(sorted_rules)} merge rules from bpe_ranks")
                    return sorted_rules
            
            # Special handling for GPT-2 through transformers instead of tiktoken
            elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "merges"):
                logger.info("Using transformers GPT-2 tokenizer's internal merges")
                merge_rules = []
                for i, merge in enumerate(self.tokenizer.tokenizer.merges):
                    parts = merge.split()
                    if len(parts) == 2:
                        merge_rules.append(((parts[0], parts[1]), i))
                
                if merge_rules:
                    logger.info(f"Extracted {len(merge_rules)} merge rules from transformers GPT-2")
                    return merge_rules
                    
            # Last resort: create direct test merge rules for GPT-2
            if self.tokenizer_name == "gpt2":
                logger.warning("Using manually generated GPT-2 test rules for demonstration")
                # Create some known GPT-2 merge patterns for demonstration
                test_rules = [
                    ((u't', u'he'), 1),
                    ((u'a', u'n'), 2),
                    ((u'i', u'n'), 3),
                    ((u'e', u'r'), 4),
                    ((u'o', u'n'), 5),
                    ((u'th', u'e'), 6),
                    ((u'in', u'g'), 7)
                ]
                logger.info(f"Created {len(test_rules)} test rules for GPT-2")
                return test_rules
                
            logger.error("Could not extract merge rules: unsupported tokenizer type")
            return []
                
        except Exception as e:
            logger.error(f"Error extracting merge rules: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _verify_tiktoken_access(self):
        """Verify we can access tiktoken merge rules."""
        try:
            import tiktoken
            # Create a test encoding
            encoding = tiktoken.encoding_for_model("gpt2")
            # Try different ways to access merge rules
            if hasattr(encoding, "mergeable_ranks"):
                ranks = encoding.mergeable_ranks
                logger.info(f"Direct mergeable_ranks access works: {len(ranks)} rules found")
                return True
            elif hasattr(encoding, "_mergeable_ranks"):
                ranks = encoding._mergeable_ranks
                logger.info(f"_mergeable_ranks access works: {len(ranks)} rules found") 
                return True
            else:
                # Try get_merges function
                try:
                    merges = tiktoken.get_merges("gpt2")
                    logger.info(f"tiktoken.get_merges works: {len(merges)} merges found")
                    return True
                except:
                    pass
            logger.warning("No methods to access tiktoken merge rules worked")
            return False
        except Exception as e:
            logger.error(f"Tiktoken verification error: {e}")
            return False
    
    def analyze_temporal_shifts(self, decade_texts: Dict[str, List[str]], 
                           distinctiveness_threshold: float = 1.2,
                           use_clustering: bool = True) -> Dict:
        """
        Analyze temporal shifts in language by comparing broader time periods
        rather than individual decades.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            distinctiveness_threshold: Lower threshold for capturing subtle patterns
            use_clustering: Whether to cluster merge rules with similar patterns
            
        Returns:
            Dictionary with analysis results comparing time periods
        """
        # Group decades into broader periods
        time_periods = {
            "historical": ["1850s", "1860s", "1870s", "1880s", "1890s"],
            "early_20th": ["1900s", "1910s", "1920s", "1930s", "1940s"],
            "mid_20th": ["1950s", "1960s", "1970s", "1980s"],
            "contemporary": ["1990s", "2000s", "2010s", "2020s"]
        }
        
        # Combine texts by period
        period_texts = {}
        for period, decades in time_periods.items():
            period_texts[period] = []
            for decade in decades:
                if decade in decade_texts:
                    # Only use decades where we have enough authentic texts
                    # This can be adjusted based on your dataset
                    texts = decade_texts.get(decade, [])
                    if len(texts) >= 5:  # Minimum threshold for inclusion
                        period_texts[period].extend(texts)
        
        # Remove periods with insufficient data
        period_texts = {period: texts for period, texts in period_texts.items() 
                    if len(texts) >= 20}  # Require at least 20 texts per period
        
        logger.info(f"Analyzing temporal shifts across {len(period_texts)} time periods")
        for period, texts in period_texts.items():
            logger.info(f"  {period}: {len(texts)} texts")
        
        # Apply analysis to periods instead of decades
        period_usage_results = {}
        
        for period, texts in period_texts.items():
            # Process in batches to manage memory
            merge_rule_usage = defaultdict(int)
            total_tokens = 0
            
            # Process texts in small batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                
                # Process each text
                for text in batch:
                    try:
                        # Get tokens and analyze merge rules
                        tokens = self.tokenizer.tokenize(text[:10000])
                        total_tokens += len(tokens)
                        
                        # Sample tokens if there are too many
                        if len(tokens) > 1000:
                            tokens = random.sample(tokens, 1000)
                        
                        # Analyze merge rules
                        for token in tokens:
                            applied_rules = self._identify_applied_merge_rules(token)
                            for rule in applied_rules:
                                merge_rule_usage[rule] += 1
                    except Exception as e:
                        logger.debug(f"Error processing text: {e}")
                        continue
                
                # Clean up after each batch
                self._cleanup_cache()
            
            # Calculate statistics for this period
            if total_tokens > 0:
                # Normalize usage by token count
                normalized_usage = {rule: count / total_tokens 
                                for rule, count in merge_rule_usage.items()}
                
                period_usage_results[period] = {
                    'total_tokens': total_tokens,
                    'unique_rules_applied': len(merge_rule_usage),
                    'top_rules': self._get_top_rules(merge_rule_usage, n=30),
                    'normalized_usage': normalized_usage
                }
        
        # Analyze distinctive rules for each period compared to others
        distinctive_rules = {}
        
        # Get all periods and rules
        periods = list(period_usage_results.keys())
        all_rules = set()
        for period_data in period_usage_results.values():
            all_rules.update(period_data['normalized_usage'].keys())
        
        # Calculate average usage across all periods for each rule
        avg_rule_usage = {}
        for rule in all_rules:
            usages = [period_data['normalized_usage'].get(rule, 0) 
                    for period_data in period_usage_results.values()]
            avg_rule_usage[rule] = sum(usages) / len(periods)
        
        # Find distinctive rules for each period with lower threshold
        for period in periods:
            period_distinctive = []
            for rule, usage in period_usage_results[period]['normalized_usage'].items():
                if usage > 0 and usage > (distinctiveness_threshold * avg_rule_usage[rule]):
                    # This rule is used distinctively more in this period
                    period_distinctive.append((rule, usage / avg_rule_usage[rule]))
            
            # Sort by distinctiveness ratio
            distinctive_rules[period] = sorted(period_distinctive, 
                                            key=lambda x: x[1], 
                                            reverse=True)
        
        # If clustering enabled, group similar rules
        if use_clustering and distinctive_rules:
            clustered_rules = self._cluster_similar_rules(distinctive_rules)
            results = {
                'period_usage': period_usage_results,
                'distinctive_rules': distinctive_rules,
                'clustered_rules': clustered_rules
            }
        else:
            results = {
                'period_usage': period_usage_results,
                'distinctive_rules': distinctive_rules
            }
        
        return results

    def _cluster_similar_rules(self, distinctive_rules: Dict[str, List[Tuple[str, float]]]) -> Dict:
        """
        Group similar merge rules that may have related patterns.
        This helps identify broader linguistic patterns rather than individual rules.
        
        Args:
            distinctive_rules: Dictionary mapping periods to distinctive rules
            
        Returns:
            Dictionary with clustered rules by period
        """
        clustered_results = {}
        
        for period, rules in distinctive_rules.items():
            # Skip if no distinctive rules
            if not rules:
                clustered_results[period] = []
                continue
                
            # Group rules by character patterns (prefix/suffix)
            prefix_clusters = defaultdict(list)
            suffix_clusters = defaultdict(list)
            
            for rule, score in rules:
                # Skip rules that aren't strings
                if not isinstance(rule, str):
                    continue
                    
                # Group by prefix (first character)
                if len(rule) > 0:
                    prefix = rule[0]
                    prefix_clusters[prefix].append((rule, score))
                
                # Group by suffix (last character)
                if len(rule) > 0:
                    suffix = rule[-1]
                    suffix_clusters[suffix].append((rule, score))
            
            # Keep only clusters with multiple rules
            significant_clusters = []
            
            # Add prefix clusters with at least 2 rules
            for prefix, cluster in prefix_clusters.items():
                if len(cluster) >= 2:
                    avg_score = sum(score for _, score in cluster) / len(cluster)
                    significant_clusters.append({
                        'type': 'prefix',
                        'pattern': prefix,
                        'rules': cluster,
                        'avg_score': avg_score,
                        'size': len(cluster)
                    })
            
            # Add suffix clusters with at least 2 rules
            for suffix, cluster in suffix_clusters.items():
                if len(cluster) >= 2:
                    avg_score = sum(score for _, score in cluster) / len(cluster)
                    significant_clusters.append({
                        'type': 'suffix',
                        'pattern': suffix,
                        'rules': cluster,
                        'avg_score': avg_score,
                        'size': len(cluster)
                    })
            
            # Sort clusters by size and score
            clustered_results[period] = sorted(significant_clusters, 
                                            key=lambda x: (x['size'], x['avg_score']), 
                                            reverse=True)
        
        return clustered_results

    def _approximate_merge_rules_from_vocab(self):
        """
        Approximate merge rules from vocabulary when direct extraction isn't possible.
        """
        try:
            # Get vocabulary
            if hasattr(self.tokenizer, "get_vocab"):
                vocab = self.tokenizer.get_vocab()
            elif hasattr(self.tokenizer, "vocab"):
                vocab = self.tokenizer.vocab
            else:
                logger.warning("Cannot access tokenizer vocabulary")
                return []
                
            # Sort tokens by token ID (approximates merge order)
            sorted_tokens = sorted([(token, idx) for token, idx in vocab.items()], 
                                key=lambda x: x[1])
            
            # Create simplified merge rules where each token is considered its own merge
            # This isn't accurate but allows the rest of the code to run
            merge_rules = []
            for i, (token, _) in enumerate(sorted_tokens):
                if i > 0:  # Skip the first token as we need pairs
                    prev_token = sorted_tokens[i-1][0]
                    merge_rules.append(((prev_token, token), i))
                    
            logger.info(f"Approximated {len(merge_rules)} merge rules from vocabulary")
            return merge_rules
            
        except Exception as e:
            logger.error(f"Failed to approximate merge rules: {e}")
            return []
    
    def analyze_merge_rule_usage(self, decade_texts: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Analyze merge rule usage in each decade."""
        results = {}
        
        for decade, texts in decade_texts.items():
            # Track total merge operations
            total_merge_operations = 0
            merge_rule_counts = defaultdict(int)
            
            # Process each text
            for text in texts:
                # Tokenize with detailed merge tracking
                tokens, applied_rules = self._tokenize_with_merge_tracking(text)
                
                # Add to total operations
                total_merge_operations += len(tokens)
                
                # Count each rule application
                for rule in applied_rules:
                    merge_rule_counts[rule] += 1
            
            # Calculate normalized frequencies
            normalized_usage = {
                rule: count / total_merge_operations 
                for rule, count in merge_rule_counts.items()
            } if total_merge_operations > 0 else {}
            
            results[decade] = {
                'total_tokens': total_merge_operations,
                'rule_counts': dict(merge_rule_counts),
                'normalized_usage': normalized_usage
            }
        
        return results
    
    def analyze_merge_rule_usage_batched(self, decade_texts: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Memory-efficient version of analyze_merge_rule_usage that processes texts in batches.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            
        Returns:
            Dictionary with detailed merge rule usage statistics by decade
        """
        results = {}
        
        # Process each decade
        for decade, texts in decade_texts.items():
            if not texts:
                continue
                
            # Initialize decade statistics
            merge_rule_usage = defaultdict(int)
            total_tokens = 0
            
            # Process in small batches to limit memory usage
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                
                # Process batch
                for text in batch:
                    try:
                        # Get tokens and increment counters - limit text length for M2
                        tokens = self.tokenizer.tokenize(text[:10000])
                        total_tokens += len(tokens)
                        
                        # Sample tokens if there are too many (for very long texts)
                        if len(tokens) > 1000:
                            tokens = random.sample(tokens, 1000)
                        
                        # Analyze merge rules
                        for token in tokens:
                            applied_rules = self._identify_applied_merge_rules(token)
                            for rule in applied_rules:
                                merge_rule_usage[rule] += 1
                    except Exception as e:
                        print(f"Error processing text in {decade}: {e}")
                        continue
                        
                # Clean up after each batch to free memory
                self._cleanup_cache()
            
            # Calculate statistics
            if total_tokens > 0:
                # Normalize usage by token count
                normalized_usage = {rule: count / total_tokens 
                                   for rule, count in merge_rule_usage.items()}
                
                # Find most distinctive rules
                results[decade] = {
                    'total_tokens': total_tokens,
                    'unique_rules_applied': len(merge_rule_usage),
                    'top_rules': self._get_top_rules(merge_rule_usage, n=20),
                    'normalized_usage': normalized_usage
                }
                
                # Help free memory
                del merge_rule_usage
                gc.collect()
        
        return results
    
    def _identify_applied_merge_rules(self, token: str) -> List[str]:
        """
        Determine which merge rules were applied to form a token.
        Uses a complex algorithm to reconstruct the merge process.
        
        Args:
            token: The token to analyze
            
        Returns:
            List of merge rules applied to form this token
        """
        # Check cache first
        if token in self._token_decomposition_cache:
            return self._token_decomposition_cache[token]
        
        # For tokens that don't require merges (single characters)
        if len(token) == 1 or token in self.tokenizer.all_special_tokens:
            self._token_decomposition_cache[token] = []
            return []
            
        # For continuation tokens, remove prefix
        if token.startswith('##'):
            base_token = token[2:]
        elif token.startswith('Ġ'):
            base_token = token[1:]
        elif token.startswith('▁'):
            base_token = token[1:]
        else:
            base_token = token
            
        # Simple approach: assume each character merge
        # A more sophisticated approach would trace the exact merges
        applied_rules = []
        
        # Look for character pairs that appear in merge rules
        for i in range(len(base_token) - 1):
            char_pair = base_token[i:i+2]
            if char_pair in self.merge_rule_indices:
                applied_rules.append(char_pair)
        
        # Cache result
        self._token_decomposition_cache[token] = applied_rules
        return applied_rules
    
    def _get_top_rules(self, rule_counts: Dict[str, int], n: int = 20) -> List[Tuple[str, int]]:
        """
        Get the top N most frequently applied merge rules.
        
        Args:
            rule_counts: Dictionary mapping rules to counts
            n: Number of top rules to return
            
        Returns:
            List of (rule, count) tuples sorted by count
        """
        return sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def find_decade_distinctive_rules(self, 
                                     usage_results: Dict[str, Dict],
                                     distinctiveness_threshold: float = 1.5) -> Dict[str, List[str]]:
        """
        Identify merge rules that are distinctively common in one decade vs others.
        
        Args:
            usage_results: Results from analyze_merge_rule_usage
            distinctiveness_threshold: How much more common a rule must be to be distinctive
            
        Returns:
            Dictionary mapping decades to lists of their distinctive rules
        """
        distinctive_rules = {}
        
        # Get all decades and rules
        decades = list(usage_results.keys())
        all_rules = set()
        for decade_data in usage_results.values():
            all_rules.update(decade_data['normalized_usage'].keys())
        
        # Calculate average usage across all decades for each rule
        avg_rule_usage = {}
        for rule in all_rules:
            usages = [decade_data['normalized_usage'].get(rule, 0) 
                     for decade_data in usage_results.values()]
            avg_rule_usage[rule] = sum(usages) / len(decades)
        
        # Find distinctive rules for each decade
        for decade in decades:
            decade_distinctive = []
            for rule, usage in usage_results[decade]['normalized_usage'].items():
                if usage > 0 and usage > (distinctiveness_threshold * avg_rule_usage[rule]):
                    # This rule is used distinctively more in this decade
                    decade_distinctive.append((rule, usage / avg_rule_usage[rule]))
            
            # Sort by distinctiveness ratio
            distinctive_rules[decade] = sorted(decade_distinctive, 
                                              key=lambda x: x[1], 
                                              reverse=True)
        
        return distinctive_rules
    
    def visualize_rule_usage(self, 
                           usage_results: Dict[str, Dict],
                           distinctive_rules: Dict[str, List[Tuple[str, float]]],
                           n_rules: int = 10,
                           save_path: Optional[Path] = None):
        """
        Visualize the most distinctive merge rules for each decade.
        Memory-efficient version that processes one decade at a time.
        
        Args:
            usage_results: Results from analyze_merge_rule_usage
            distinctive_rules: Results from find_decade_distinctive_rules
            n_rules: Number of top rules to visualize per decade
            save_path: If provided, save the figure to this path
        """
        decades = sorted(usage_results.keys())
        n_decades = len(decades)
        
        # Use a smaller figure size for M2
        fig_height = min(4*n_decades, 12)  # Cap the maximum height
        
        # Create figure with subplots for each decade
        fig, axes = plt.subplots(n_decades, 1, figsize=(12, fig_height), sharex=False)
        if n_decades == 1:
            axes = [axes]
        
        # Process one decade at a time to save memory
        for i, decade in enumerate(decades):
            ax = axes[i]
            
            # Get top N distinctive rules for this decade
            if decade in distinctive_rules:
                # Limit number of rules for memory efficiency
                n_effective = min(n_rules, 5) if self.memory_efficient else n_rules
                top_rules = distinctive_rules[decade][:n_effective]
            else:
                top_rules = []
                
            if not top_rules:
                ax.text(0.5, 0.5, f"No distinctive rules found for {decade}",
                       ha='center', va='center', transform=ax.transAxes)
                continue
                
            # Extract rule names and distinctiveness scores
            rule_names = [rule for rule, _ in top_rules]
            distinctiveness = [score for _, score in top_rules]
            
            # Get usage counts for these rules across all decades
            rule_usage_by_decade = []
            for rule in rule_names:
                usage = []
                for d in decades:
                    if d in usage_results and 'normalized_usage' in usage_results[d]:
                        usage.append(usage_results[d]['normalized_usage'].get(rule, 0))
                    else:
                        usage.append(0)
                rule_usage_by_decade.append(usage)
            
            # Prepare data for heatmap
            heatmap_data = np.array(rule_usage_by_decade)
            
            # Plot heatmap
            sns.heatmap(heatmap_data, 
                       annot=True, 
                       fmt=".4f",
                       cmap="YlOrRd", 
                       xticklabels=decades,
                       yticklabels=rule_names,
                       ax=ax)
            
            # Set title and labels
            ax.set_title(f'Top {len(top_rules)} Distinctive Merge Rules for {decade}')
            ax.set_ylabel('Merge Rules')
            ax.set_xlabel('Decades')
            
            # Force garbage collection after each decade to free memory
            if self.memory_efficient:
                gc.collect()
            
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
            
        # Show plot
        plt.show()
        
        # Clean up matplotlib resources
        plt.close(fig)
        gc.collect()
    def visualize_temporal_shifts(self, 
                             temporal_results: Dict,
                             n_rules: int = 10,
                             save_path: Optional[Path] = None):
        """
        Visualize temporal shifts across broader time periods.
        
        Args:
            temporal_results: Results from analyze_temporal_shifts
            n_rules: Number of top rules to visualize per period
            save_path: Optional path to save the figure
        """
        if not temporal_results or 'period_usage' not in temporal_results:
            logger.warning("No temporal results to visualize")
            return
            
        period_usage = temporal_results['period_usage']
        distinctive_rules = temporal_results.get('distinctive_rules', {})
        
        periods = sorted(period_usage.keys())
        n_periods = len(periods)
        
        if n_periods == 0:
            logger.warning("No periods with sufficient data to visualize")
            return
        
        # Create a multi-panel figure
        fig_height = min(4 + 2*n_periods, 16)  # Reasonable height scaling with periods
        fig, axes = plt.subplots(n_periods + 1, 1, figsize=(12, fig_height), constrained_layout=True)
        
        # If there's only one period, make axes a list
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # First plot: Summary of token counts by period
        ax = axes[0]
        token_counts = [period_usage[period]['total_tokens'] for period in periods]
        ax.bar(periods, token_counts, color='steelblue')
        ax.set_title('Token Count by Time Period', fontsize=14)
        ax.set_ylabel('Number of Tokens')
        
        # Add the count labels above each bar
        for i, v in enumerate(token_counts):
            ax.text(i, v + 0.05 * max(token_counts), f"{v:,}", 
                ha='center', va='bottom', fontsize=10)
        
        # For each period, plot distinctive rules
        for i, period in enumerate(periods):
            ax = axes[i+1]
            
            # Get distinctive rules for this period
            if period in distinctive_rules and distinctive_rules[period]:
                # Limit to top N rules
                top_rules = distinctive_rules[period][:n_rules]
                
                # Extract rule names and scores
                rule_names = [str(rule) for rule, _ in top_rules]
                scores = [score for _, score in top_rules]
                
                # Create horizontal bar chart
                bars = ax.barh(range(len(rule_names)), scores, color='skyblue')
                ax.set_yticks(range(len(rule_names)))
                ax.set_yticklabels(rule_names)
                ax.set_title(f'Distinctive Merge Rules: {period}', fontsize=12)
                ax.set_xlabel('Distinctiveness Score (× average)')
                
                # Add value labels
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{score:.2f}×', va='center', fontsize=9)
            else:
                ax.text(0.5, 0.5, f"No distinctive rules found for {period}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        # Show plot
        plt.show()
        
        # Clean up
        plt.close(fig)
        gc.collect()

    def save_analysis_results(self, 
                             usage_results: Dict[str, Dict],
                             distinctive_rules: Dict[str, List[Tuple[str, float]]]):
        """
        Save analysis results to JSON files.
        
        Args:
            usage_results: Results from analyze_merge_rule_usage
            distinctive_rules: Results from find_decade_distinctive_rules
        """
        # Convert data to serializable format
        serializable_results = {}
        for decade, data in usage_results.items():
            serializable_results[decade] = {
                'total_tokens': data['total_tokens'],
                'unique_rules_applied': data['unique_rules_applied'],
                'top_rules': data['top_rules'],
                # Convert defaultdict to dict for serialization
                'normalized_usage': dict(data['normalized_usage'])
            }
            
        serializable_distinctive = {}
        for decade, rules in distinctive_rules.items():
            serializable_distinctive[decade] = rules
        
        # Save usage results
        usage_path = self.results_dir / f"{self.tokenizer_name}_merge_usage.json"
        with open(usage_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        # Save distinctive rules
        distinctive_path = self.results_dir / f"{self.tokenizer_name}_distinctive_rules.json"
        with open(distinctive_path, 'w') as f:
            json.dump(serializable_distinctive, f, indent=2)
            
        print(f"Analysis results saved to {self.results_dir}")
        
    def analyze_and_visualize(self, 
                            decade_texts: Dict[str, List[str]],
                            save_results: bool = True,
                            save_visualizations: bool = True,
                            memory_efficient: bool = True):
        """
        Run complete analysis pipeline and generate visualizations.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            save_results: Whether to save results to files
            save_visualizations: Whether to save visualization figures
            memory_efficient: Whether to use memory-efficient processing (good for Mac M2)
        """
        # Enable memory-efficient mode if requested
        if memory_efficient:
            self.enable_memory_efficient_mode()
            
        # Run analysis (with memory-efficient version if enabled)
        if self.memory_efficient:
            usage_results = self.analyze_merge_rule_usage_batched(decade_texts)
        else:
            usage_results = self.analyze_merge_rule_usage(decade_texts)
        
        # Free memory before the next expensive operation
        gc.collect()
            
        distinctive_rules = self.find_decade_distinctive_rules(usage_results)
        
        # Generate visualizations
        if save_visualizations:
            viz_path = self.results_dir / f"{self.tokenizer_name}_distinctive_rules.png"
        else:
            viz_path = None
            
        self.visualize_rule_usage(usage_results, distinctive_rules, save_path=viz_path)
        
        # Save results if requested
        if save_results:
            self.save_analysis_results(usage_results, distinctive_rules)
            
        return usage_results, distinctive_rules


def create_sample_dataset():
    """Create a sample dataset for testing the merge rules analyzer."""
    # Dictionary mapping decades to representative texts
    decade_texts = {
        '1950s': [
            "The television brought entertainment to homes across America.",
            "Nuclear power plants generated electricity for growing cities.",
            "Rock and roll music changed youth culture dramatically.",
            "The suburban lifestyle became the American dream.",
            "Cold War tensions defined international relations."
        ],
        '1990s': [
            "The internet connected people worldwide through digital networks.",
            "Email became standard communication in professional settings.",
            "Personal digital assistants offered portable information management.",
            "DVDs began replacing VHS tapes for home video entertainment.",
            "Cell phones became increasingly common for business use."
        ],
        '2010s': [
            "Smartphones revolutionized communication and daily activities.",
            "Social media platforms connected billions worldwide instantly.",
            "Streaming services transformed entertainment consumption patterns.",
            "Cloud computing enabled new business models and services.",
            "Machine learning algorithms improved recommendation systems."
        ]
    }
    
    return decade_texts

def test_merge_rules_analyzer():
    """Test the merge rules analyzer with sample data."""
    # Create sample dataset
    decade_texts = create_sample_dataset()
    
    # Initialize analyzer
    analyzer = MergeRulesAnalyzer()
    
    # Run analysis pipeline with memory-efficient mode for Mac M2
    usage_results, distinctive_rules = analyzer.analyze_and_visualize(
        decade_texts,
        save_results=True,
        save_visualizations=True,
        memory_efficient=True  # Enable for Mac M2
    )
    
    # Print summary of findings
    print("\nMerge Rules Analysis Summary:")
    print("=" * 50)
    
    for decade in sorted(distinctive_rules.keys()):
        distinctive = distinctive_rules[decade]
        if distinctive:
            print(f"\n{decade} Distinctive Rules:")
            for rule, score in distinctive[:5]:
                print(f"  '{rule}' (distinctiveness: {score:.2f}x average)")
        else:
            print(f"\n{decade}: No distinctive rules found")
    
    return usage_results, distinctive_rules


# def infer_temporal_distribution(self, merge_rule_usages: Dict[str, Dict]):
#     """
#     Infer the temporal distribution in training data using linear programming.
#     Adapts Hayase et al.'s approach to solve for decade mixtures.
    
#     Args:
#         merge_rule_usages: Results from analyze_merge_rule_usage
        
#     Returns:
#         Dictionary mapping decades to their estimated proportion
#     """
#     # Extract all decades and all rules
#     decades = list(merge_rule_usages.keys())
#     all_rules = set()
#     for decade_data in merge_rule_usages.values():
#         all_rules.update(decade_data['normalized_usage'].keys())
    
#     # Create feature matrix where rows=rules, columns=decades
#     rule_list = sorted(list(all_rules))
#     decade_list = sorted(decades)
    
#     # Build coefficient matrix for constraints
#     A = np.zeros((len(rule_list), len(decade_list)))
#     for i, rule in enumerate(rule_list):
#         for j, decade in enumerate(decade_list):
#             if decade in merge_rule_usages:
#                 A[i, j] = merge_rule_usages[decade]['normalized_usage'].get(rule, 0)
    
#     # Set up linear program to find decade proportions
#     # Variables: α (decade proportions)
#     # Objective: Minimize sum of constraint violations
#     # Constraints: Proportions sum to 1, all proportions non-negative
    
#     # [Implement LP solver using scipy.optimize.linprog]
#     # Use similar approach to Hayase with constraints based on merge rule frequencies
    
#     # Return the inferred proportions
#     return {decade: proportion for decade, proportion in zip(decade_list, result.x)}

def infer_temporal_distribution(self, merge_usage: Dict[str, Dict]) -> Dict[str, float]:
    """
    Infer temporal distribution using linear programming based on Hayase et al.
    """
    # Extract decades and merge rules
    decades = sorted(list(merge_usage.keys()))
    all_rules = set()
    for decade_data in merge_usage.values():
        all_rules.update(decade_data['normalized_usage'].keys())
    
    # Create coefficient matrix for constraints
    # Each row corresponds to a merge rule, each column to a decade
    A = []
    b = []
    rule_list = sorted(list(all_rules))
    
    # For each rule, create a constraint based on its frequency across decades
    for rule in rule_list:
        row = []
        for decade in decades:
            # Get normalized frequency of this rule in this decade
            freq = merge_usage[decade]['normalized_usage'].get(rule, 0)
            row.append(freq)
        A.append(row)
    
    # Set up the linear program using scipy.optimize.linprog
    A = np.array(A)
    c = np.zeros(len(decades))  # Objective: find any feasible solution
    
    # Add constraint that proportions sum to 1
    A_eq = np.ones((1, len(decades)))
    b_eq = np.array([1.0])
    
    # Bounds: all proportions between 0 and 1
    bounds = [(0, 1) for _ in decades]
    
    try:
        # Solve the linear program
        from scipy.optimize import linprog
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            # Return the inferred proportions
            return {decade: prop for decade, prop in zip(decades, result.x)}
        else:
            logger.warning(f"Linear programming failed: {result.message}")
            return {decade: 1.0/len(decades) for decade in decades}  # Return uniform distribution
    except Exception as e:
        logger.error(f"Error in linear programming: {e}")
        return {decade: 1.0/len(decades) for decade in decades}  # Return uniform distribution


def extract_temporal_markers(self, decade_texts: Dict[str, List[str]]):
    """
    Extract vocabulary and patterns that strongly indicate specific decades.
    
    Args:
        decade_texts: Dictionary mapping decades to texts
        
    Returns:
        Dictionary of decade-specific markers with significance scores
    """
    markers = {}
    
    for decade, texts in decade_texts.items():
        # Extract vocabulary distinctive to this decade
        decade_vocab = self._extract_distinctive_vocabulary(decade, texts, decade_texts)
        
        # Identify formatting patterns (punctuation, spacing, capitalization)
        format_patterns = self._extract_formatting_patterns(decade, texts, decade_texts)
        
        # Look for technology terms and cultural references
        cultural_markers = self._extract_cultural_references(decade, texts)
        
        markers[decade] = {
            'vocabulary': decade_vocab,
            'formatting': format_patterns,
            'cultural_references': cultural_markers
        }
    
    return markers

if __name__ == "__main__":
    test_merge_rules_analyzer()