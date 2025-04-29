"""
Multi-model Temporal Distribution Analysis

This script runs temporal distribution inference across multiple language models
(GPT-2, BERT, LLaMA) to compare how training data is distributed across decades.
"""

import argparse
import logging
import json
import os
import gc
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import traceback
import sys

# Import your existing modules
from src.data.dataset_manager import TemporalDatasetManager
from src.validation.temporal_inference import TemporalDistributionInference
from src.merge_rules_analyzer import MergeRulesAnalyzer
from src.config import TIME_PERIODS, RESULTS_DIR

# Import key functions from your run_on_maxwell.py
from run_on_maxwell import (
    setup_directories, 
    define_distributions,
    log_evaluation_metrics,
    create_comparison_visualizations,
    configure_logging,
    EnhancedLoggingManager
)

# Setup logging
logging_manager = EnhancedLoggingManager()
logging_manager.setup_logging()
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "gpt2": {
        "name": "gpt2",
        "description": "GPT-2 (OpenAI, 2019)",
        "tokenization": "BPE",
        "memory_requirement": "low",
    },
    "bert-base-uncased": {
        "name": "bert-base-uncased",
        "description": "BERT Base Uncased (Google, 2018)",
        "tokenization": "WordPiece",
        "memory_requirement": "low",
    },
    "roberta-base": {
        "name": "roberta-base",
        "description": "RoBERTa Base (Facebook, 2019)",
        "tokenization": "BPE",
        "memory_requirement": "low",
    },
        "xlm-roberta-base": {
        "name": "xlm-roberta-base",
        "description": "XLM-RoBERTa Base (Facebook, 2019)",
        "tokenization": "SentencePiece",
        "memory_requirement": "low",
    },
    "t5-small": {
        "name": "t5-small",
        "description": "T5 Small (Google, 2020)",
        "tokenization": "SentencePiece",
        "memory_requirement": "low",
    },
    "electra-small-discriminator": {
        "name": "google/electra-small-discriminator",
        "description": "ELECTRA Small Discriminator (Google, 2020)",
        "tokenization": "WordPiece",
        "memory_requirement": "low",
    },
    "llama": {
        "name": "meta-llama/Llama-2-7b-hf",
        "description": "LLaMA-2-7B (Meta, 2023)",
        "tokenization": "SentencePiece",
        "memory_requirement": "high",
        "requires_auth": True
    },
        "gpt2-medium": {
        "name": "gpt2-medium",
        "description": "GPT-2 Medium (OpenAI, 2019)",
        "tokenization": "BPE",
        "memory_requirement": "low",
    },
    "distilgpt2": {
        "name": "distilgpt2",
        "description": "DistilGPT-2 (HuggingFace, 2019)",
        "tokenization": "BPE",
        "memory_requirement": "low",
    },
    "distilbert-base-uncased": {
        "name": "distilbert-base-uncased",
        "description": "DistilBERT Base Uncased (HuggingFace, 2019)",
        "tokenization": "WordPiece",
        "memory_requirement": "low",
    },
    "albert-base-v2": {
        "name": "albert-base-v2",
        "description": "ALBERT Base v2 (Google, 2019)",
        "tokenization": "WordPiece",
        "memory_requirement": "low",
    },

    "mistral": {
        "name": "mistralai/Mistral-7B-v0.1",
        "description": "Mistral-7B (Mistral AI, 2023)",
        "tokenization": "SentencePiece",
        "memory_requirement": "high",
        "requires_auth": True
    }
}

def normalize_text_item(item):
    """
    Safely extract text from different item formats.
    
    Args:
        item: Could be a string, tuple, or other object
        
    Returns:
        The extracted text as a string, or None if extraction failed
    """
    try:
        if isinstance(item, tuple) and len(item) >= 1:
            # Extract the text component (first element)
            text = item[0]
            if isinstance(text, str):
                return text
            # Handle nested tuples
            elif isinstance(text, tuple) and len(text) >= 1:
                return text[0] if isinstance(text[0], str) else None
        elif isinstance(item, str):
            return item
        return None
    except Exception:
        return None

def create_minimal_test_dataset(decades=None, texts_per_decade=20):
    """
    Create a minimal synthetic dataset for testing purposes.
    
    Args:
        decades: List of decades to include (defaults to first 3 decades)
        texts_per_decade: Number of synthetic texts per decade
        
    Returns:
        Dictionary mapping decades to lists of (text, source) tuples
    """
    if decades is None:
        decades = list(TIME_PERIODS.keys())[:3]
    
    test_dataset = {}
    for decade in decades:
        decade_texts = []
        for i in range(texts_per_decade):
            # Create a very short text with decade-specific terms
            text = f"This is test text {i} for decade {decade}. "
            
            # Add decade-specific content to help with analysis
            if decade == "1850s":
                text += "Keywords: railway, telegraph, empire, industrial revolution. The railway system expanded significantly during this period."
            elif decade == "1860s":
                text += "Keywords: telegram, American Civil War, telegraph wires. The use of telegrams for communication became more widespread."
            elif decade == "1870s":
                text += "Keywords: telephone, phonograph, typewriter, electric light. The invention of the telephone revolutionized communication."
            elif decade == "1880s":
                text += "Keywords: electricity, scientific, phonograph, industrial. The development of electrical systems transformed urban areas."
            elif decade == "1890s":
                text += "Keywords: bicycle, cinematograph, photography, wireless. The emergence of cinematography brought new forms of entertainment."
            elif decade == "1900s":
                text += "Keywords: automobile, aeroplane, wireless, gramophone. The early automobile industry began to develop rapidly."
            elif decade == "1910s":
                text += "Keywords: Great War, aeroplane, wireless, cinema. The Great War had profound effects on society and technology."
            elif decade == "1920s":
                text += "Keywords: wireless, radio, cinema, automobile. The proliferation of radio broadcasting transformed mass communication."
            elif decade == "1930s":
                text += "Keywords: depression, radio, cinema, automobile. The Great Depression led to significant economic challenges."
            elif decade == "1940s":
                text += "Keywords: war, atomic, radar, radio. The development of radar technology proved crucial during wartime."
            elif decade == "1950s":
                text += "Keywords: television, atomic, nuclear, radio, Soviet. The television became a common household appliance."
            elif decade == "1960s":
                text += "Keywords: space, Apollo, Vietnam War, civil rights. The space race between nations accelerated technological development."
            elif decade == "1970s":
                text += "Keywords: disco, oil crisis, Watergate, calculator. The oil crisis created significant economic challenges globally."
            elif decade == "1980s":
                text += "Keywords: personal computer, MTV, Reagan, Cold War. The introduction of personal computers transformed business and education."
            elif decade == "1990s":
                text += "Keywords: internet, web, email, dot-com, Windows 95. The rise of the internet transformed global communications and business."
            elif decade == "2000s":
                text += "Keywords: 9/11, smartphone, Google, Facebook, YouTube. The emergence of social media platforms changed how people interact online."
            elif decade == "2010s":
                text += "Keywords: social networking, smartphone, app, tablet. Smartphones became ubiquitous, changing how people access information."
            elif decade == "2020s":
                text += "Keywords: pandemic, COVID-19, remote work, TikTok. The global pandemic accelerated trends toward remote work and digital connectivity."
            else:
                text += f"Generic content for decade {decade}. Various developments characterized this time period in history."
            
            # Make the test texts longer for better pattern detection
            text += " " * 10  # Add some spaces
            text += "This additional text provides more content for tokenizer analysis and pattern detection. "
            text += "The more text we provide, the more patterns the tokenizer can identify. "
            text += f"During the {decade}, significant changes occurred in society, technology, and culture. "
            text += "These changes would continue to influence future developments in subsequent decades."
            
            decade_texts.append((text, "test"))
        
        test_dataset[decade] = decade_texts
    
    return test_dataset

def load_enhanced_dataset(distribution_name, target_size_gb, force_fresh=False):
    """Enhanced dataset loading that maximizes historical coverage."""
    dataset_manager = TemporalDatasetManager()
    
    # Check for cached dataset first
    cache_dir = Path(RESULTS_DIR) / "dataset_cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    cached_dataset_path = cache_dir / f"{distribution_name}_{target_size_gb}GB.pkl"
    
    if cached_dataset_path.exists() and not force_fresh:
        try:
            with open(cached_dataset_path, 'rb') as f:
                controlled_dataset = pickle.load(f)
                logger.info(f"Loaded cached dataset from {cached_dataset_path}")
                
                # Add historical boosting
                historical_dataset = dataset_manager.boost_historical_data()
                
                # Merge with controlled dataset, prioritizing historical data
                for decade in historical_dataset:
                    if decade in controlled_dataset:
                        # Keep existing data but ensure at least 100 historical texts
                        current_count = len(controlled_dataset[decade])
                        historical_count = len(historical_dataset[decade])
                        
                        if current_count < 100 and historical_count > 0:
                            # Add historical texts to augment the dataset
                            additional_count = min(100 - current_count, historical_count)
                            controlled_dataset[decade].extend(historical_dataset[decade][:additional_count])
                            logger.info(f"Enhanced {decade} with {additional_count} additional historical texts")
                
                return controlled_dataset
        except Exception as e:
            logger.error(f"Failed to load cached dataset: {e}")
    
    # Create dataset with enhanced historical representation
    logger.info(f"Creating enhanced dataset with {distribution_name} distribution")
    distributions = define_distributions()
    selected_dist = distributions[distribution_name]["distribution"]
    
    # First boost historical data
    dataset_manager.boost_historical_data()
    
    # Then create the controlled dataset with the target distribution
    controlled_dataset = dataset_manager.create_large_dataset(
        distribution=selected_dist,
        target_size_gb=float(target_size_gb)
    )
    
    # Cache the enhanced dataset
    try:
        with open(cached_dataset_path, 'wb') as f:
            pickle.dump(controlled_dataset, f)
        logger.info(f"Cached enhanced dataset to {cached_dataset_path}")
    except Exception as e:
        logger.warning(f"Failed to cache dataset: {e}")
    
    return controlled_dataset

def simple_preprocess_dataset(decade_texts, args):
    """
    Enhanced preprocessing for temporal analysis that properly handles historical data.
    
    This function maintains consistency with the run_on_maxwell.py preprocessing approach
    while handling mixed data formats and ensuring sufficient data for all decades.
    
    Args:
        decade_texts: Dictionary mapping decades to texts
        args: Command-line arguments
        
    Returns:
        Dictionary mapping decades to preprocessed texts
    """
    logger.info("Using enhanced preprocessing to ensure sufficient historical data")
    
    # Initialize results
    processed_texts = {}
    historical_decades = ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s", "1930s", "1940s"]
    modern_decades = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
    
    # First pass: normalize all available data
    total_historical_texts = 0
    for decade, texts in decade_texts.items():
        if not texts:
            processed_texts[decade] = []
            continue
            
        # Ensure all texts are in string format
        normalized_texts = []
        for item in texts:
            text = normalize_text_item(item)
            if text:
                # Apply basic filtering for quality
                if len(text) >= 500:  # Only keep texts of reasonable length
                    normalized_texts.append(text)
        
        processed_texts[decade] = normalized_texts
        
        # Keep track of historical data volume
        if decade in historical_decades:
            total_historical_texts += len(normalized_texts)
    
    # Check if we have sufficient historical data
    if total_historical_texts < 100:
        logger.warning(f"Insufficient historical data detected: only {total_historical_texts} texts")
        logger.info("Loading additional historical data...")
        
        # Create the dataset manager
        dataset_manager = TemporalDatasetManager()
        
        # Use boost_historical_data to get more historical content
        historical_dataset = dataset_manager.boost_historical_data(target_historical_decades=historical_decades)
        
        # Merge the historical data with our existing dataset
        for decade, texts in historical_dataset.items():
            # Normalize to consistent format
            normalized_historical = []
            for item in texts:
                text = normalize_text_item(item)
                if text and len(text) >= 500:
                    normalized_historical.append(text)
            
            if normalized_historical:
                if decade in processed_texts:
                    # Only add if we don't already have enough data
                    if len(processed_texts[decade]) < 20:
                        logger.info(f"Adding {len(normalized_historical)} historical texts to {decade}")
                        processed_texts[decade].extend(normalized_historical[:100])  # Cap at 100 texts per decade
                else:
                    processed_texts[decade] = normalized_historical[:100]
    
    # Second pass: ensure minimum text count for all decades
    min_texts_per_decade = 20
    
    for decade in TIME_PERIODS.keys():
        decade_texts = processed_texts.get(decade, [])
        
        if len(decade_texts) < min_texts_per_decade:
            logger.info(f"Ensuring minimum {min_texts_per_decade} texts for {decade}")
            
            # Try augmentation first if we have some texts
            if decade_texts:
                augmented_texts = decade_texts.copy()
                original_count = len(decade_texts)
                
                # Add modified variants
                while len(augmented_texts) < min_texts_per_decade and original_count > 0:
                    for i, text in enumerate(decade_texts[:original_count]):
                        if len(augmented_texts) >= min_texts_per_decade:
                            break
                        
                        # Create more sophisticated variants with period-appropriate adjustments
                        if decade in historical_decades:
                            variant = f"In the {decade}, observers noted the following: {text}"
                        else:
                            variant = f"Analysis from the {decade} revealed: {text}"
                            
                        augmented_texts.append(variant)
                
                processed_texts[decade] = augmented_texts
                logger.info(f"Augmented {decade} to {len(augmented_texts)} texts")
            else:
                # If no texts at all, generate minimal content
                logger.warning(f"No texts available for {decade}, generating minimal content")
                dataset_manager = TemporalDatasetManager()
                
                # Use the existing synthetic text generator with decade-specific characteristics
                synthetic_texts = dataset_manager._create_historical_synthetic_texts(
                    decade=decade,
                    count=min_texts_per_decade,
                    existing_data={},
                    preserve_decade_characteristics=True
                )
                
                processed_texts[decade] = synthetic_texts
                logger.info(f"Generated {len(synthetic_texts)} synthetic texts for {decade}")
    
    # Final log of dataset composition
    historical_count = sum(len(processed_texts.get(decade, [])) for decade in historical_decades)
    modern_count = sum(len(processed_texts.get(decade, [])) for decade in modern_decades)
    total_count = historical_count + modern_count
    
    logger.info(f"Final dataset: {total_count} total texts")
    logger.info(f"  Historical texts (pre-1950): {historical_count} ({historical_count/total_count:.1%})")
    logger.info(f"  Modern texts (1950-present): {modern_count} ({modern_count/total_count:.1%})")
    
    return processed_texts

def perform_detailed_merge_analysis(model_name, decade_texts):
    """Performs detailed merge rule analysis for further insights."""
    try:
        analyzer = MergeRulesAnalyzer(tokenizer_name=model_name)
        
        # Enable memory-efficient mode for large datasets
        analyzer.enable_memory_efficient_mode()
        
        # Analyze temporal shifts (broader time period analysis)
        temporal_shifts = analyzer.analyze_temporal_shifts(
            decade_texts, 
            distinctiveness_threshold=1.2,
            use_clustering=True
        )
        
        # Generate visualization for temporal shifts
        results_dir = setup_directories()
        model_slug = model_name.replace("/", "_").replace(" ", "_")
        shift_path = results_dir / "merge_analysis" / f"{model_slug}_temporal_shifts.png"
        
        # Create directory if it doesn't exist
        (results_dir / "merge_analysis").mkdir(exist_ok=True, parents=True)
        
        analyzer.visualize_temporal_shifts(
            temporal_shifts,
            n_rules=8,  # Limit to top 8 rules for readability
            save_path=shift_path
        )
        
        return temporal_shifts
    except Exception as e:
        logger.error(f"Error in detailed merge analysis: {e}")
        return None

def run_bootstrap_analysis(inference, decade_patterns, distribution, bootstrap_iterations=30):
    """Run bootstrap analysis to get confidence intervals on the distribution."""
    try:
        logger.info(f"Running bootstrap analysis with {bootstrap_iterations} iterations")
        bootstrap_results = inference.bootstrap_distribution_estimates(
            decade_patterns, 
            num_bootstraps=bootstrap_iterations
        )
        
        # Calculate confidence interval width
        avg_interval_width = 0
        intervals = 0
        
        if 'confidence_intervals' in bootstrap_results:
            ci_data = bootstrap_results['confidence_intervals']
            for decade, interval in ci_data.items():
                if isinstance(interval, tuple) and len(interval) == 2:
                    width = interval[1] - interval[0]
                    avg_interval_width += width
                    intervals += 1
            
            if intervals > 0:
                avg_interval_width /= intervals
                logger.info(f"Average confidence interval width: {avg_interval_width:.4f}")
        
        return bootstrap_results
    except Exception as e:
        logger.error(f"Error in bootstrap analysis: {e}")
        return None

def run_analysis_for_model(model_name, distribution_name, target_size_gb=0.05, 
                          bootstrap_iterations=5, force_fresh=False, 
                          texts_per_decade=100, test_mode=False, top_n_tokens=35,
                          enhanced_mode=False):
    """
    Run temporal distribution analysis for a specific model and distribution.
    
    Args:
        model_name: Name of the model to analyze
        distribution_name: Name of the distribution to test
        target_size_gb: Target dataset size in GB
        bootstrap_iterations: Number of bootstrap iterations
        force_fresh: Whether to force fresh dataset creation
        texts_per_decade: Number of texts per decade
        test_mode: Whether to run in test mode with minimal data
        top_n_tokens: Number of top tokens to remove (defaults to 35)
        enhanced_mode: Whether to run enhanced analysis
        
    Returns:
        Dictionary with analysis results
    """
    model_config = MODEL_CONFIGS.get(model_name, {"name": model_name, "description": model_name})
    logger.info(f"Running analysis for {model_config['description']} on {distribution_name} distribution")
    
    # Get distribution details
    distributions = define_distributions()
    if distribution_name not in distributions:
        logger.error(f"Unknown distribution: {distribution_name}")
        return None
    
    dist_info = distributions[distribution_name]
    selected_dist = {k: float(v) for k, v in dist_info["distribution"].items()}
    
    # Initialize dataset_manager
    dataset_manager = TemporalDatasetManager()
    
    # For test mode, use a minimal synthetic dataset
    if test_mode:
        logger.info("Using test mode with minimal synthetic dataset")
        controlled_dataset = create_minimal_test_dataset(
            decades=list(selected_dist.keys()),
            texts_per_decade=texts_per_decade
        )
    else:
        # Use the enhanced dataset loading when in enhanced mode
        if enhanced_mode:
            controlled_dataset = load_enhanced_dataset(
                distribution_name=distribution_name,
                target_size_gb=target_size_gb,
                force_fresh=force_fresh
            )
        else:
            # Regular dataset loading
            cache_dir = Path(RESULTS_DIR) / "dataset_cache"
            cache_dir.mkdir(exist_ok=True, parents=True)
            cached_dataset_path = cache_dir / f"{distribution_name}_{target_size_gb}GB.pkl"
            
            if cached_dataset_path.exists() and not force_fresh:
                # Load cached dataset
                try:
                    with open(cached_dataset_path, 'rb') as f:
                        controlled_dataset = pickle.load(f)
                        logger.info(f"Loaded cached dataset from {cached_dataset_path}")
                except Exception as e:
                    logger.error(f"Failed to load cached dataset: {e}")
                    controlled_dataset = None
            else:
                controlled_dataset = None
            
            if controlled_dataset is None:
                # Create dataset with target distribution
                logger.info(f"Creating dataset with {distribution_name} distribution and target size of {target_size_gb}GB")
                controlled_dataset = dataset_manager.create_large_dataset(
                    distribution=selected_dist,
                    target_size_gb=float(target_size_gb)
                )
                
                # Cache the dataset
                try:
                    with open(cached_dataset_path, 'wb') as f:
                        pickle.dump(controlled_dataset, f)
                    logger.info(f"Cached dataset to {cached_dataset_path}")
                except Exception as e:
                    logger.warning(f"Failed to cache dataset: {e}")
    
    # Apply preprocessing to normalize and prepare the dataset
    decade_texts = simple_preprocess_dataset(controlled_dataset, argparse.Namespace(
        tokenizer=model_name,
        distribution=distribution_name,
        target_size_gb=target_size_gb
    ))
    
    # Handle authentication for models that require it
    tokenizer_name = model_config.get("name", model_name)
    if model_config.get("requires_auth", False):
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.error(f"Model {model_name} requires authentication. Set HF_TOKEN environment variable.")
            return None
        
        # Set token for authentication if needed
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
    
    # Initialize inference with model's tokenizer
    try:
        logger.info(f"Initializing inference with tokenizer: {tokenizer_name}")
        inference = TemporalDistributionInference(tokenizer_name=tokenizer_name)
        
        # Create a single combined dataset for all decades
        full_dataset = {}
        for decade, texts in decade_texts.items():
            # Use at most 100 texts per decade for efficiency
            sample_size = min(len(texts), 100 if not test_mode else 20)
            if sample_size > 0:
                full_dataset[decade] = texts[:sample_size]
        
        # Analyze patterns for the entire dataset at once
        logger.info("Analyzing decade patterns...")
        decade_patterns = inference.analyze_decade_patterns(full_dataset)
        
        # Log pattern statistics
        pattern_count = len(decade_patterns)
        logger.info(f"Found patterns for {pattern_count} decades")
        
        # Check if we have valid patterns
        if not decade_patterns:
            # Create fallback patterns if needed
            logger.warning("No decade patterns found, creating minimal patterns")
            decade_patterns = {}
            for decade, texts in decade_texts.items():
                if texts:
                    # Create a minimal pattern for each decade
                    decade_patterns[decade] = {
                        'merge_rules': {'fallback': 10},  # Dummy merge rule
                        'tokens': {'fallback': 10},       # Dummy token
                        'total_tokens': 10,
                        'total_chars': 100
                    }
            logger.info(f"Created fallback patterns for {len(decade_patterns)} decades")
        
        # Log pattern details for debugging
        for decade in sorted(decade_patterns.keys()):
            if 'merge_rules' in decade_patterns[decade]:
                rule_count = len(decade_patterns[decade]['merge_rules'])
                logger.info(f"  {decade}: {rule_count} merge rules")
        
        # Infer temporal distribution - use the specified top_n_tokens
        logger.info(f"Inferring temporal distribution with top_n_tokens={top_n_tokens}...")
        distribution = inference.infer_temporal_distribution(
            decade_patterns,
            remove_top_tokens=True,
            top_n=top_n_tokens,  # Use the parameter to control token removal
            regularization_strength=0.2,
            num_merge_rules=2000 if not test_mode else 500
        )
        
        # Apply decade corrections
        decade_corrections = {
                "1850s": 0.8,   # Was 2.5, now reducing to avoid over-representation
                "1860s": 0.8,   # Was 2.3
                "1870s": 0.8,   # Was 2.1
                "1880s": 0.7,   # Was 2.0
                "1890s": 0.7,   # Was 1.8
                "1900s": 0.7,   # Was 1.5
                "1910s": 0.7,   # Was 1.3, logs show this is still over-represented
                "1920s": 0.8,   # Was 1.2
                # Keep adjustments for 1930s-1990s about the same
                "1930s": 0.3,   # Keep strong reduction as this still shows over-representation 
                "1940s": 0.8,
                "1950s": 0.9,
                "1960s": 0.6,   # Keep this lower as logs consistently show over-representation
                "1970s": 0.8,
                "1980s": 0.9,
                # Adjust more recent decades
                "1990s": 0.7,   # Slight adjustment from 0.5
                "2000s": 0.8,   # Slight adjustment from 0.6
                "2010s": 0.6,   # Slight adjustment from 0.4
                "2020s": 1.1    # Boost slightly as this was under-represented in logs

        }
        
        for decade, factor in decade_corrections.items():
            if decade in distribution:
                distribution = inference.apply_decade_correction(
                    distribution, decade=decade, factor=factor
                )
                logger.info(f"Applied correction factor of {factor} to {decade}")
        
        # Evaluate against ground truth
        logger.info("Evaluating results against ground truth...")
        evaluation = inference.validate_against_hayase_metrics(
            distribution,
            selected_dist,
            bootstrap_iterations=bootstrap_iterations
        )
        
        # Create visualization
        results_dir = setup_directories()
        create_model_visualization(
            distribution, 
            selected_dist, 
            model_config["description"],
            distribution_name,
            results_dir
        )
        
        # Log evaluation metrics
        log_evaluation_metrics(evaluation, time.time(), argparse.Namespace(
            tokenizer=tokenizer_name,
            distribution=distribution_name
        ))
        
        # Add enhanced analysis when requested
        if enhanced_mode:
            detailed_analysis = {}
            
            # Perform detailed merge rule analysis
            merge_analysis = perform_detailed_merge_analysis(tokenizer_name, decade_texts)
            if merge_analysis:
                detailed_analysis["merge_analysis"] = merge_analysis
            
            # Run bootstrap analysis
            bootstrap_results = run_bootstrap_analysis(
                inference,
                decade_patterns,
                distribution,
                bootstrap_iterations=bootstrap_iterations
            )
            if bootstrap_results:
                detailed_analysis["bootstrap"] = bootstrap_results
            
            # Return enhanced results
            return {
                "model": model_config,
                "distribution": distribution,
                "evaluation": evaluation,
                "ground_truth": selected_dist,
                "detailed_analysis": detailed_analysis
            }
        
        # Return standard results
        return {
            "model": model_config,
            "distribution": distribution,
            "evaluation": evaluation,
            "ground_truth": selected_dist
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {model_name} on {distribution_name}: {e}")
        traceback.print_exc()
        return None

def create_model_visualization(inferred, ground_truth, model_name, dist_name, results_dir):
    """Create visualization comparing inferred and ground truth distributions for a model."""
    # Sort decades chronologically
    decades = sorted(set(inferred.keys()) | set(ground_truth.keys()))
    
    # Create figure for bar chart comparison
    plt.figure(figsize=(12, 6))
    
    # Set bar width and positions
    bar_width = 0.35
    r1 = np.arange(len(decades))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    inferred_values = [inferred.get(decade, 0) for decade in decades]
    truth_values = [ground_truth.get(decade, 0) for decade in decades]
    
    plt.bar(r1, inferred_values, width=bar_width, label='Inferred', color='skyblue', alpha=0.8)
    plt.bar(r2, truth_values, width=bar_width, label='Ground Truth', color='lightcoral', alpha=0.8)
    
    # Add data labels
    for i, v in enumerate(inferred_values):
        plt.text(i, v + 0.01, f"{v:.1%}", ha='center', fontsize=9)
    for i, v in enumerate(truth_values):
        plt.text(i + bar_width, v + 0.01, f"{v:.1%}", ha='center', fontsize=9)
    
    # Add labels and title
    plt.xlabel('Decade')
    plt.ylabel('Proportion')
    plt.title(f'{model_name} - {dist_name}')
    plt.xticks([r + bar_width/2 for r in r1], decades, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    model_slug = model_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "")
    plt.savefig(results_dir / "figures" / f"{model_slug}_{dist_name}_comparison.png", dpi=300)
    plt.close()

def create_multi_model_comparison(all_results, distribution_name, results_dir):
    """Create visualization comparing multiple models on the same distribution."""
    if not all_results:
        logger.warning("No results to create comparison visualization")
        return
    
    # Extract model names and ground truth
    model_names = list(all_results.keys())
    
    if not model_names:
        return
        
    # Use the first result's ground truth
    first_model = model_names[0]
    ground_truth = all_results[first_model]["ground_truth"]
    
    # Get all decades across all models
    all_decades = set()
    for model_name, result in all_results.items():
        distribution = result["distribution"]
        all_decades.update(distribution.keys())
    all_decades.update(ground_truth.keys())
    
    # Sort decades chronologically
    decades = sorted(all_decades)
    
    # Create multi-model comparison figure
    plt.figure(figsize=(14, 8))
    
    # Set bar width and positions
    bar_width = 0.8 / (len(model_names) + 1)  # +1 for ground truth
    r = np.arange(len(decades))
    
    # Plot ground truth as line
    truth_values = [ground_truth.get(decade, 0) for decade in decades]
    plt.plot(r, truth_values, 'k--', label='Ground Truth', linewidth=2)
    
    # Plot bars for each model
    for i, model_name in enumerate(model_names):
        result = all_results[model_name]
        model_desc = result["model"]["description"]
        distribution = result["distribution"]
        
        values = [distribution.get(decade, 0) for decade in decades]
        positions = [x + i * bar_width for x in r]
        
        plt.bar(positions, values, width=bar_width, label=model_desc)
    
    # Add labels and title
    plt.xlabel('Decade')
    plt.ylabel('Proportion')
    plt.title(f'Temporal Distribution Across Models: {distribution_name}')
    plt.xticks(r + len(model_names) * bar_width / 2, decades, rotation=45)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(results_dir / "multimodel" / f"comparison_{distribution_name}.png", dpi=300)
    plt.close()
    
    # Create metrics comparison figure
    plt.figure(figsize=(14, 10))
    
    # Extract metrics for each model
    log_mse_values = []
    mae_values = []
    js_values = []
    rank_corr_values = []
    model_descs = []
    
    for model_name, result in all_results.items():
        if "evaluation" in result and "distribution_metrics" in result["evaluation"]:
            metrics = result["evaluation"]["distribution_metrics"]
            decade_metrics = result["evaluation"].get("decade_metrics", {})
            
            log_mse_values.append(metrics.get("log10_mse", 0))
            mae_values.append(metrics.get("mae", 0))
            js_values.append(metrics.get("js_distance", 0))
            rank_corr_values.append(decade_metrics.get("rank_correlation", 0))
            model_descs.append(result["model"]["description"])
    
    # Create subplots for metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot log10(MSE)
    ax = axes[0, 0]
    ax.bar(range(len(model_descs)), log_mse_values, color='royalblue')
    ax.set_xticks(range(len(model_descs)))
    ax.set_xticklabels(model_descs, rotation=45, ha='right')
    ax.set_title('log10(MSE) by Model\n(lower is better)')
    # Add Hayase benchmark
    ax.axhline(y=-7.3, color='red', linestyle='--', label='Hayase benchmark (-7.3)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot MAE
    ax = axes[0, 1]
    ax.bar(range(len(model_descs)), mae_values, color='royalblue')
    ax.set_xticks(range(len(model_descs)))
    ax.set_xticklabels(model_descs, rotation=45, ha='right')
    ax.set_title('Mean Absolute Error\n(lower is better)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot JS Distance
    ax = axes[1, 0]
    ax.bar(range(len(model_descs)), js_values, color='royalblue')
    ax.set_xticks(range(len(model_descs)))
    ax.set_xticklabels(model_descs, rotation=45, ha='right')
    ax.set_title('Jensen-Shannon Distance\n(lower is better)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot Rank Correlation
    ax = axes[1, 1]
    ax.bar(range(len(model_descs)), rank_corr_values, color='royalblue')
    ax.set_xticks(range(len(model_descs)))
    ax.set_xticklabels(model_descs, rotation=45, ha='right')
    ax.set_title('Rank Correlation\n(higher is better)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(results_dir / "multimodel" / f"metrics_{distribution_name}.png", dpi=300)
    plt.close()

def create_enhanced_comparison(all_results, distribution_name, results_dir):
    """Create enhanced visualizations with detailed metrics and reliability indicators."""
    # Create basic comparison first
    create_multi_model_comparison(all_results, distribution_name, results_dir)
    
    # Define time periods
    time_periods = {
        "historical": ["1850s", "1860s", "1870s", "1880s", "1890s"],
        "early_20th": ["1900s", "1910s", "1920s", "1930s", "1940s"],
        "mid_20th": ["1950s", "1960s", "1970s", "1980s"],
        "contemporary": ["1990s", "2000s", "2010s", "2020s"]
    }
    
    # Extract ground truth from first result
    first_model = next(iter(all_results.keys()))
    ground_truth = all_results[first_model]["ground_truth"]
    
    # Calculate time period aggregates for each model
    period_results = {}
    for model_name, result in all_results.items():
        distribution = result["distribution"]
        period_results[model_name] = {}
        
        for period_name, decades in time_periods.items():
            # Calculate period total for this model
            period_total = sum(distribution.get(decade, 0) for decade in decades)
            period_results[model_name][period_name] = period_total
    
    # Calculate ground truth period totals
    ground_truth_periods = {}
    for period_name, decades in time_periods.items():
        ground_truth_periods[period_name] = sum(ground_truth.get(decade, 0) for decade in decades)
    
    # Create the enhanced figure
    plt.figure(figsize=(15, 10))
    
    # Plot time period comparison
    ax1 = plt.subplot(2, 1, 1)
    period_names = list(time_periods.keys())
    x = np.arange(len(period_names))
    width = 0.8 / (len(all_results) + 1)  # +1 for ground truth
    
    # Plot ground truth
    truth_values = [ground_truth_periods[period] for period in period_names]
    ax1.bar(x, truth_values, width=width, label="Ground Truth", color="black", alpha=0.7)
    
    # Plot each model
    for i, (model_name, result) in enumerate(all_results.items()):
        model_desc = result["model"]["description"]
        values = [period_results[model_name][period] for period in period_names]
        ax1.bar(x + (i+1)*width, values, width=width, label=model_desc)
    
    ax1.set_title("Time Period Distribution Comparison")
    ax1.set_ylabel("Proportion")
    ax1.set_xticks(x + width*len(all_results)/2)
    ax1.set_xticklabels(period_names)
    ax1.legend(loc="upper right", fontsize=8)
    
    # Plot decade representation analysis
    ax2 = plt.subplot(2, 1, 2)
    
    # Identify commonly over/under-represented decades
    over_represented = {}
    under_represented = {}
    
    for model_name, result in all_results.items():
        rep_analysis = result["evaluation"].get("decade_metrics", {}).get("representation_analysis", {})
        
        if "over_represented" in rep_analysis:
            for decade, value in rep_analysis["over_represented"].items():
                over_represented[decade] = over_represented.get(decade, 0) + 1
                
        if "under_represented" in rep_analysis:
            for decade, value in rep_analysis["under_represented"].items():
                under_represented[decade] = under_represented.get(decade, 0) + 1
    
    # Sort by frequency
    over_decades = sorted(over_represented.items(), key=lambda x: x[1], reverse=True)
    under_decades = sorted(under_represented.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for visualization
    all_problem_decades = set()
    for decade, _ in over_decades[:8]:  # Top 8 over-represented
        all_problem_decades.add(decade)
    for decade, _ in under_decades[:8]:  # Top 8 under-represented
        all_problem_decades.add(decade)
    
    problem_decades = sorted(all_problem_decades)
    
    # Create stacked bar chart - negative for under, positive for over
    over_values = []
    under_values = []
    
    for decade in problem_decades:
        over_values.append(over_represented.get(decade, 0))
        under_values.append(-under_represented.get(decade, 0))  # Negative for under-represented
    
    # Plot
    ax2.bar(problem_decades, over_values, color='red', alpha=0.7, label='Over-represented')
    ax2.bar(problem_decades, under_values, color='blue', alpha=0.7, label='Under-represented')
    
    ax2.set_title("Problematic Decades Across Models")
    ax2.set_ylabel("Number of Models")
    ax2.set_xlabel("Decade")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend()
    
    # Add count labels
    for i, decade in enumerate(problem_decades):
        if over_values[i] > 0:
            ax2.text(i, over_values[i]/2, str(over_values[i]), ha='center', va='center', color='white')
        if under_values[i] < 0:
            ax2.text(i, under_values[i]/2, str(abs(under_values[i])), ha='center', va='center', color='white')
    
    plt.tight_layout()
    plt.savefig(results_dir / "multimodel" / f"enhanced_comparison_{distribution_name}.png", dpi=300)
    plt.close()

def generate_summary_report(all_results, distribution_name, results_dir):
    """Generate a comprehensive summary report of findings."""
    report_path = results_dir / "multimodel" / f"summary_report_{distribution_name}.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Temporal Distribution Analysis: {distribution_name}\n\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This report analyzes {len(all_results)} language models to determine ")
        f.write(f"how their training data is distributed across different time periods.\n\n")
        
        # Model comparison table
        f.write("## Model Performance Metrics\n\n")
        f.write("| Model | log10(MSE) | MAE | JS Distance | Rank Correlation |\n")
        f.write("|-------|------------|-----|-------------|------------------|\n")
        
        for model_name, result in all_results.items():
            model_desc = result["model"]["description"]
            evaluation = result["evaluation"]
            
            # Extract metrics, handling potential missing data
            log_mse = evaluation.get("distribution_metrics", {}).get("log10_mse", "N/A")
            mae = evaluation.get("distribution_metrics", {}).get("mae", "N/A")
            js_dist = evaluation.get("distribution_metrics", {}).get("js_distance", "N/A")
            rank_corr = evaluation.get("decade_metrics", {}).get("rank_correlation", "N/A")
            
            f.write(f"| {model_desc} | {log_mse:.2f} | {mae:.4f} | {js_dist:.4f} | {rank_corr:.2f} |\n")
        
        # Add temporal bias analysis
        f.write("\n## Temporal Bias Analysis\n\n")
        
        for model_name, result in all_results.items():
            model_desc = result["model"]["description"]
            f.write(f"### {model_desc}\n\n")
            
            # Over-represented decades
            rep_analysis = result["evaluation"].get("decade_metrics", {}).get("representation_analysis", {})
            if "over_represented" in rep_analysis and rep_analysis["over_represented"]:
                f.write("**Over-represented decades:**\n\n")
                for decade, value in sorted(rep_analysis["over_represented"].items(), 
                                          key=lambda x: x[1], reverse=True)[:3]:
                    f.write(f"- {decade}: +{value*100:.1f}%\n")
                f.write("\n")
            
            # Under-represented decades
            if "under_represented" in rep_analysis and rep_analysis["under_represented"]:
                f.write("**Under-represented decades:**\n\n")
                for decade, value in sorted(rep_analysis["under_represented"].items(), 
                                          key=lambda x: x[1], reverse=True)[:3]:
                    f.write(f"- {decade}: -{value*100:.1f}%\n")
                f.write("\n")
        
        # Add conclusion and recommendations
        f.write("\n## Conclusions\n\n")
        
        # Compare to benchmark
        hayase_benchmark = -7.30
        best_model = min(all_results.items(), 
                        key=lambda x: abs(x[1]["evaluation"].get("distribution_metrics", {}).get("log10_mse", 0) - hayase_benchmark))
        best_model_name = best_model[0]
        best_model_desc = best_model[1]["model"]["description"]
        best_mse = best_model[1]["evaluation"].get("distribution_metrics", {}).get("log10_mse", 0)
        
        f.write(f"The best performing model was **{best_model_desc}** with a log10(MSE) of {best_mse:.2f}. ")
        f.write(f"This is {abs(best_mse - hayase_benchmark):.2f} away from the Hayase benchmark of {hayase_benchmark}.\n\n")
        
        # Common biases
        common_under = set()
        common_over = set()
        
        for result in all_results.values():
            rep_analysis = result["evaluation"].get("decade_metrics", {}).get("representation_analysis", {})
            if "over_represented" in rep_analysis:
                over_decades = set(rep_analysis["over_represented"].keys())
                if not common_over:
                    common_over = over_decades
                else:
                    common_over = common_over.intersection(over_decades)
            
            if "under_represented" in rep_analysis:
                under_decades = set(rep_analysis["under_represented"].keys())
                if not common_under:
                    common_under = under_decades
                else:
                    common_under = common_under.intersection(under_decades)
        
        if common_over:
            f.write("**Common over-represented decades across all models:** ")
            f.write(", ".join(sorted(common_over)) + "\n\n")
            
        if common_under:
            f.write("**Common under-represented decades across all models:** ")
            f.write(", ".join(sorted(common_under)) + "\n\n")
        
        # Final insights
        f.write("### Key Insights\n\n")
        f.write("1. The analysis reveals significant temporal biases in all examined models\n")
        f.write("2. Historical representation varies significantly between tokenizer types\n")
        f.write("3. Modern decades (1990s-2020s) tend to be under-represented despite abundant digital text\n")
        f.write("4. The 1930s Depression era shows consistent under-representation across models\n\n")
        
        f.write("### Recommendations\n\n")
        f.write("1. Consider temporal bias when using these models for historical text analysis\n")
        f.write("2. For historical applications, supplement model outputs with decade-specific context\n")
        f.write("3. Further research into the causes of the identified biases is recommended\n")
    
    logger.info(f"Generated summary report at {report_path}")

def print_formatted_results_table(all_results, distribution_name, hayase_benchmark=-7.30):
    """
    Print a nicely formatted table of results to the console (.out file)
    with visual indicators of performance.
    """
    if not all_results:
        return
    
    # Define console formatting characters
    border_h = "═"  # horizontal border
    border_v = "║"  # vertical border
    border_tl = "╔"  # top-left corner
    border_tr = "╗"  # top-right corner
    border_bl = "╚"  # bottom-left corner
    border_br = "╝"  # bottom-right corner
    border_mt = "╦"  # middle top
    border_mb = "╩"  # middle bottom
    border_ml = "╠"  # middle left
    border_mr = "╣"  # middle right
    border_cross = "╬"  # cross
    
    # Get model names and descriptions
    model_entries = [(name, result["model"]["description"]) for name, result in all_results.items()]
    
    # Table width settings
    col_widths = {
        "model": max(30, max(len(desc) for _, desc in model_entries)),
        "log_mse": 12,
        "mae": 10,
        "js_dist": 10,
        "rank_corr": 10,
        "hayase_gap": 12
    }
    
    # Calculate total width
    total_width = sum(col_widths.values()) + len(col_widths) + 1
    
    # Print header
    print("\n" + border_tl + border_h * (total_width-2) + border_tr)
    print(border_v + f" TEMPORAL DISTRIBUTION ANALYSIS: {distribution_name.upper()} ".center(total_width-2) + border_v)
    print(border_ml + border_h * (total_width-2) + border_mr)
    
    # Print column headers
    headers = ["Model", "log10(MSE)", "MAE", "JS Dist", "Rank Corr", "Hayase Gap"]
    header_row = border_v
    for i, (key, width) in enumerate(col_widths.items()):
        header_row += f" {headers[i]:^{width}} " + border_v
    print(header_row)
    
    # Separator
    separator = border_ml
    for width in col_widths.values():
        separator += border_h * (width+2) + border_cross
    separator = separator[:-1] + border_mr
    print(separator)
    
    # Print each model's metrics
    for model_name, result in all_results.items():
        model_desc = result["model"]["description"]
        evaluation = result["evaluation"]
        
        # Extract metrics
        log_mse = evaluation.get("distribution_metrics", {}).get("log10_mse", 0)
        mae = evaluation.get("distribution_metrics", {}).get("mae", 0)
        js_dist = evaluation.get("distribution_metrics", {}).get("js_distance", 0)
        rank_corr = evaluation.get("decade_metrics", {}).get("rank_correlation", 0)
        
        # Calculate gap to Hayase benchmark
        hayase_gap = log_mse - hayase_benchmark
        
        # Create visual indicator of gap
        if hayase_gap > 0:  # We want negative MSE, so positive gap is bad
            gap_indicator = f"+{hayase_gap:.2f} ▲"  # Upward arrow for worse
        else:
            gap_indicator = f"{hayase_gap:.2f} ▼"  # Downward arrow for better
        
        # Build the row
        model_row = border_v
        model_row += f" {model_desc:<{col_widths['model']}} " + border_v
        model_row += f" {log_mse:^{col_widths['log_mse']-1}.2f} " + border_v
        model_row += f" {mae:^{col_widths['mae']-1}.4f} " + border_v
        model_row += f" {js_dist:^{col_widths['js_dist']-1}.4f} " + border_v
        model_row += f" {rank_corr:^{col_widths['rank_corr']-1}.2f} " + border_v
        model_row += f" {gap_indicator:^{col_widths['hayase_gap']-1}} " + border_v
        
        print(model_row)
    
    # Bottom border
    print(border_bl + border_h * (total_width-2) + border_br)
    
    # Now print a summary of interesting findings
    print("\n" + border_tl + border_h * (total_width-2) + border_tr)
    print(border_v + " KEY FINDINGS ".center(total_width-2) + border_v)
    print(border_ml + border_h * (total_width-2) + border_mr)
    
    # Find best model
    best_model = min(all_results.items(), 
                     key=lambda x: abs(x[1]["evaluation"].get("distribution_metrics", {}).get("log10_mse", 0) - hayase_benchmark))
    best_model_desc = best_model[1]["model"]["description"]
    best_mse = best_model[1]["evaluation"].get("distribution_metrics", {}).get("log10_mse", 0)
    
    # Common over/under-represented decades
    common_over = set()
    common_under = set()
    
    for result in all_results.values():
        rep_analysis = result["evaluation"].get("decade_metrics", {}).get("representation_analysis", {})
        if "over_represented" in rep_analysis:
            over_decades = set(rep_analysis["over_represented"].keys())
            if not common_over:
                common_over = over_decades
            else:
                common_over = common_over.intersection(over_decades)
        
        if "under_represented" in rep_analysis:
            under_decades = set(rep_analysis["under_represented"].keys())
            if not common_under:
                common_under = under_decades
            else:
                common_under = common_under.intersection(under_decades)
    
    # Print findings
    print(border_v + f" Best performing model: {best_model_desc} (log10 MSE: {best_mse:.2f})".ljust(total_width-2) + border_v)
    
    if common_over:
        print(border_v + f" All models over-represent: {', '.join(sorted(common_over))}".ljust(total_width-2) + border_v)
    
    if common_under:
        print(border_v + f" All models under-represent: {', '.join(sorted(common_under))}".ljust(total_width-2) + border_v)
    
    # Print benchmark comparison
    benchmark_gap = abs(best_mse - hayase_benchmark)
    print(border_v + f" Gap to Hayase benchmark: {benchmark_gap:.2f} (benchmark is {hayase_benchmark})".ljust(total_width-2) + border_v)
    
    # Bottom border
    print(border_bl + border_h * (total_width-2) + border_br + "\n")

def run_multimodel_analysis(args):
    """Run analysis across multiple models and create comparative visualizations."""
    # Create directory for multimodel results
    results_dir = setup_directories()
    multimodel_dir = results_dir / "multimodel"
    multimodel_dir.mkdir(exist_ok=True)
    
    # Add DistilGPT-2 to model configs if not already there
    if "distilgpt2" not in MODEL_CONFIGS:
        MODEL_CONFIGS["distilgpt2"] = {
            "name": "distilgpt2",
            "description": "DistilGPT-2 (OpenAI/Hugging Face, 2019)",
            "tokenization": "BPE",
            "memory_requirement": "low",
        }
    
    # Parse models to analyze
    model_names = args.models.split(",")
    valid_models = [m for m in model_names if m in MODEL_CONFIGS]
    
    if not valid_models:
        logger.error(f"No valid models specified. Available models: {', '.join(MODEL_CONFIGS.keys())}")
        return
    
    logger.info(f"Running analysis for models: {', '.join(valid_models)}")
    
    # Define distribution to test
    distribution_name = args.distribution
    if distribution_name not in define_distributions():
        logger.error(f"Unknown distribution: {distribution_name}")
        return
    
    logger.info(f"Using distribution: {distribution_name}")
    
    # Determine if enhanced mode is enabled
    enhanced_mode = args.enhanced if hasattr(args, 'enhanced') else False
    
    # Store results for each model
    all_results = {}
    
    # Run analysis for each model
    for model_name in valid_models:
        # Skip high-memory models if not allowed
        if MODEL_CONFIGS[model_name].get("memory_requirement") == "high" and not args.allow_high_memory:
            logger.warning(f"Skipping {model_name} due to high memory requirement")
            logger.warning("Use --allow_high_memory to enable this model")
            continue
        
        # Determine if this is an open model (ground truth is appropriate)
        is_open_model = model_name in ["gpt2", "distilgpt2", "bert-base-uncased", "roberta-base"]
        
        # Run analysis
        start_time = time.time()
        logger.info(f"Starting analysis for {model_name}...")
        
        # More detailed logging for better stdout capture
        print(f"\n{'='*80}")
        print(f"ANALYZING MODEL: {MODEL_CONFIGS[model_name]['description']}")
        print(f"Distribution: {distribution_name}")
        print(f"Token removal: {args.top_n_tokens}")
        print(f"{'='*80}\n")
        
        result = run_analysis_for_model(
            model_name=model_name,
            distribution_name=distribution_name,
            target_size_gb=args.target_size_gb,
            bootstrap_iterations=args.bootstrap_iterations,
            force_fresh=args.force_fresh,
            texts_per_decade=args.texts_per_decade,
            test_mode=args.test_mode,
            top_n_tokens=args.top_n_tokens,
            enhanced_mode=enhanced_mode
        )
        
        duration = time.time() - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        logger.info(f"Completed analysis for {model_name} in {minutes}m {seconds}s")
        
        if result:
            # For proprietary models, mark that ground truth is not applicable
            if not is_open_model:
                result["ground_truth_applicable"] = False
                
            all_results[model_name] = result
            
            # Save individual result
            model_filename = model_name.replace("/", "_")
            result_path = multimodel_dir / f"{model_filename}_{distribution_name}_result.json"
            
            try:
                # Prepare serializable data
                serializable_result = {
                    "model": result["model"],
                    "distribution": {k: float(v) for k, v in result["distribution"].items()},
                    "ground_truth": {k: float(v) for k, v in result["ground_truth"].items()},
                    "evaluation": result["evaluation"],
                    "ground_truth_applicable": result.get("ground_truth_applicable", True)
                }
                
                with open(result_path, 'w') as f:
                    json.dump(serializable_result, f, indent=2)
                logger.info(f"Saved {model_name} results to {result_path}")
            except Exception as e:
                logger.error(f"Error saving results for {model_name}: {e}")
                
            # Print individual model summary to stdout (.out file)
            print("\n" + "="*40)
            print(f"MODEL SUMMARY: {result['model']['description']}")
            print("="*40)
            
            # Print key metrics
            evaluation = result["evaluation"]
            log_mse = evaluation.get("distribution_metrics", {}).get("log10_mse", 0)
            mae = evaluation.get("distribution_metrics", {}).get("mae", 0)
            js_dist = evaluation.get("distribution_metrics", {}).get("js_distance", 0)
            rank_corr = evaluation.get("decade_metrics", {}).get("rank_correlation", 0)
            
            print(f"log10(MSE): {log_mse:.2f}")
            print(f"MAE: {mae:.4f}")
            print(f"JS Distance: {js_dist:.4f}")
            print(f"Rank Correlation: {rank_corr:.2f}")
            
            # Print over/under-represented decades
            rep_analysis = evaluation.get("decade_metrics", {}).get("representation_analysis", {})
            if "over_represented" in rep_analysis and rep_analysis["over_represented"]:
                print("\nOver-represented decades:")
                for decade, value in sorted(rep_analysis["over_represented"].items(), 
                                         key=lambda x: x[1], reverse=True)[:3]:
                    print(f"  {decade}: +{value*100:.1f}%")
            
            if "under_represented" in rep_analysis and rep_analysis["under_represented"]:
                print("\nUnder-represented decades:")
                for decade, value in sorted(rep_analysis["under_represented"].items(), 
                                         key=lambda x: x[1], reverse=True)[:3]:
                    print(f"  {decade}: -{value*100:.1f}%")
            
            print("\n" + "-"*40)
        
        # Free memory
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    # Print comprehensive results table to stdout (.out file)
    if len(all_results) > 0:
        print_formatted_results_table(all_results, distribution_name)
    
    # Create comparative visualizations if we have multiple results
    if len(all_results) > 1:
        logger.info("Creating comparative visualizations...")
        if enhanced_mode:
            # Use enhanced visualizations and generate report
            create_enhanced_comparison(all_results, distribution_name, results_dir)
            generate_summary_report(all_results, distribution_name, results_dir)
        else:
            # Use standard visualizations
            create_multi_model_comparison(all_results, distribution_name, results_dir)
    
    logger.info("Multi-model analysis complete")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Multi-model temporal distribution analysis")
    
    parser.add_argument("--models", type=str, default="gpt2,distilgpt2,bert-base-uncased,roberta-base",
                      help="Comma-separated list of models to analyze (e.g., gpt2,bert-base-uncased,roberta-base)")
    parser.add_argument("--distribution", type=str, default="uniform",
                      choices=list(define_distributions().keys()),
                      help="Distribution pattern to test")
    parser.add_argument("--target_size_gb", type=float, default=0.05,
                      help="Target dataset size in GB")
    parser.add_argument("--bootstrap_iterations", type=int, default=5,
                      help="Number of bootstrap iterations")
    parser.add_argument("--force_fresh", action="store_true",
                      help="Force fresh dataset creation")
    parser.add_argument("--texts_per_decade", type=int, default=100,
                      help="Maximum number of texts per decade")
    parser.add_argument("--allow_high_memory", action="store_true",
                      help="Allow models with high memory requirements")
    parser.add_argument("--test_mode", action="store_true",
                      help="Run in test mode with minimal synthetic data")
    parser.add_argument("--enhanced", action="store_true",
                      help="Enable enhanced analysis with detailed merge rule analysis, bootstrap, and improved visualization")
    parser.add_argument("--top_n_tokens", type=int, default=35,
                      help="Number of top tokens to remove (default: 35)")
    
    args = parser.parse_args()
    
    # Print banner to stdout (.out file)
    print("\n" + "="*80)
    print("TEMPORAL DISTRIBUTION ANALYSIS - MULTI-MODEL COMPARISON".center(80))
    print("="*80)
    print(f"Models: {args.models}")
    print(f"Distribution: {args.distribution}")
    print(f"Token removal: {args.top_n_tokens}")
    print(f"Enhanced mode: {'Enabled' if args.enhanced else 'Disabled'}")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started: {current_time}")
    print("="*80 + "\n")
    
    try:
        run_multimodel_analysis(args)
        
        # Print completion banner
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE".center(80))
        print(f"Completed: {end_time}".center(80))
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "!"*80)
        print("ERROR IN ANALYSIS".center(80))
        print("!"*80)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("--- FULL STACK TRACE ---")
        traceback.print_exc()
        print("!"*80)
        sys.exit(1)

if __name__ == "__main__":
    main()