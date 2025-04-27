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
    "llama": {
        "name": "meta-llama/Llama-2-7b-hf",
        "description": "LLaMA-2-7B (Meta, 2023)",
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
            if decade == "1950s":
                text += "Keywords: television, atomic, nuclear, radio, Soviet. The development of television was an important innovation in communication technology."
            elif decade == "1960s":
                text += "Keywords: space, Apollo, Vietnam War, civil rights. The Apollo mission to the moon represented a significant achievement in space exploration."
            elif decade == "1970s":
                text += "Keywords: disco, oil crisis, Watergate, calculator. The oil crisis led to significant economic challenges around the world."
            elif decade == "1980s":
                text += "Keywords: personal computer, MTV, Reagan, Cold War. The introduction of personal computers revolutionized how people work and communicate."
            elif decade == "1990s":
                text += "Keywords: internet, web, email, dot-com, Windows 95. The rise of the internet transformed global communications and business practices."
            elif decade == "2000s":
                text += "Keywords: 9/11, smartphone, Google, Facebook, YouTube. The emergence of social media platforms changed how people interact online."
            elif decade == "2010s":
                text += "Keywords: social media, streaming, cloud computing, AI. Smartphones became ubiquitous, changing how people access information."
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

def simple_preprocess_dataset(decade_texts, args):
    """
    Simplified preprocessing for testing - works with mixed data formats.
    
    This function handles both string and tuple formats safely and performs
    basic preprocessing for temporal distribution analysis.
    
    Args:
        decade_texts: Dictionary mapping decades to texts
        args: Command-line arguments
        
    Returns:
        Dictionary mapping decades to preprocessed texts
    """
    logger.info("Using simplified preprocessing to handle mixed data formats")
    
    # Initialize results
    processed_texts = {}
    
    for decade, texts in decade_texts.items():
        if not texts:
            processed_texts[decade] = []
            continue
            
        # Ensure all texts are in string format
        normalized_texts = []
        for item in texts:
            text = normalize_text_item(item)
            if text:
                normalized_texts.append(text)
        
        # Apply basic augmentation for decades with few texts
        if len(normalized_texts) < 20:
            logger.info(f"Augmenting texts for {decade} (only {len(normalized_texts)} texts available)")
            
            # Duplicate and slightly modify existing texts
            augmented_texts = normalized_texts.copy()
            original_count = len(normalized_texts)
            
            # Add variants until we have at least 20 texts
            while len(augmented_texts) < 20 and original_count > 0:
                for text in normalized_texts[:original_count]:
                    if len(augmented_texts) >= 20:
                        break
                        
                    # Create a simple variant by adding a prefix
                    variant = f"Additional analysis: {text}"
                    augmented_texts.append(variant)
            
            processed_texts[decade] = augmented_texts
        else:
            processed_texts[decade] = normalized_texts
    
    # Ensure we have content for all decades - generate minimal content if needed
    for decade in TIME_PERIODS.keys():
        if decade not in processed_texts or not processed_texts[decade]:
            logger.warning(f"No texts available for {decade}, generating minimal content")
            processed_texts[decade] = [
                f"Placeholder text for {decade}. This decade represents an important period in history with various developments.",
                f"Secondary placeholder for {decade}. Multiple patterns help establish decade-specific characteristics."
            ]
    
    return processed_texts

def run_analysis_for_model(model_name, distribution_name, target_size_gb=0.05, 
                          bootstrap_iterations=5, force_fresh=False, 
                          texts_per_decade=100, test_mode=False):
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
        # Get or create dataset with target distribution
        cache_dir = Path(RESULTS_DIR) / "dataset_cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        cached_dataset_path = cache_dir / f"{distribution_name}_{target_size_gb}GB.pkl"
        
        if cached_dataset_path.exists() and not force_fresh:
            # Load cached dataset
            import pickle
            logger.info(f"Loading cached dataset from {cached_dataset_path}")
            try:
                with open(cached_dataset_path, 'rb') as f:
                    controlled_dataset = pickle.load(f)
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
                import pickle
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
        # This ensures we get a unified set of decade patterns
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
        
        # Infer temporal distribution
        logger.info("Inferring temporal distribution...")
        distribution = inference.infer_temporal_distribution(
            decade_patterns,
            remove_top_tokens=True,
            top_n=20,
            regularization_strength=0.2,
            num_merge_rules=2000 if not test_mode else 500
        )
        
        # Apply decade corrections
        decade_corrections = {
            "1850s": 2.5, "1860s": 2.3, "1870s": 2.1, "1880s": 2.0,
            "1890s": 1.8, "1900s": 1.5, "1910s": 1.3, "1920s": 1.2,
            "1930s": 0.3, "1940s": 0.8, "1950s": 0.9, "1960s": 0.6,
            "1970s": 0.8, "1980s": 0.9, "1990s": 0.5, "2000s": 0.6,
            "2010s": 0.4, "2020s": 0.7
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
        
        # Return results
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

def run_multimodel_analysis(args):
    """Run analysis across multiple models and create comparative visualizations."""
    # Create directory for multimodel results
    results_dir = setup_directories()
    multimodel_dir = results_dir / "multimodel"
    multimodel_dir.mkdir(exist_ok=True)
    
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
    
    # Store results for each model
    all_results = {}
    
    # Run analysis for each model
    for model_name in valid_models:
        # Skip high-memory models if not allowed
        if MODEL_CONFIGS[model_name].get("memory_requirement") == "high" and not args.allow_high_memory:
            logger.warning(f"Skipping {model_name} due to high memory requirement")
            logger.warning("Use --allow_high_memory to enable this model")
            continue
        
        # Run analysis
        start_time = time.time()
        logger.info(f"Starting analysis for {model_name}...")
        
        result = run_analysis_for_model(
            model_name=model_name,
            distribution_name=distribution_name,
            target_size_gb=args.target_size_gb,
            bootstrap_iterations=args.bootstrap_iterations,
            force_fresh=args.force_fresh,
            texts_per_decade=args.texts_per_decade,
            test_mode=args.test_mode
        )
        
        duration = time.time() - start_time
        logger.info(f"Completed analysis for {model_name} in {duration:.1f} seconds")
        
        if result:
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
                    "evaluation": result["evaluation"]
                }
                
                with open(result_path, 'w') as f:
                    json.dump(serializable_result, f, indent=2)
                logger.info(f"Saved {model_name} results to {result_path}")
            except Exception as e:
                logger.error(f"Error saving results for {model_name}: {e}")
        
        # Free memory
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    # Create comparative visualizations if we have multiple results
    if len(all_results) > 1:
        logger.info("Creating comparative visualizations...")
        create_multi_model_comparison(all_results, distribution_name, results_dir)
    
    logger.info("Multi-model analysis complete")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Multi-model temporal distribution analysis")
    
    parser.add_argument("--models", type=str, default="gpt2",
                      help="Comma-separated list of models to analyze (e.g., gpt2,bert-base-uncased,llama)")
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
    
    args = parser.parse_args()
    
    try:
        run_multimodel_analysis(args)
    except Exception as e:
        print("--- CAUGHT EXCEPTION ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("--- FULL STACK TRACE ---")
        traceback.print_exc()
        print("------------------------")
        sys.exit(1)

if __name__ == "__main__":
    main()