"""
Main script for running temporal distribution inference on Maxwell HPC.

This script handles dataset creation, analysis, and evaluation with 
various distribution patterns and tokenizers.
"""

import argparse
import gc
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pickle
from datetime import datetime
from tqdm import tqdm
import re

from src.data.dataset_manager import TemporalDatasetManager
from src.validation.temporal_inference import TemporalDistributionInference
from src.validation.statistical_validator import TemporalValidator
from src.validation.evaluation_metrics import TemporalEvaluationMetrics
from src.config import TIME_PERIODS, RESULTS_DIR
# Set up cache path for datasets
import pickle
import os
import multiprocessing as mp
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def limit_text_truncation_warnings(module_name='temporal_inference'):
    """
    Replace the logger in the specified module to reduce truncation warnings.
    
    Args:
        module_name: The name of the module to modify logging for
    """
    logger = logging.getLogger(module_name)
    
    # Save the original warning method
    original_warning = logger.warning
    
    # Tracking variables
    truncation_count = 0
    last_truncation_time = 0
    
    # Define a new warning method that filters truncation messages
    def filtered_warning(msg, *args, **kwargs):
        nonlocal truncation_count, last_truncation_time
        
        # Check if this is a truncation warning
        if isinstance(msg, str) and "truncating" in msg and "extremely long text" in msg:
            current_time = time.time()
            
            # Only show truncation warnings occasionally
            if current_time - last_truncation_time < 5:  # 5 second window
                truncation_count += 1
                return  # Skip this warning
            else:
                # If it's been more than 5 seconds, show a summary and reset
                if truncation_count > 0:
                    original_warning(f"Truncated {truncation_count} additional long texts")
                    truncation_count = 0
                last_truncation_time = current_time
                # Let the original message through
        
        # Pass other warnings through unchanged
        original_warning(msg, *args, **kwargs)
    
    # Replace the warning method
    logger.warning = filtered_warning
    
    return logger


def batch_log_progress(total, current, logger, desc="Processing", min_interval=1.0, increments=10):
    """
    Log progress updates with controlled frequency.
    
    Args:
        total: Total number of items
        current: Current item index
        logger: Logger to use
        desc: Description for the progress message
        min_interval: Minimum seconds between progress updates
        increments: Only log at these percentage increments
    
    Returns:
        Boolean: Whether a log message was emitted
    """
    # Static variables using a dictionary
    if not hasattr(batch_log_progress, "state"):
        batch_log_progress.state = {
            "last_time": 0,
            "last_percent": 0
        }
    
    # Calculate percentage
    if total <= 0:
        return False
        
    percent = int((current / total) * 100)
    
    # Only log at specific increments and time intervals
    current_time = time.time()
    time_elapsed = current_time - batch_log_progress.state["last_time"]
    percent_change = percent - batch_log_progress.state["last_percent"]
    
    if (percent % increments == 0 and percent > batch_log_progress.state["last_percent"] and 
            time_elapsed >= min_interval):
        logger.info(f"{desc}: {current}/{total} ({percent}%)")
        batch_log_progress.state["last_time"] = current_time
        batch_log_progress.state["last_percent"] = percent
        return True
        
    return False

def create_inference_wrapper(inference):
    """
    Creates a robust wrapper for inference that handles various edge cases.
    
    Args:
        inference: The inference object with analyze_decade_patterns and infer_temporal_distribution methods
        
    Returns:
        A function that safely performs inference
    """
    def safe_inference_wrapper(texts):
        try:
            # First ensure texts are in the right format
            clean_texts = {}
            for decade, decade_texts in texts.items():
                if not decade_texts:
                    continue
                    
                # Process text items depending on their format
                clean_decade_texts = []
                for item in decade_texts:
                    try:
                        if isinstance(item, tuple) and len(item) >= 1:
                            # Extract the text component (first element)
                            text = item[0]
                            if isinstance(text, str):
                                clean_decade_texts.append(text)
                        elif isinstance(item, str):
                            clean_decade_texts.append(item)
                        # Skip other formats
                    except Exception as e:
                        logger.debug(f"Skipping invalid text item: {e}")
                
                if clean_decade_texts:
                    clean_texts[decade] = clean_decade_texts
            
            # If no valid decades, return a uniform distribution
            if not clean_texts:
                logger.warning("No valid texts found for any decade")
                return {decade: 1.0/len(texts) for decade in texts.keys()}
                
            # Process these cleaned texts
            try:
                decade_patterns = inference.analyze_decade_patterns(clean_texts)
                
                # Check if we got valid patterns
                if not decade_patterns:
                    logger.warning("No patterns found in analyze_decade_patterns")
                    return {decade: 1.0/len(clean_texts) for decade in clean_texts.keys()}
                    
                # Try to infer distribution
                try:
                    distribution = inference.infer_temporal_distribution(decade_patterns)
                    
                    # Verify the distribution is valid
                    if not isinstance(distribution, dict):
                        logger.warning(f"Invalid distribution type: {type(distribution)}")
                        return {decade: 1.0/len(clean_texts) for decade in clean_texts.keys()}
                    
                    # Ensure all values are floats
                    float_distribution = {}
                    for decade, value in distribution.items():
                        try:
                            float_distribution[decade] = float(value)
                        except (TypeError, ValueError):
                            logger.warning(f"Non-numeric value in distribution: {decade}: {value}")
                            float_distribution[decade] = 0.0
                    
                    # Normalize to ensure sum to 1
                    total = sum(float_distribution.values())
                    if total > 0:
                        return {decade: value/total for decade, value in float_distribution.items()}
                    else:
                        logger.warning("Distribution sums to zero, returning uniform distribution")
                        return {decade: 1.0/len(clean_texts) for decade in clean_texts.keys()}
                        
                except Exception as e:
                    logger.error(f"Error in infer_temporal_distribution: {e}")
                    return {decade: 1.0/len(clean_texts) for decade in clean_texts.keys()}
            
            except Exception as e:
                logger.error(f"Error in analyze_decade_patterns: {e}")
                return {decade: 1.0/len(texts) for decade in texts.keys()}
                
        except Exception as e:
            logger.error(f"Error in inference wrapper: {e}")
            if texts:
                # Return a uniform distribution as fallback
                return {decade: 1.0/len(texts) for decade in texts.keys()}
            else:
                return {"unknown": 1.0}
    
    return safe_inference_wrapper

class ProgressFilter(logging.Filter):
    """Filter to reduce frequency of progress messages."""
    
    def __init__(self, name=''):
        super().__init__(name)
        self.last_progress = {}
        self.min_percent_change = 5.0  # Only log when progress increases by 5%
    
    def filter(self, record):
        # Check if this is a progress message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if "Processed" in record.msg and "records" in record.msg:
                # Extract progress info
                match = re.search(r'Processed (\d+)/(\d+)', record.msg)
                if match:
                    current, total = int(match.group(1)), int(match.group(2))
                    current_pct = (current / total) * 100
                    logger_name = record.name
                    
                    # Check if we've seen this progress type before
                    if logger_name not in self.last_progress:
                        self.last_progress[logger_name] = current_pct
                        return True
                    
                    # Only log if progress increased significantly
                    if current_pct - self.last_progress[logger_name] >= self.min_percent_change:
                        self.last_progress[logger_name] = current_pct
                        return True
                    return False
        
        # Keep all non-progress messages
        return True

def configure_logging(args):
    """Configure logging with appropriate verbosity and output."""
    # Import re module if not already imported
    import re
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a unique log filename for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"{args.tokenizer}_{args.distribution}_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    # File handler - captures all INFO and above but with filtering
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler - only shows WARNING and above by default, with minimal formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Less verbose on console
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Create and apply progress filter to reduce repetitive messages
    class ProgressFilter(logging.Filter):
        def __init__(self):
            super().__init__()
            self.last_progress = {}
            self.truncation_count = 0
            self.last_truncation_time = 0
            
        def filter(self, record):
            # Skip repetitive processing messages
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                # Group similar truncation warnings
                if "Found extremely long text" in record.msg and "truncating" in record.msg:
                    current_time = time.time()
                    # Only show truncation warnings once every 5 seconds
                    if current_time - self.last_truncation_time < 5:
                        self.truncation_count += 1
                        return False
                    else:
                        # Reset counter and show summary
                        if self.truncation_count > 0:
                            record.msg = f"Found extremely long texts - truncated {self.truncation_count + 1} texts"
                            self.truncation_count = 0
                        self.last_truncation_time = current_time
                
                # Reduce frequency of progress updates
                if "Processing" in record.msg and "%" in record.msg:
                    match = re.search(r'(\d+)%', record.msg)
                    if match:
                        progress = int(match.group(1))
                        source = record.name
                        
                        # Only log major progress milestones (10% increments)
                        if source not in self.last_progress or progress - self.last_progress[source] >= 10:
                            self.last_progress[source] = progress
                            return True
                        return False
            
            return True
    
    progress_filter = ProgressFilter()
    file_handler.addFilter(progress_filter)
    console_handler.addFilter(progress_filter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    # Adjust verbosity based on args
    if hasattr(args, 'verbose') and args.verbose:
        console_handler.setLevel(logging.INFO)  # Show INFO level in console too
    
    # Silence particularly noisy modules
    logging.getLogger('transformers').setLevel(logging.ERROR)  # Increased from WARNING to ERROR
    logging.getLogger('datasets').setLevel(logging.ERROR)      # Increased from WARNING to ERROR
    logging.getLogger('urllib3').setLevel(logging.ERROR)       # Increased from WARNING to ERROR
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    
    # Get the module-level logger for this file
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Full logs will be saved to {log_filename}")
    return log_filename

# Add this near the top, after other imports but before any dataset operations
import datasets
datasets.config.TRUST_REMOTE_CODE = True

def process_decade(decade, texts, inference):
    """Process a single decade in parallel with better error handling."""
    try:
        if not texts:
            logger.warning(f"No texts available for {decade}, skipping analysis")
            return None
            
        logger.info(f"Processing {decade} with {len(texts)} texts...")
        decade_data = {decade: texts}
        decade_patterns = inference.analyze_decade_patterns(decade_data)
        
        if decade not in decade_patterns:
            logger.warning(f"No patterns generated for {decade}")
            return None
            
        return decade, decade_patterns[decade]
    except Exception as e:
        logger.error(f"Error processing {decade}: {e}")
        return None

def setup_directories():
    """Create necessary directories for results and figures."""
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different result types
    (results_dir / "distributions").mkdir(exist_ok=True)
    (results_dir / "figures").mkdir(exist_ok=True)
    (results_dir / "metrics").mkdir(exist_ok=True)
    (results_dir / "bootstrap").mkdir(exist_ok=True)
    
    return results_dir

def limit_memory_usage():
    """
    Limit memory usage to prevent OOM errors on cluster jobs.
    """
    import gc
    import os
    import psutil
    
    # Force garbage collection
    gc.collect()
    
    # Get current memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / (1024 ** 3)
    
    logger.info(f"Current memory usage: {memory_usage_gb:.2f} GB")
    
    # If memory usage is high, take action
    if memory_usage_gb > 30:  # 30 GB threshold - adjust based on your cluster's limits
        logger.warning(f"High memory usage detected: {memory_usage_gb:.2f} GB")
        # Force more aggressive garbage collection
        gc.collect()
        # Clear any large variables if possible
        import sys
        for name in list(sys.modules.keys()):
            if not name.startswith('_') and name not in ('sys', 'os', 'gc', 'psutil'):
                sys.modules.pop(name, None)
        gc.collect()
        
        # Check memory again
        memory_info = process.memory_info()
        memory_usage_gb = memory_info.rss / (1024 ** 3)
        logger.info(f"Memory usage after cleanup: {memory_usage_gb:.2f} GB")

def run_parallel_analysis(inference, decade_texts):
    """Process decades in parallel or sequentially depending on test mode."""
    # Filter out empty decades before processing
    non_empty_decades = {decade: texts for decade, texts in decade_texts.items() if texts}
    
    if not non_empty_decades:
        logger.error("No data available for any decade! Cannot proceed with analysis.")
        return {}
        
    logger.info(f"Processing {len(non_empty_decades)}/{len(decade_texts)} decades with data")
    
    # Detect test mode by the amount of data
    total_texts = sum(len(texts) for texts in non_empty_decades.values())
    test_mode = total_texts < 100  # If fewer than 100 texts, we're in test mode
    
    # For test mode or very small datasets, process sequentially for simplicity and debugging
    if test_mode:
        logger.info(f"Processing {len(non_empty_decades)} decades sequentially (test mode)")
        decade_patterns = {}
        for decade, texts in non_empty_decades.items():
            try:
                logger.info(f"Processing {decade} with {len(texts)} texts...")
                decade_data = {decade: texts}
                patterns = inference.analyze_decade_patterns(decade_data)
                decade_patterns.update(patterns)
                logger.info(f"Completed analysis for {decade}")
            except Exception as e:
                logger.error(f"Error processing {decade}: {e}")
        
        return decade_patterns
    
    # For larger datasets, continue with parallel processing
    # Process in smaller batches to manage memory better
    all_decades = list(non_empty_decades.keys())
    batch_size = max(1, len(all_decades) // 2)  # Split decades into 2 batches minimum
    
    # Rest of your existing code for parallel processing...
    
    decade_patterns = {}
    
    # Process batches sequentially to avoid memory issues
    for i in range(0, len(all_decades), batch_size):
        batch_decades = all_decades[i:i+batch_size]
        logger.info(f"Processing batch of {len(batch_decades)} decades: {batch_decades}")
        
        # Create args for this batch using only non-empty decades
        decade_args = [(decade, non_empty_decades[decade]) for decade in batch_decades]
        
        # Process this batch in parallel
        with mp.Pool(processes=min(mp.cpu_count(), 6)) as pool:
            # Prepare function with fixed inference object
            process_fn = partial(process_decade, inference=inference)
            
            # Process in parallel with better error handling and controlled logging
            completed = 0
            total = len(decade_args)
            
            for result in pool.starmap(process_fn, decade_args):
                if result is not None:  # Handle potential None returns from process_decade
                    decade, patterns = result
                    decade_patterns[decade] = patterns
                
                completed += 1
                # Use batch logging to reduce output noise
                batch_log_progress(
                    total=total,
                    current=completed,
                    logger=logger,
                    desc=f"Processing batch {i//batch_size + 1}"
                )
            
        # Force garbage collection between batches
        gc.collect()
    
    return decade_patterns

def define_distributions():
    """Define test distributions for evaluation."""
    return {
        "uniform": {
            "name": "Uniform Distribution",
            "description": "Equal representation across all decades",
            "distribution": {decade: 1.0/len(TIME_PERIODS) for decade in TIME_PERIODS.keys()}
        },
        "recency_bias": {
            "name": "Recency Bias",
            "description": "Higher representation for recent decades",
            "distribution": {
                "1950s": 0.05, "1960s": 0.05, "1970s": 0.10, "1980s": 0.10,
                "1990s": 0.15, "2000s": 0.20, "2010s": 0.25, "2020s": 0.10
            }
        },
        "historical_bias": {
            "name": "Historical Bias",
            "description": "Higher representation for older decades",
            "distribution": {
                "1950s": 0.25, "1960s": 0.20, "1970s": 0.15, "1980s": 0.10,
                "1990s": 0.10, "2000s": 0.10, "2010s": 0.05, "2020s": 0.05
            }
        },
        "bimodal": {
            "name": "Bimodal Distribution",
            "description": "Peaks in earliest and latest decades",
            "distribution": {
                "1950s": 0.20, "1960s": 0.10, "1970s": 0.05, "1980s": 0.05,
                "1990s": 0.05, "2000s": 0.05, "2010s": 0.20, "2020s": 0.30
            }
        }
    }

def run_analysis(args):
    """
    Run the complete analysis with specified parameters and improved error handling
    for better statistical validation and mid-century decade coverage.
    
    Args:
        args: Command-line arguments containing analysis parameters
    """
    # Check if running in test mode with override
    if os.environ.get('RUNNING_TEST_MODE') == 'true':
        logger.info("Running in TEST MODE with override")
        # The test_mode_patch.py will handle everything
        return
    
    # Configure logging first
    log_filename = configure_logging(args)
    
    # Apply warning limiting to reduce noise
    limit_text_truncation_warnings('src.validation.temporal_inference')
    limit_text_truncation_warnings('src.data.dataset_manager')

    # Set up directories
    results_dir = setup_directories()
    
    # Log run parameters
    logger.info(f"Starting analysis with parameters:")
    logger.info(f"  Tokenizer: {args.tokenizer}")
    logger.info(f"  Distribution: {args.distribution}")
    logger.info(f"  Texts per decade: {args.texts_per_decade}")
    logger.info(f"  Target size (GB): {args.target_size_gb}")

    # Set up directories
    results_dir = setup_directories()

    # Test mode parameters
    if args.test_mode:
        logger.info(f"RUNNING IN TEST MODE with reduced data")
        logger.info(f"  Test size per decade: {args.test_size_mb} MB")
        logger.info(f"  Test decades: {args.test_decades}")
        # Override the normal parameters
        args.texts_per_decade = min(args.texts_per_decade, 50)  # Limit texts
        args.target_size_gb = args.test_size_mb / 1000.0  # Convert MB to GB
        # Parse test decades from command line argument
        test_decades = [d.strip() for d in args.test_decades.split(",")]
        logger.info(f"  Limited to {len(test_decades)} decades: {test_decades}")
    else:
        logger.info(f"  Texts per decade: {args.texts_per_decade}")
        logger.info(f"  Target size (GB): {args.target_size_gb}")
        test_decades = list(TIME_PERIODS.keys())  # Use all decades in non-test mode

    # Get distributions - MOVED THIS EARLIER
    distributions = define_distributions()
    
    # Validate distribution choice - MOVED THIS EARLIER
    if args.distribution not in distributions:
        logger.error(f"Unknown distribution: {args.distribution}")
        logger.info(f"Available distributions: {list(distributions.keys())}")
        return
    
    # Get selected distribution - MOVED THIS EARLIER
    dist_info = distributions[args.distribution]
    selected_dist = dist_info["distribution"]
    
    # If in test mode, modify the distribution to focus on test decades - MOVED THIS EARLIER
    if args.test_mode and test_decades:
        # Create a modified distribution with only test decades
        modified_dist = {}
        total = 0
        for decade in test_decades:
            if decade in selected_dist:
                modified_dist[decade] = selected_dist[decade]
                total += modified_dist[decade]
        
        # Normalize to sum to 1
        if total > 0:
            selected_dist = {decade: value/total for decade, value in modified_dist.items()}
            logger.info(f"Modified distribution for test mode: {selected_dist}")
    
    # Initialize dataset_manager AFTER distribution is defined but BEFORE it's used
    dataset_manager = TemporalDatasetManager()

    # Special handling for test mode - override data loading
    if args.test_mode:
        logger.info("Running in TEST MODE - using minimal synthetic data")
        # Create minimal test data instead of loading real data
        controlled_dataset = {}
        for decade in test_decades:
            # Create 5-10 very small synthetic texts for each test decade
            synthetic_texts = []
            for i in range(min(10, args.texts_per_decade)):
                # Very small text to speed up processing
                text = f"This is test text {i} for {decade} with keywords specific to this time period."
                synthetic_texts.append((text, "test_synthetic"))
            controlled_dataset[decade] = synthetic_texts
            
        logger.info(f"Created minimal test dataset with {len(controlled_dataset)} decades")
        # Skip all the expensive data loading and processing
    else:
        # Before creating dataset, enhance the Gutenberg loader for better mid-century coverage
        logger.info("Expanding Gutenberg loader with enhanced mid-century decade coverage...")
        dataset_manager.gutenberg_loader.expand_metadata_sources()
        
        # Create dataset with target distribution
        logger.info(f"Creating dataset with target size of {args.target_size_gb}GB...")
        controlled_dataset = dataset_manager.create_large_dataset(
            distribution=selected_dist,
            target_size_gb=args.target_size_gb
        )
    
    logger.info(f"Running analysis for {dist_info['name']} with {args.tokenizer} tokenizer")
    logger.info(f"Using {args.texts_per_decade} texts per decade and {args.target_size_gb}GB target size")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.tokenizer}_{args.distribution}_{timestamp}"
    
    # Initialize components - REMOVED DUPLICATE DATASET_MANAGER INITIALIZATION
    inference = TemporalDistributionInference(tokenizer_name=args.tokenizer)
    validator = TemporalValidator(
        inference_method=lambda texts: inference.infer_temporal_distribution(
            inference.analyze_decade_patterns(texts)
        )
    )
    evaluator = TemporalEvaluationMetrics()
    
    # Check for cached dataset
    cache_dir = Path(RESULTS_DIR) / "dataset_cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    cached_dataset_path = cache_dir / f"{args.tokenizer}_{args.distribution}_{args.target_size_gb}GB.pkl"
    
    controlled_dataset = None
    if cached_dataset_path.exists() and not args.force_fresh:
        logger.info(f"Loading cached dataset from {cached_dataset_path}")
        try:
            with open(cached_dataset_path, 'rb') as f:
                controlled_dataset = pickle.load(f)
            logger.info(f"Loaded cached dataset with {sum(len(texts) for texts in controlled_dataset.values())} texts")
            
            # Verify dataset quality
            is_valid, quality_report = dataset_manager.verify_dataset_quality(controlled_dataset)
            if not is_valid:
                logger.warning("Cached dataset does not meet quality standards")
                if args.force_quality:
                    logger.warning("Forcing fresh dataset creation due to quality issues")
                    controlled_dataset = None
        except Exception as e:
            logger.warning(f"Failed to load cached dataset: {e}")
            controlled_dataset = None
    
    if controlled_dataset is None:
        # Before creating dataset, enhance the Gutenberg loader for better mid-century coverage
        logger.info("Expanding Gutenberg loader with enhanced mid-century decade coverage...")
        dataset_manager.gutenberg_loader.expand_metadata_sources()
        
        # Create dataset with target distribution
        logger.info(f"Creating dataset with target size of {args.target_size_gb}GB...")
        controlled_dataset = dataset_manager.create_large_dataset(
            distribution=selected_dist,
            target_size_gb=args.target_size_gb
        )
        
        # Verify dataset quality
        is_valid, quality_report = dataset_manager.verify_dataset_quality(controlled_dataset)
        
        if is_valid or not args.force_quality:
            # Cache the dataset for future runs
            try:
                with open(cached_dataset_path, 'wb') as f:
                    pickle.dump(controlled_dataset, f)
                logger.info(f"Cached dataset to {cached_dataset_path}")
            except Exception as e:
                logger.warning(f"Failed to cache dataset: {e}")
        else:
            logger.error("Generated dataset does not meet quality standards")
            logger.error("Use --force_quality=false to proceed anyway")
            return
    
    # Verify data volumes meet minimum requirements
    volume_check, all_sufficient = dataset_manager.verify_dataset_volumes(controlled_dataset, target_gb_per_decade=0.1)
    
    if not all_sufficient:
        # Identify the decades that need augmentation
        insufficient_decades = []
        for decade, volume in volume_check.items():
            if volume < 0.1:  # Less than 0.1 GB
                insufficient_decades.append(decade)
                logger.warning(f"Insufficient data for {decade}: {volume:.2f} GB < 0.1 GB")
        
        # Special handling for mid-century decades with known data sparsity
        sparse_decades = ["1930s", "1940s", "1950s", "1960s", "1970s", "1980s"]
        target_decades = [decade for decade in insufficient_decades if decade in sparse_decades]
        
        if target_decades and args.apply_enhancements:
            logger.info(f"Applying targeted enhancement for sparse decades: {target_decades}")
            
            # Get focused samples for these decades
            focused_samples = dataset_manager.gutenberg_loader.load_focused_decade_samples(
                target_decades=target_decades,
                texts_per_decade=max(5000, args.texts_per_decade)
            )
            
            # Add to our dataset
            for decade, texts in focused_samples.items():
                if decade in controlled_dataset:
                    controlled_dataset[decade].extend(texts)
                else:
                    controlled_dataset[decade] = texts
            
            # Verify volumes again after enhancement
            volume_check, all_sufficient = dataset_manager.verify_dataset_volumes(
                controlled_dataset, target_gb_per_decade=0.1  # Lower threshold for sparse decades
            )
            
            if all_sufficient:
                logger.info("Targeted enhancement successful - all decades now meet minimum requirements")
            else:
                logger.warning("Some decades still have insufficient data even after targeted enhancement")
                for decade, volume in volume_check.items():
                    logger.info(f"  {decade}: {volume:.2f} GB")
        else:
            logger.warning("Dataset volumes do not meet minimum target requirements")
            for decade, volume in volume_check.items():
                logger.info(f"  {decade}: {volume:.2f} GB")
    
    # Extract just texts (without source info) and filter out empty decades
    decade_texts = {}
    for decade, texts in controlled_dataset.items():
        if texts:
            decade_texts[decade] = [item[0] if isinstance(item, tuple) else item for item in texts]
            logger.info(f"Found {len(decade_texts[decade])} texts for {decade}")
    
    # Check if we have enough decades with data
    if len(decade_texts) < 4:  # Need at least 4 decades for meaningful analysis
        logger.error(f"Insufficient data: only {len(decade_texts)} decades have data")
        logger.info(f"Decades with data: {list(decade_texts.keys())}")
        
        if args.allow_synthetic_fallback:
            # Create synthetic data for missing decades
            logger.warning("Using synthetic fallback for missing decades")
            for decade in TIME_PERIODS.keys():
                if decade not in decade_texts or not decade_texts[decade]:
                    logger.info(f"Generating synthetic texts for {decade}")
                    
                    # Use existing methods to create synthetic texts
                    synthetic_texts = dataset_manager._create_synthetic_texts_for_decade(
                        decade=decade, 
                        count=args.texts_per_decade // 2,  # Use half the target count
                    )
                    
                    decade_texts[decade] = synthetic_texts
                    logger.info(f"Generated {len(synthetic_texts)} synthetic texts for {decade}")
        else:
            logger.error("Insufficient decade coverage and synthetic fallback not enabled")
            logger.error("Use --allow_synthetic_fallback to continue with synthetic data")
            return
    
    # Process texts in chunks
    chunked_decade_texts = {}
    for decade, texts in decade_texts.items():
        # Skip empty decades
        if not texts:
            logger.warning(f"No texts for {decade}, skipping chunking")
            continue
            
        chunked_texts = dataset_manager.chunk_texts_for_tokenizer(texts)
        chunked_decade_texts[decade] = chunked_texts
        logger.info(f"Processed {decade}: {len(texts)} texts → {len(chunked_texts)} chunks")
    
    # Apply test mode filtering if needed
    if args.test_mode and test_decades:
        # Filter to only include test decades
        chunked_decade_texts = {decade: texts for decade, texts in chunked_decade_texts.items() 
                              if decade in test_decades}
        logger.info(f"Filtered to {len(chunked_decade_texts)} test decades for analysis")
    
    # Running tokenizer analysis with better handling for mid-century decades
    logger.info("Running tokenizer analysis with ensemble method...")
    start_time = time.time()

    # Analyze decade patterns with new focus on the 1960s decade
    decade_patterns = run_parallel_analysis(inference, chunked_decade_texts)
    
    # Analyze specific issues with the 1960s decade
    if '1960s' in decade_patterns:
        logger.info("Performing specific analysis of 1960s decade patterns...")
        sixties_analysis = inference.analyze_decade_specific_issues(decade_patterns, "1960s")
        
        # Save results to a separate file
        sixties_path = results_dir / "distributions" / f"{run_id}_1960s_analysis.json"
        with open(sixties_path, 'w') as f:
            json.dump(sixties_analysis, f, indent=2)
            
        # Log key findings
        logger.info(f"1960s analysis: {sixties_analysis['analysis_summary']}")
        
        # Apply correction based on analysis
        if sixties_analysis['total_distinctive_contribution'] > 20:  # More than 20%
            logger.info("1960s has highly distinctive rules, using modified inference approach")
            
            # Infer with specific 1960s correction and token removal
            distribution = inference.infer_temporal_distribution(
                decade_patterns,
                remove_top_tokens=True,
                top_n=15  # Increased to 15 to better filter problematic tokens
            )
            
            # Apply correction factor as recommended
            distribution = inference.apply_decade_correction(
                distribution,
                decade="1960s", 
                factor=sixties_analysis.get('suggested_correction_factor', 0.6)
            )
            
            logger.info("Applied 1960s correction factor")
        else:
            # Use standard ensemble approach
            distribution = inference.infer_distribution_ensemble(decade_patterns)
    else:
        # Standard approach if 1960s data not available
        distribution = inference.infer_distribution_ensemble(decade_patterns)

    # Construct results in expected format
    results = {
        "tokenizer": args.tokenizer,
        "distribution": distribution,
        "distinctive_patterns": inference.find_distinctive_patterns(decade_patterns)
    }

    inference_time = time.time() - start_time
    
    # Evaluate results
    logger.info("Evaluating results...")
    evaluation = evaluator.evaluate_distribution(
        results["distribution"],
        selected_dist,
        model_name=args.tokenizer
    )
    
    # Save detailed results
    save_distribution_results(results, evaluation, run_id, results_dir)
    
    # Output evaluation metrics
    log_evaluation_metrics(evaluation, inference_time, args)
    
    # Create comparison visualizations
    create_comparison_visualizations(results["distribution"], selected_dist, 
                                   args.distribution, args.tokenizer, results_dir)
        
    # Bootstrap validation (if requested)
    if args.bootstrap:
        # If in test mode, use fewer iterations
        if args.test_mode:
            bootstrap_iterations = min(args.bootstrap_iterations, 5)  # Just 5 iterations for testing
        else:
            bootstrap_iterations = min(args.bootstrap_iterations, 30)
            
        logger.info(f"Performing bootstrap validation with {bootstrap_iterations} iterations...")
        
        try:
            # First check for psutil
            try:
                import psutil
                has_psutil = True
            except ImportError:
                logger.warning("psutil module not available, bootstrapping may use more memory")
                has_psutil = False
                
            # Apply memory limiting function if psutil is available
            if has_psutil:
                limit_memory_usage()
                
            # Create the safe inference wrapper
            safe_inference_wrapper = create_inference_wrapper(inference)
            
            # Create validator with safe wrapper
            bootstrap_validator = TemporalValidator(inference_method=safe_inference_wrapper)
            
            # Filter texts to ensure only valid data is used
            filtered_texts = {}
            for decade, texts in decade_texts.items():
                if texts:
                    # Convert tuples to just text strings if needed
                    filtered_decade_texts = []
                    for item in texts:
                        try:
                            if isinstance(item, tuple) and len(item) >= 1:
                                # Extract the text component
                                text = item[0]
                                if isinstance(text, str):
                                    filtered_decade_texts.append((text, "filtered"))
                            elif isinstance(item, str):
                                filtered_decade_texts.append((item, "filtered"))
                        except Exception as e:
                            logger.debug(f"Skipping invalid item: {e}")
                    
                    if filtered_decade_texts:
                        filtered_texts[decade] = filtered_decade_texts
            
            # Run with the filtered texts
            confidence_intervals = bootstrap_validator.bootstrap_analysis(
                decade_texts=filtered_texts,
                n_bootstrap=bootstrap_iterations,
                sample_ratio=0.8
            )
            
            # Save bootstrap results
            bootstrap_path = results_dir / "bootstrap" / f"{run_id}_bootstrap.json"
            with open(bootstrap_path, 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                ci_json = {}
                for decade, stats in confidence_intervals.items():
                    ci_json[decade] = {k: float(v) for k, v in stats.items() if not isinstance(v, list)}
                json.dump(ci_json, f, indent=2)
            
            # Visualize with confidence intervals
            create_bootstrap_visualization(results["distribution"], selected_dist, 
                                         confidence_intervals, args.distribution, 
                                         args.tokenizer, results_dir)
            
        except Exception as e:
            logger.error(f"Error in bootstrap validation: {e}")
            logger.error("Skipping bootstrap analysis")
    
    logger.info(f"Analysis completed for {args.distribution} with {args.tokenizer}")
    
def save_distribution_results(results, evaluation, run_id, results_dir):
    """Save detailed analysis results."""
    # Save inferred distribution
    dist_path = results_dir / "distributions" / f"{run_id}_distribution.json"
    with open(dist_path, 'w') as f:
        json.dump({
            "tokenizer": results["tokenizer"],
            "distribution": {k: float(v) for k, v in results["distribution"].items()},
            "evaluation": {
                "log10_mse": float(evaluation["distribution_metrics"]["log10_mse"]),
                "mae": float(evaluation["distribution_metrics"]["mae"]),
                "js_distance": float(evaluation["distribution_metrics"]["js_distance"]),
                "rank_correlation": float(evaluation["decade_metrics"]["rank_correlation"])
            }
        }, f, indent=2)
    
    # Save distinctive patterns
    patterns_path = results_dir / "distributions" / f"{run_id}_patterns.json"
    with open(patterns_path, 'w') as f:
        # Convert tuples to lists for JSON serialization
        patterns_json = {}
        for decade, patterns in results["distinctive_patterns"].items():
            patterns_json[decade] = [[p, float(s)] for p, s in patterns]
        json.dump(patterns_json, f, indent=2)

def log_evaluation_metrics(evaluation, inference_time, args):
    """Log detailed evaluation metrics."""
    logger.info(f"Evaluation results for {args.tokenizer} on {args.distribution} distribution:")
    logger.info(f"  log10(MSE): {evaluation['distribution_metrics']['log10_mse']:.2f}")
    logger.info(f"  MAE: {evaluation['distribution_metrics']['mae']:.4f}")
    logger.info(f"  Jensen-Shannon Distance: {evaluation['distribution_metrics']['js_distance']:.4f}")
    logger.info(f"  Rank Correlation: {evaluation['decade_metrics']['rank_correlation']:.2f}")
    logger.info(f"  Inference Time: {inference_time:.2f} seconds")
    
    # Get over/under represented decades
    rep_analysis = evaluation["decade_metrics"]["representation_analysis"]
    if rep_analysis["over_represented"]:
        over_rep = sorted(rep_analysis["over_represented"].items(), key=lambda x: x[1], reverse=True)
        logger.info("  Over-represented decades:")
        for decade, value in over_rep[:3]:  # Top 3
            logger.info(f"    {decade}: +{value:.1%}")
            
    if rep_analysis["under_represented"]:
        under_rep = sorted(rep_analysis["under_represented"].items(), key=lambda x: x[1], reverse=True)
        logger.info("  Under-represented decades:")
        for decade, value in under_rep[:3]:  # Top 3
            logger.info(f"    {decade}: -{value:.1%}")

def create_comparison_visualizations(inferred, ground_truth, dist_name, tokenizer_name, results_dir):
    """Create visualizations comparing inferred and ground truth distributions."""
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
    plt.title(f'Inferred vs Ground Truth: {dist_name}')
    plt.xticks([r + bar_width/2 for r in r1], decades, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(results_dir / "figures" / f"{tokenizer_name}_{dist_name}_comparison.png", dpi=300)
    plt.close()
    
    # Create absolute error visualization
    plt.figure(figsize=(12, 6))
    
    # Calculate absolute errors
    errors = [abs(inferred.get(decade, 0) - ground_truth.get(decade, 0)) for decade in decades]
    
    # Create color-coded bars based on error magnitude
    colors = plt.cm.RdYlGn_r(np.array(errors) / max(errors) if max(errors) > 0 else np.zeros(len(errors)))
    plt.bar(decades, errors, color=colors)
    
    # Add data labels
    for i, v in enumerate(errors):
        plt.text(i, v + 0.005, f"{v:.1%}", ha='center')
    
    # Add labels and title
    plt.xlabel('Decade')
    plt.ylabel('Absolute Error')
    plt.title(f'Absolute Error by Decade: {dist_name}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(results_dir / "figures" / f"{tokenizer_name}_{dist_name}_error.png", dpi=300)
    plt.close()

def create_bootstrap_visualization(inferred_distribution, ground_truth_distribution, confidence_intervals, dist_name, tokenizer_name, results_dir):
    """Create visualization of bootstrap results with confidence intervals."""
    plt.figure(figsize=(14, 7))
    
    # Sort decades chronologically
    decades = sorted(inferred_distribution.keys())
    
    # Extract data
    means = [inferred_distribution.get(d, 0) for d in decades]
    lower = [confidence_intervals.get(d, {}).get('lower_ci', means[i] * 0.8) for i, d in enumerate(decades)]
    upper = [confidence_intervals.get(d, {}).get('upper_ci', means[i] * 1.2) for i, d in enumerate(decades)]
    
    # Calculate error bars AS POSITIVE DISTANCES
    errors_lower = [max(0, means[i] - lower[i]) for i in range(len(means))]  # Ensure positive
    errors_upper = [max(0, upper[i] - means[i]) for i in range(len(means))]  # Ensure positive
    
    # Plot with confidence intervals
    plt.bar(
        decades,
        means,
        alpha=0.7,
        color='skyblue',
        yerr=[errors_lower, errors_upper],  # This format expects positive values
        capsize=5,
        label="Bootstrap Estimate"
    )
    
    # Add ground truth as points
    plt.plot(decades, [ground_truth_distribution.get(d, 0) for d in decades], 'ro', label="Ground Truth")
    
    # Add data labels
    for i, v in enumerate(means):
        plt.text(i, v + 0.01, f"{v:.1%}", ha='center')
    
    # Add title and labels
    plt.title(f'Temporal Distribution with Confidence Intervals: {dist_name}')
    plt.xlabel('Decade')
    plt.ylabel('Estimated Proportion')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(results_dir / "figures" / f"{tokenizer_name}_{dist_name}_bootstrap.png", dpi=300)
    plt.close()

def calculate_reliability_metrics(confidence_intervals):
    """Calculate metrics to assess the reliability of the statistical analysis."""
    if not confidence_intervals:
        return {"reliability_score": 0, "coefficient_of_variation": 1.0, "normalized_ci_width": 1.0}
    
    # Calculate coefficient of variation (CV) for each decade
    cv_values = []
    ci_widths = []
    
    for decade, stats in confidence_intervals.items():
        mean = stats.get("mean", 0)
        if mean > 0:
            std_dev = stats.get("std_dev", 0)
            cv = std_dev / mean if mean > 0 else 1.0
            cv_values.append(cv)
            
            # Calculate normalized confidence interval width
            lower = stats.get("lower_ci", 0)
            upper = stats.get("upper_ci", 1)
            width = (upper - lower) / mean if mean > 0 else 1.0
            ci_widths.append(width)
    
    # Calculate average metrics
    avg_cv = sum(cv_values) / len(cv_values) if cv_values else 1.0
    avg_ci_width = sum(ci_widths) / len(ci_widths) if ci_widths else 1.0
    
    # Calculate overall reliability score (higher is better)
    # Score ranges from 0-100, with penalties for high CV and wide CIs
    cv_penalty = min(50, 50 * avg_cv)
    width_penalty = min(50, 50 * avg_ci_width / 2)  # Normalize by expected width
    reliability_score = 100 - cv_penalty - width_penalty
    
    return {
        "reliability_score": reliability_score,
        "coefficient_of_variation": avg_cv,
        "normalized_ci_width": avg_ci_width
    }

def compare_all_distributions(args):
    """Run analysis on all distributions and create comparison visualizations."""
    distributions = define_distributions()
    results_by_dist = {}
    
    # Run analysis for each distribution
    for dist_name in distributions:
        # Copy args and update distribution
        dist_args = argparse.Namespace(**vars(args))
        dist_args.distribution = dist_name
        
        # Run analysis and collect results
        logger.info(f"Running analysis for {dist_name}...")
        run_analysis(dist_args)
        
        # Load results from saved files
        results_dir = setup_directories()
        timestamp = datetime.now().strftime("%Y%m%d")  # Just use today's date
        result_files = list((results_dir / "distributions").glob(f"{args.tokenizer}_{dist_name}_*_distribution.json"))
        
        if result_files:
            # Use the most recent file
            result_file = sorted(result_files)[-1]
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                results_by_dist[dist_name] = {
                    "distribution": result_data["distribution"],
                    "evaluation": result_data["evaluation"]
                }
    
    # Create comparative visualizations
    if len(results_by_dist) > 1:
        logger.info("Creating comparative visualizations...")
        create_distribution_comparison(results_by_dist, distributions, args.tokenizer, setup_directories())

def create_distribution_comparison(results_by_dist, distributions, tokenizer_name, results_dir):
    """Create visualizations comparing results across different distributions."""
    # Extract metrics for comparison
    dist_names = list(results_by_dist.keys())
    log_mse_values = [results_by_dist[d]["evaluation"]["log10_mse"] for d in dist_names]
    mae_values = [results_by_dist[d]["evaluation"]["mae"] for d in dist_names]
    js_values = [results_by_dist[d]["evaluation"]["js_distance"] for d in dist_names]
    correlation_values = [results_by_dist[d]["evaluation"]["rank_correlation"] for d in dist_names]
    
    # Create figure with 2x2 subplots for metrics comparison
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    # Plot log10(MSE)
    axs[0].bar(dist_names, log_mse_values, color='royalblue')
    axs[0].set_title('log10(MSE) by Distribution Pattern\n(lower is better)')
    axs[0].set_ylabel('log10(MSE)')
    # Add Hayase benchmark line
    axs[0].axhline(y=-7.3, color='red', linestyle='--', 
                 label='Hayase benchmark: -7.3')
    axs[0].legend()
    
    # Plot Mean Absolute Error
    axs[1].bar(dist_names, mae_values, color='royalblue')
    axs[1].set_title('Mean Absolute Error by Distribution Pattern\n(lower is better)')
    axs[1].set_ylabel('MAE')
    
    # Plot Jensen-Shannon Distance
    axs[2].bar(dist_names, js_values, color='royalblue')
    axs[2].set_title('Jensen-Shannon Distance by Distribution Pattern\n(lower is better)')
    axs[2].set_ylabel('Jensen-Shannon Distance')
    
    # Plot Rank Correlation
    axs[3].bar(dist_names, correlation_values, color='royalblue')
    axs[3].set_title('Rank Correlation by Distribution Pattern\n(higher is better)')
    axs[3].set_ylabel('Rank Correlation')
    
    # Add labels and adjust layout
    for ax in axs:
        ax.set_xticklabels(dist_names, rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / f"{tokenizer_name}_metric_comparison.png", dpi=300)
    plt.close()
    
    # Create error by decade comparison
    plt.figure(figsize=(14, 8))
    
    # Get all decades across all distributions
    all_decades = set()
    for dist_name in dist_names:
        inferred = results_by_dist[dist_name]["distribution"]
        ground_truth = distributions[dist_name]["distribution"]
        all_decades.update(set(inferred.keys()) | set(ground_truth.keys()))
    
    # Sort decades chronologically
    decades = sorted(all_decades)
    
    # Calculate errors for each distribution
    for i, dist_name in enumerate(dist_names):
        inferred = results_by_dist[dist_name]["distribution"]
        ground_truth = distributions[dist_name]["distribution"]
        
        errors = [abs(inferred.get(decade, 0) - ground_truth.get(decade, 0)) for decade in decades]
        plt.plot(decades, errors, 'o-', label=dist_name, color=plt.cm.tab10(i))
    
    # Add labels and title
    plt.xlabel('Decade')
    plt.ylabel('Absolute Error')
    plt.title('Error by Decade Across Distribution Patterns')
    plt.xticks(rotation=45)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(results_dir / "figures" / f"{tokenizer_name}_error_comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run temporal distribution inference on Maxwell")
    parser.add_argument("--test_mode", action="store_true", 
                  help="Run in test mode with minimal data for error checking")
    parser.add_argument("--test_size_mb", type=float, default=10.0,
                    help="Size of test data in MB per decade (only used with --test_mode)")
    parser.add_argument("--test_decades", type=str, default="1950s,2000s",
                    help="Comma-separated list of decades to use in test mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to analyze")
    parser.add_argument("--texts_per_decade", type=int, default=5000, 
                      help="Number of texts per decade (higher = more accurate)")
    parser.add_argument("--target_size_gb", type=float, default=1.0,
                      help="Target size in GB per category (higher = more accurate, matching Hayase paper)")
    parser.add_argument("--distribution", type=str, default="uniform", 
                      choices=["uniform", "recency_bias", "historical_bias", "bimodal", "all"],
                      help="Distribution pattern to test (use 'all' to run all patterns)")
    parser.add_argument("--bootstrap", action="store_true", 
                      help="Perform bootstrap validation for confidence intervals")
    parser.add_argument("--bootstrap_iterations", type=int, default=30,
                      help="Number of bootstrap iterations to perform")
    parser.add_argument("--force_fresh", action="store_true", 
                      help="Force fresh dataset creation (ignore cache)")
    parser.add_argument("--force_quality", action="store_true",
                      help="Only proceed with analysis if dataset meets quality standards")
    parser.add_argument("--apply_enhancements", action="store_true",
                      help="Apply targeted enhancements for sparse decades")
    parser.add_argument("--allow_synthetic_fallback", action="store_true",
                      help="Allow synthetic data generation for missing decades")

    args = parser.parse_args()
    
    # Run all distributions or just the specified one
    if args.distribution == "all":
        compare_all_distributions(args)
    else:
        run_analysis(args)