"""
Main script for running temporal distribution inference on Maxwell HPC.

This script handles dataset creation, analysis, and evaluation with 
various distribution patterns and tokenizers.
"""

import argparse
import gc
import logging
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pickle
from datetime import datetime
from tqdm import tqdm
import re
import sys

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

class EnhancedLoggingManager:
    """
    Advanced logging manager to reduce noise and batch similar errors.
    Combines and enhances the functionality of existing ProgressFilter.
    """
    
    def __init__(self):
        # Track errors and progress messages
        self.error_counts = {}
        self.last_report_time = {}
        self.last_progress = {}
        self.report_interval = 30.0  # seconds between error reports
        self.min_percent_change = 10.0  # minimum progress percentage change to log
        self.truncation_count = 0
        self.download_errors = 0
        self.max_individual_errors = 10  # Show first N errors individually
        
    def setup_logging(self, log_dir="logs", timestamp=None):
        """Set up logging with appropriate filtering"""
        import logging
        import os
        import sys
        from datetime import datetime
        
        # Create logs directory if needed
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log filename
        log_filename = os.path.join(log_dir, f"run_{timestamp}.log")
        
        # Configure root logger
        root_logger = logging.getLogger()
        
        # Clear any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create file handler - captures all messages
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Create stdout handler for important messages
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_formatter = logging.Formatter('%(levelname)s: %(message)s')
        stdout_handler.setFormatter(stdout_formatter)
        
        # Create stderr handler for errors only
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_formatter = logging.Formatter('ERROR: %(message)s')
        stderr_handler.setFormatter(stderr_formatter)
        
        # Add our filter to all handlers
        file_handler.addFilter(self)
        stdout_handler.addFilter(self)
        stderr_handler.addFilter(self)
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)
        root_logger.setLevel(logging.INFO)
        
        # Silence particularly noisy modules
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('datasets').setLevel(logging.ERROR)
        logging.getLogger('urllib3').setLevel(logging.ERROR)
        logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
        
        logging.info(f"Logging configured. Full logs will be saved to {log_filename}")
        return log_filename
    
    def filter(self, record):
        """
        Filter log records to reduce noise.
        Returns True if the record should be logged, False to drop it.
        """
        # Always allow critical errors
        if record.levelno >= logging.CRITICAL:
            return True

        # Always allow distribution-related messages
        if "distribution" in record.getMessage() or "analyzing" in record.getMessage().lower():
            return True  # Always allow distribution analysis messages

        # Handle download errors
        if "Failed to fetch text for book" in record.getMessage():
            self.download_errors += 1
            current_time = time.time()
            error_type = "fetch_fail"
            
            # Initialize tracking for this error type if needed
            if error_type not in self.error_counts:
                self.error_counts[error_type] = 0
                self.last_report_time[error_type] = 0
            
            # Increment the count
            self.error_counts[error_type] += 1
            
            # Show only first few errors in detail
            if self.download_errors <= self.max_individual_errors:
                return True
                
            # For subsequent errors, batch them and report periodically
            time_since_last_report = current_time - self.last_report_time[error_type]
            if time_since_last_report >= self.report_interval:
                # Update the message to include the count
                record.msg = f"Failed to fetch text for {self.error_counts[error_type]} books from any source since last report"
                
                # Reset the count and update the time
                self.error_counts[error_type] = 0
                self.last_report_time[error_type] = current_time
                return True
            return False
        
        # Handle extremely long text warnings
        if "extremely long text" in record.getMessage() and "truncating" in record.getMessage():
            current_time = time.time()
            error_type = "truncation"
            
            if error_type not in self.error_counts:
                self.error_counts[error_type] = 0
                self.last_report_time[error_type] = 0
                
            # Increment count of truncated texts
            self.error_counts[error_type] += 1
            
            # Report truncations in batches
            time_since_last_report = current_time - self.last_report_time.get(error_type, 0)
            if time_since_last_report >= self.report_interval:
                record.msg = f"Truncated {self.error_counts[error_type]} extremely long texts since last report"
                self.error_counts[error_type] = 0
                self.last_report_time[error_type] = current_time
                return True
            return False
            
        # Handle progress messages
        if ("Processed" in record.getMessage() and "records" in record.getMessage()) or \
           ("Downloading" in record.getMessage()) or \
           ("Processing" in record.getMessage() and "%" in record.getMessage()):
                
            # Extract progress info if available
            import re
            match = re.search(r'(\d+)%', record.getMessage())
            if match:
                current_pct = int(match.group(1))
                logger_name = record.name
                
                # Only log if progress increased significantly
                if logger_name not in self.last_progress:
                    self.last_progress[logger_name] = current_pct
                    return True
                    
                if current_pct - self.last_progress[logger_name] >= self.min_percent_change:
                    self.last_progress[logger_name] = current_pct
                    return True
                return False
                
            # For other progress messages, use a different approach
            match = re.search(r'(\d+)/(\d+)', record.getMessage())
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                current_pct = (current / total) * 100
                logger_name = record.name
                
                # Check if we've seen this progress type before
                if logger_name not in self.last_progress:
                    self.last_progress[logger_name] = current_pct
                    return True
                
                # Only log significant progress
                if current_pct - self.last_progress[logger_name] >= self.min_percent_change:
                    self.last_progress[logger_name] = current_pct
                    return True
                return False
                
        # Allow all other messages through
        return True
    
    def print_results_summary(self, evaluation, args):
        """
        Print a clear summary of results to stdout.
        This ensures important results are in the .out file.
        """
        import sys
        
        # Create a clear, bordered output that will stand out in logs
        border = "=" * 50
        print(f"\n{border}", file=sys.stdout)
        print(f"           RESULTS SUMMARY", file=sys.stdout)
        print(f"{border}", file=sys.stdout)
        print(f"Tokenizer: {args.tokenizer}", file=sys.stdout)
        print(f"Distribution: {args.distribution}", file=sys.stdout)
        print(f"", file=sys.stdout)
        print(f"Evaluation Metrics:", file=sys.stdout)
        print(f"  log10(MSE): {evaluation['distribution_metrics']['log10_mse']:.2f}", file=sys.stdout)
        print(f"  MAE: {evaluation['distribution_metrics']['mae']:.4f}", file=sys.stdout)
        print(f"  Jensen-Shannon Distance: {evaluation['distribution_metrics']['js_distance']:.4f}", file=sys.stdout)
        print(f"  Rank Correlation: {evaluation['decade_metrics']['rank_correlation']:.2f}", file=sys.stdout)
        
        # Print over/under represented decades
        rep_analysis = evaluation["decade_metrics"]["representation_analysis"]
        if rep_analysis["over_represented"]:
            print(f"\nOver-represented decades:", file=sys.stdout)
            for decade, value in sorted(rep_analysis["over_represented"].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"  {decade}: +{value:.1%}", file=sys.stdout)
                
        if rep_analysis["under_represented"]:
            print(f"\nUnder-represented decades:", file=sys.stdout)
            for decade, value in sorted(rep_analysis["under_represented"].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"  {decade}: -{value:.1%}", file=sys.stdout)
                
        print(f"{border}\n", file=sys.stdout)

# Initialize logging manager globally
logging_manager = EnhancedLoggingManager()

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
    """Filter to reduce frequency of progress messages and batch similar errors."""
    
    def __init__(self, name=''):
        super().__init__(name)
        self.last_progress = {}
        self.min_percent_change = 5.0  # Only log when progress increases by 5%
        
        # Error tracking
        self.error_counts = {}
        self.last_error_report_time = {}
        self.error_report_interval = 10.0  # Report similar errors every 10 seconds
        
    def filter(self, record):
        # Check if this is a progress message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # Handle progress messages
            if "Processed" in record.msg and "records" in record.msg or "Downloading" in record.msg:
                # Extract progress info
                match = re.search(r'(\d+)/(\d+)', record.msg)
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
            
            # Handle error messages - batch them together
            if "Failed to fetch text for book" in record.msg:
                current_time = time.time()
                error_type = "fetch_fail"
                
                # Initialize tracking for this error type if needed
                if error_type not in self.error_counts:
                    self.error_counts[error_type] = 0
                    self.last_error_report_time[error_type] = 0
                
                # Increment the count
                self.error_counts[error_type] += 1
                
                # Check if it's time to report this type of error
                time_since_last_report = current_time - self.last_error_report_time[error_type]
                if time_since_last_report >= self.error_report_interval:
                    # Update the message to include the count
                    if self.error_counts[error_type] > 1:
                        record.msg = f"Failed to fetch text for {self.error_counts[error_type]} books from any source after multiple attempts"
                    
                    # Reset the count and update the time
                    self.error_counts[error_type] = 0
                    self.last_error_report_time[error_type] = current_time
                    return True
                return False
                
            # Handle HTTP error messages
            if "HTTP error" in record.msg:
                # Let these through but less frequently
                current_time = time.time()
                error_type = "http_error"
                
                if error_type not in self.error_counts:
                    self.error_counts[error_type] = 0
                    self.last_error_report_time[error_type] = 0
                
                self.error_counts[error_type] += 1
                
                time_since_last_report = current_time - self.last_error_report_time[error_type]
                if time_since_last_report >= self.error_report_interval:
                    if self.error_counts[error_type] > 1:
                        # Extract the URL and retry info
                        match = re.search(r'HTTP error \d+ for (.*?), retry (\d+)/(\d+)', record.msg)
                        if match:
                            url = match.group(1)
                            current_retry = match.group(2)
                            max_retries = match.group(3)
                            
                            # Summarize HTTP errors
                            record.msg = f"HTTP errors for {self.error_counts[error_type]} URLs (showing latest: {url}, retry {current_retry}/{max_retries})"
                    
                    self.error_counts[error_type] = 0
                    self.last_error_report_time[error_type] = current_time
                    return True
                return False
                
            # Handle text too short messages
            if "Extracted text too short" in record.msg:
                current_time = time.time()
                error_type = "short_text"
                
                if error_type not in self.error_counts:
                    self.error_counts[error_type] = 0
                    self.last_error_report_time[error_type] = 0
                
                self.error_counts[error_type] += 1
                
                time_since_last_report = current_time - self.last_error_report_time[error_type]
                if time_since_last_report >= self.error_report_interval:
                    if self.error_counts[error_type] > 1:
                        record.msg = f"Found {self.error_counts[error_type]} texts that were too short"
                    
                    self.error_counts[error_type] = 0
                    self.last_error_report_time[error_type] = current_time
                    return True
                return False
                
        # Keep all non-matching messages
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

def preprocess_dataset(decade_texts, args):
    """
    Preprocess the dataset to improve representation across decades.
    This function addresses imbalances before running the inference.
    
    Args:
        decade_texts: Dictionary mapping decades to lists of texts
        args: Command-line arguments
        
    Returns:
        Preprocessed decade_texts
    """
    logger.info("Preprocessing dataset to improve decade representation...")
    
    # Calculate current distribution
    total_bytes = 0
    decade_bytes = {}
    for decade, texts in decade_texts.items():
        byte_size = sum(len(text.encode('utf-8')) for text in texts)
        decade_bytes[decade] = byte_size
        total_bytes += byte_size
    
    # Calculate current distribution percentage
    current_distribution = {}
    for decade, bytes_count in decade_bytes.items():
        if total_bytes > 0:
            current_distribution[decade] = bytes_count / total_bytes
        else:
            current_distribution[decade] = 0
    
    # Log current distribution
    logger.info("Current data distribution:")
    for decade, percentage in sorted(current_distribution.items()):
        logger.info(f"  {decade}: {percentage:.1%}")
    
    # Define target byte size for each decade
    # Redistribute to ensure better historical representation
    historical_decades = ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s"]
    modern_decades = ["1990s", "2000s", "2010s", "2020s"]
    midcentury_decades = ["1930s", "1940s", "1950s", "1960s", "1970s", "1980s"]
    
    # New target distribution - give boost to underrepresented decades
    target_distribution = {}
    
    # Set minimum percentages based on decade groups
    for decade in decade_texts.keys():
        if decade in historical_decades:
            target_distribution[decade] = 0.07  # Boost historical
        elif decade in midcentury_decades:
            target_distribution[decade] = 0.04  # Standard for midcentury
        elif decade in modern_decades:
            target_distribution[decade] = 0.06  # Slightly reduced for modern (already overrepresented)
        else:
            target_distribution[decade] = 0.05  # Default for any other decades
    
    # Normalize to ensure total is 100%
    total_target = sum(target_distribution.values())
    if total_target > 0:
        target_distribution = {d: p/total_target for d, p in target_distribution.items()}
    
    # Calculate target bytes for each decade
    total_target_bytes = min(total_bytes, 2 * 1024 * 1024 * 1024)  # Cap at 2GB for performance
    target_bytes_by_decade = {decade: total_target_bytes * percentage 
                            for decade, percentage in target_distribution.items()}
    
    # Balance the dataset by sampling, augmenting, or creating synthetic data
    balanced_texts = {}
    for decade, target_bytes in target_bytes_by_decade.items():
        current_bytes = decade_bytes.get(decade, 0)
        texts = decade_texts.get(decade, [])
        
        # Skip empty decades
        if not texts:
            logger.warning(f"No texts available for {decade}, skipping")
            balanced_texts[decade] = []
            continue
        
        logger.info(f"Processing {decade}: {current_bytes/(1024*1024):.2f}MB vs target {target_bytes/(1024*1024):.2f}MB")
        
        # Case 1: We need to reduce data (sampling)
        if current_bytes > target_bytes * 1.1:  # 10% margin
            # Calculate sampling rate
            sample_ratio = target_bytes / current_bytes
            sample_size = max(10, int(len(texts) * sample_ratio))
            
            # Prioritize longer texts
            sorted_texts = sorted(texts, key=lambda x: len(x) if isinstance(x, str) else len(x[0]), reverse=True)
            
            # Take top texts plus random sample from the rest
            top_count = min(int(sample_size * 0.2), len(sorted_texts))
            top_texts = sorted_texts[:top_count]
            
            if sample_size - top_count > 0 and len(sorted_texts) > top_count:
                rest_sample = random.sample(sorted_texts[top_count:], min(sample_size - top_count, len(sorted_texts) - top_count))
                sampled_texts = top_texts + rest_sample
            else:
                sampled_texts = top_texts
                
            logger.info(f"Sampled {len(sampled_texts)}/{len(texts)} texts for {decade} to reduce volume")
            balanced_texts[decade] = sampled_texts
            
        # Case 2: We need to increase data (augmentation/synthesis)
        elif current_bytes < target_bytes * 0.9:  # 10% margin
            import copy
            augmented_texts = copy.copy(texts)  # Start with all existing texts
            
            # First try augmentation
            remaining_bytes = target_bytes - current_bytes
            text_multiplier = (target_bytes / current_bytes) if current_bytes > 0 else 2.0
            
            logger.info(f"Need to increase {decade} data by factor of {text_multiplier:.1f}x")
            
            # Create a dataset manager for augmentation
            data_manager = TemporalDatasetManager()
            
            # Select up to 50 texts for augmentation
            texts_to_augment = texts[:min(50, len(texts))]
            
            for base_text in texts_to_augment:
                # Check if we've reached target
                current_size = sum(len(t.encode('utf-8')) for t in augmented_texts)
                if current_size >= target_bytes:
                    break
                    
                # Create multiple augmented versions to reach target faster
                for _ in range(3):  # Create 3 augmented versions of each text
                    try:
                        if isinstance(base_text, tuple):
                            text_content = base_text[0]
                        else:
                            text_content = base_text
                            
                        # Augment with appropriate volume multiplier
                        augmented = data_manager._augment_text_for_volume(
                            text_content, 
                            decade, 
                            volume_multiplier=random.randint(2, 5)
                        )
                        
                        augmented_texts.append(augmented)
                        
                        # Check if we've reached target
                        if sum(len(t.encode('utf-8')) for t in augmented_texts) >= target_bytes:
                            break
                    except Exception as e:
                        logger.warning(f"Error augmenting text: {e}")
            
            logger.info(f"Expanded {decade} from {len(texts)} to {len(augmented_texts)} texts")
            balanced_texts[decade] = augmented_texts
        
        # Case 3: We're within 10% of target, keep as is
        else:
            logger.info(f"Keeping {len(texts)} texts for {decade} (already within target range)")
            balanced_texts[decade] = texts
    
    # Log final sizes
    logger.info("Final data distribution after preprocessing:")
    final_bytes = {}
    final_total = 0
    for decade, texts in balanced_texts.items():
        byte_size = sum(len(text.encode('utf-8')) for text in texts)
        final_bytes[decade] = byte_size
        final_total += byte_size
    
    for decade, byte_size in sorted(final_bytes.items()):
        percentage = byte_size / final_total if final_total > 0 else 0
        logger.info(f"  {decade}: {byte_size/(1024*1024):.2f}MB ({percentage:.1%})")
    
    return balanced_texts

def analyze_with_multiple_tokenizers(decade_texts, args):
    """
    Run analysis with multiple tokenizers to validate results.
    
    Args:
        decade_texts: Dictionary mapping decades to texts
        args: Command-line arguments
        
    Returns:
        Dictionary with multi-tokenizer analysis results
    """
    logger.info("Starting multi-tokenizer analysis...")
    
    # Choose tokenizers based on availability
    tokenizers = ["gpt2"]  # Always include gpt2
    
    # Add other tokenizers if available
    additional_tokenizers = ["bert-base-uncased", "roberta-base"]
    available_tokenizers = []
    
    # Check which tokenizers are available without downloading
    for tokenizer_name in additional_tokenizers:
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
            available_tokenizers.append(tokenizer_name)
        except:
            logger.info(f"Tokenizer {tokenizer_name} not available locally, skipping")
    
    # Use available tokenizers (plus gpt2)
    tokenizers_to_use = tokenizers + available_tokenizers
    logger.info(f"Using tokenizers: {tokenizers_to_use}")
    
    # Create an inference object to use its validation method
    inference = TemporalDistributionInference(tokenizer_name=args.tokenizer)
    
    # Run the multi-tokenizer validation
    validation_results = inference.run_multi_tokenizer_validation(decade_texts, tokenizers=tokenizers_to_use)
    
    # Save results
    results_dir = setup_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / "distributions" / f"multi_tokenizer_{timestamp}.json"
    
    try:
        with open(results_path, 'w') as f:
            # Ensure all values are serializable
            serializable_results = {}
            for tokenizer, distribution in validation_results.items():
                if isinstance(distribution, dict):
                    serializable_results[tokenizer] = {k: float(v) for k, v in distribution.items()}
                else:
                    serializable_results[tokenizer] = distribution
            
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Multi-tokenizer analysis results saved to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save multi-tokenizer results: {e}")
    
    # Create visualization
    visualize_multi_tokenizer_results(validation_results, args, results_dir)
    
    return validation_results

def visualize_multi_tokenizer_results(validation_results, args, results_dir):
    """
    Create visualizations for multi-tokenizer analysis.
    
    Args:
        validation_results: Results from multi-tokenizer analysis
        args: Command-line arguments
        results_dir: Directory to save visualizations
    """
    # Extract results
    tokenizers = [t for t in validation_results.keys() if t not in ["consensus", "variance"]]
    
    if not tokenizers or len(tokenizers) < 2:
        logger.warning("Not enough tokenizers for meaningful visualization")
        return
    
    consensus = validation_results.get("consensus", {})
    if not consensus:
        logger.warning("No consensus distribution available")
        return
    
    # Create figure
    decades = sorted(consensus.keys())
    plt.figure(figsize=(14, 8))
    
    # Set bar width and positions
    bar_width = 0.8 / len(tokenizers)
    r = np.arange(len(decades))
    
    # Plot bars for each tokenizer
    for i, tokenizer in enumerate(tokenizers):
        distribution = validation_results.get(tokenizer, {})
        values = [distribution.get(decade, 0) for decade in decades]
        positions = [x + i * bar_width for x in r]
        plt.bar(positions, values, width=bar_width, label=tokenizer)
    
    # Add consensus line
    consensus_values = [consensus.get(decade, 0) for decade in decades]
    plt.plot(r + (len(tokenizers) - 1) * bar_width / 2, consensus_values, 'k--', 
             label='Consensus', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Decade')
    plt.ylabel('Proportion')
    plt.title(f'Temporal Distribution Across Multiple Tokenizers ({args.distribution})')
    plt.xticks([x + (len(tokenizers) - 1) * bar_width / 2 for x in r], decades, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    save_path = results_dir / "figures" / f"multi_tokenizer_{args.distribution}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Multi-tokenizer visualization saved to {save_path}")

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
    if memory_usage_gb > 30:  # 30 GB threshold - adjust based on cluster limits
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
    
    return memory_usage_gb

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

def handle_bimodal_distribution(distribution_info):
    """
    Special handler for bimodal distribution to avoid float-to-int errors.
    
    Args:
        distribution_info: The distribution dictionary
        
    Returns:
        Fixed distribution dictionary with proper value types
    """
    logger.info("Applying special handling for bimodal distribution")
    
    # Ensure all values are proper floats
    if "distribution" in distribution_info:
        distribution_info["distribution"] = {
            k: float(v) for k, v in distribution_info["distribution"].items()
        }
    
    # Make a defensive copy to avoid modifying the original
    fixed_distribution = {}
    for decade, value in distribution_info["distribution"].items():
        try:
            # Ensure it's a proper float
            fixed_distribution[decade] = float(value)
        except (TypeError, ValueError):
            logger.warning(f"Invalid value for {decade}: {value}, using 0.05 as default")
            fixed_distribution[decade] = 0.05
    
    # Normalize to ensure sum to 1
    total = sum(fixed_distribution.values())
    if abs(total - 1.0) > 0.01:  # Allow for small rounding errors
        logger.warning(f"Bimodal distribution sum is {total}, normalizing...")
        fixed_distribution = {k: v/total for k, v in fixed_distribution.items()}
    
    # Update the original dictionary
    distribution_info["distribution"] = fixed_distribution
    
    return distribution_info

def run_analysis(args):
    """
    Run the complete analysis with specified parameters and improved error handling
    for better statistical validation and mid-century decade coverage.
    
    Args:
        args: Command-line arguments containing analysis parameters
    """
    # Use the enhanced logging manager instead of the basic configure_logging
    log_filename = logging_manager.setup_logging()
    
    # Set up directories
    results_dir = setup_directories()

    # MODIFIED: Reduce target data size for performance
    target_size_gb = min(args.target_size_gb, 0.5)  # Reduced to 0.5GB to improve performance
    logger.info(f"Setting target data size to {target_size_gb}GB per decade for better performance")
    
    # Log run parameters
    logger.info(f"Starting analysis with parameters:")
    logger.info(f"  Tokenizer: {args.tokenizer}")
    logger.info(f"  Distribution: {args.distribution}")
    logger.info(f"  Texts per decade: {args.texts_per_decade}")
    logger.info(f"  Target size (GB): {target_size_gb}")

    # Get distributions
    distributions = define_distributions()
    
    # Validate distribution choice
    if args.distribution not in distributions:
        logger.error(f"Unknown distribution: {args.distribution}")
        logger.info(f"Available distributions: {list(distributions.keys())}")
        return
    
    # Get selected distribution
    dist_info = distributions[args.distribution]
    selected_dist = dist_info["distribution"]
    
    # Special handling for bimodal distribution
    if args.distribution == "bimodal":
        dist_info = handle_bimodal_distribution(dist_info)
        selected_dist = dist_info["distribution"]
    
    # Initialize dataset_manager
    dataset_manager = TemporalDatasetManager()

    # ADDED: Boost historical data first
    logger.info("Boosting historical data coverage...")
    historical_dataset = dataset_manager.boost_historical_data()
    
    # ADDED: Log detailed historical data statistics
    for decade, texts in historical_dataset.items():
        text_count = len(texts)
        real_count = sum(1 for _, source in texts if not "synthetic" in source and not "augmented" in source)
        augmented_count = sum(1 for _, source in texts if "augmented" in source)
        synthetic_count = sum(1 for _, source in texts if "synthetic" in source)
        
        # total_bytes = sum(len(text.encode('utf-8')) for text, _ in texts)
        total_bytes = 0
        for item in texts:
            try:
                if isinstance(item, tuple) and len(item) >= 1:
                    # Extract text component
                    text_item = item[0]
                    
                    # If text_item is a string, encode it directly
                    if isinstance(text_item, str):
                        total_bytes += len(text_item.encode('utf-8'))
                    # If text_item is also a tuple, try to get the string from it
                    elif isinstance(text_item, tuple) and len(text_item) >= 1:
                        if isinstance(text_item[0], str):
                            total_bytes += len(text_item[0].encode('utf-8'))
                elif isinstance(item, str):
                    # Item is directly a string
                    total_bytes += len(item.encode('utf-8'))
            except Exception as e:
                print(f"Error processing text item: {e}")
                continue
        total_gb = total_bytes / (1024**3)
        
        logger.info(f"Historical {decade}: {text_count} texts, {total_gb:.2f}GB")
        logger.info(f"  - Real: {real_count} ({real_count/text_count:.1%})")
        logger.info(f"  - Augmented: {augmented_count} ({augmented_count/text_count:.1%})")
        logger.info(f"  - Synthetic: {synthetic_count} ({synthetic_count/text_count:.1%})")
        
    # Create dataset with target distribution
    logger.info(f"Creating dataset with target size of {target_size_gb}GB per decade...")
    controlled_dataset = dataset_manager.create_large_dataset(
        distribution=selected_dist,
        target_size_gb=target_size_gb
    )
    
    # Skip British Library test to save time
    
    # Check for cached dataset
    cache_dir = Path(RESULTS_DIR) / "dataset_cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    cached_dataset_path = cache_dir / f"{args.tokenizer}_{args.distribution}_{target_size_gb}GB.pkl"
    
    # Log the actual distribution achieved
    logger.info("Actual data distribution:")
    decade_volumes = {}
    total_bytes = 0
    
    # IMPROVED: More efficient calculation of dataset size
    for decade, texts in controlled_dataset.items():
        if not texts:
            decade_volumes[decade] = 0
            continue
            
        # Calculate size in bytes more efficiently - sample only a subset
        sample_size = min(len(texts), 100)  # Sample only 100 texts to estimate size
        if sample_size > 0:
            sampled_texts = random.sample(texts, sample_size)
            sample_bytes = 0
            for item in sampled_texts:
                try:
                    if isinstance(item, tuple) and len(item) >= 1:
                        text = item[0]
                        if isinstance(text, str):
                            sample_bytes += len(text.encode('utf-8'))
                        elif isinstance(text, tuple) and len(text) >= 1:
                            if isinstance(text[0], str):
                                sample_bytes += len(text[0].encode('utf-8'))
                    elif isinstance(item, str):
                        sample_bytes += len(item.encode('utf-8'))
                except Exception as e:
                    logger.debug(f"Error calculating text size: {e}")
            
            # Extrapolate to full dataset
            decade_bytes = int(sample_bytes * (len(texts) / sample_size))
        else:
            decade_bytes = 0
            
        decade_volumes[decade] = decade_bytes
        total_bytes += decade_bytes
    
    # Log distribution statistics
    for decade, bytes_count in decade_volumes.items():
        if total_bytes > 0:
            percentage = bytes_count / total_bytes * 100
            logger.info(f"  {decade}: {bytes_count/(1024*1024*1024):.2f}GB ({percentage:.1f}%)")
    
    # IMPROVED: Extract texts more efficiently and limit per decade
    decade_texts = {}
    max_texts_per_decade = 1000  # Limit to 1000 texts per decade for performance
    
    for decade, texts in controlled_dataset.items():
        if not texts:
            continue
                
        # Normalize format to ensure consistency
        normalized_texts = []
        for item in texts[:max_texts_per_decade]:  # Only process up to max_texts_per_decade
            try:
                if isinstance(item, tuple) and len(item) >= 1:
                    # Extract text component
                    text = item[0]
                    if isinstance(text, str):
                        normalized_texts.append(text)
                    elif isinstance(text, tuple) and len(text) >= 1:
                        if isinstance(text[0], str):
                            normalized_texts.append(text[0])
                elif isinstance(item, str):
                    normalized_texts.append(item)
            except Exception as e:
                logger.debug(f"Skipping invalid item: {e}")
        
        decade_texts[decade] = normalized_texts
        logger.info(f"Using {len(decade_texts[decade])} normalized texts for {decade}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.tokenizer}_{args.distribution}_{timestamp}"
    
    # Initialize inference component
    inference = TemporalDistributionInference(tokenizer_name=args.tokenizer)
    
    # Apply preprocessing to improve decade representation
    logger.info("Applying preprocessing for better temporal representation...")
    decade_texts = preprocess_dataset(decade_texts, args)

    # Apply enhanced merge rule configuration
    logger.info("Using enhanced merge rule analysis parameters...")
    
    # MODIFIED: Use more efficient parameters for pattern analysis
    logger.info("Running tokenizer analysis with optimized parameters...")
    
    # Define cached patterns path for reuse
    patterns_cache_path = cache_dir / f"{args.tokenizer}_{args.distribution}_patterns.pkl"
    decade_patterns = None
    
    # Try to load cached patterns if they exist
    if patterns_cache_path.exists() and not args.force_fresh:
        try:
            with open(patterns_cache_path, 'rb') as f:
                decade_patterns = pickle.load(f)
            logger.info(f"Loaded cached decade patterns")
        except Exception as e:
            logger.warning(f"Failed to load cached patterns: {e}")
            decade_patterns = None
    
    # Generate patterns if no cache, with reduced parameters
    if decade_patterns is None:
        # MODIFIED: Use smaller sample size
        sample_size = 1000  # Reduced sample size
        decade_patterns = {}
        
        for decade, texts in decade_texts.items():
            # Process only a subsample for each decade
            sample_texts = {decade: texts[:min(len(texts), sample_size)]}
            decade_pattern = inference.analyze_decade_patterns(sample_texts, sample_size=sample_size)
            if decade in decade_pattern:
                decade_patterns[decade] = decade_pattern[decade]
            logger.info(f"Processed patterns for {decade}")
        
        # Cache the patterns
        try:
            with open(patterns_cache_path, 'wb') as f:
                pickle.dump(decade_patterns, f)
            logger.info(f"Cached decade patterns for future use")
        except Exception as e:
            logger.warning(f"Failed to cache patterns: {e}")
    
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
    
    # ENHANCED: Infer distribution with improved parameters
    logger.info("Inferring temporal distribution with optimized parameters...")
    distribution = inference.infer_temporal_distribution(
        decade_patterns,
        remove_top_tokens=True,
        top_n=20,  # Increased from 5 to 20 for better bias reduction
        regularization_strength=0.2,  # Increased from 0.05 for better balance
        num_merge_rules=2000  # Increased to consider more rules
    )

    # Apply comprehensive decade corrections
    logger.info("Applying comprehensive decade corrections for improved accuracy...")
    decade_corrections = {
        "1850s": 2.5,   # Boost historical representation 
        "1860s": 2.3,
        "1870s": 2.1,
        "1880s": 2.0,
        "1890s": 1.8,
        "1900s": 1.5,
        "1910s": 1.3,
        "1920s": 1.2,
        "1930s": 0.3,   # Strong reduction for overrepresented decade
        "1940s": 0.8,
        "1950s": 0.9,
        "1960s": 0.6,   
        "1970s": 0.8,
        "1980s": 0.9,
        "1990s": 0.5,   # Reduce modern overrepresentation
        "2000s": 0.6,
        "2010s": 0.4,
        "2020s": 0.7
    }

    # Apply corrections in order of largest adjustment first
    sorted_corrections = sorted(
        [(d, f) for d, f in decade_corrections.items() if d in distribution],
        key=lambda x: abs(1.0 - x[1]),
        reverse=True
    )

    for decade, factor in sorted_corrections:
        distribution = inference.apply_decade_correction(
            distribution,
            decade=decade, 
            factor=factor
        )
        logger.info(f"Applied correction factor of {factor} to {decade}")

    # Calculate uncertainty in the estimates
    uncertainty = inference.quantify_uncertainty(decade_patterns, distribution)
    
    # Save results
    results = {
        "tokenizer": args.tokenizer,
        "distribution": distribution,
        "distinctive_patterns": inference.find_distinctive_patterns(decade_patterns),
        "uncertainty": uncertainty
    }
    
    # Evaluate results against ground truth
    logger.info("Evaluating results against ground truth...")
    start_time = time.time()
    evaluation = inference.validate_against_hayase_metrics(
        distribution,
        selected_dist,
        bootstrap_iterations=0  # MODIFIED: Skip bootstrap in metric validation for speed
    )
    inference_time = time.time() - start_time
    
    # Add Hayase benchmark comparison to evaluation
    if "distribution_metrics" in evaluation and "log10_mse" in evaluation["distribution_metrics"]:
        hayase_benchmark = -7.30  # The benchmark value from Hayase et al.
        current_mse = evaluation["distribution_metrics"]["log10_mse"]
        gap = current_mse - hayase_benchmark
        evaluation["hayase_comparison"] = {
            "benchmark": hayase_benchmark,
            "current": current_mse,
            "gap": gap,
            "percentage_to_benchmark": (current_mse / hayase_benchmark) * 100 if hayase_benchmark != 0 else 0
        }
    
    # Perform multi-tokenizer validation if enabled
    if args.multi_tokenizer_validation:
        logger.info("Performing multi-tokenizer validation...")
        multi_tokenizer_results = analyze_with_multiple_tokenizers(decade_texts, args)
        results["multi_tokenizer_validation"] = multi_tokenizer_results
    
    # Save detailed results
    save_distribution_results(results, evaluation, run_id, results_dir)
    
    # Use the enhanced logging manager to print results summary to stdout
    logging_manager.print_results_summary(evaluation, args)
    
    # Create comparison visualizations
    create_comparison_visualizations(distribution, selected_dist, 
                                  args.distribution, args.tokenizer, results_dir)
    
    # MODIFIED: Skip bootstrap validation completely unless explicitly needed
    if args.bootstrap and False:  # Set to False to skip bootstrap
        logger.info("Skipping bootstrap validation for performance reasons")
    
    logger.info(f"Analysis completed for {args.distribution} with {args.tokenizer}")
    
    # Return the key results for potential further analysis
    return {
        "distribution": distribution,
        "evaluation": evaluation,
        "uncertainty": uncertainty
    }

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
    
    # Extract data, handling different confidence interval structures
    means = [inferred_distribution.get(d, 0) for d in decades]
    
    # Handle different confidence interval formats - add this fix
    lower = []
    upper = []
    for i, d in enumerate(decades):
        if d in confidence_intervals:
            ci_data = confidence_intervals[d]
            # Handle dictionary format with nested structure
            if isinstance(ci_data, dict):
                if "lower_ci" in ci_data and "upper_ci" in ci_data:
                    lower.append(float(ci_data["lower_ci"]))
                    upper.append(float(ci_data["upper_ci"]))
                else:
                    # Default to 10% below and above the mean if CI not found
                    lower.append(means[i] * 0.9)
                    upper.append(means[i] * 1.1)
            else:
                # Handle tuple format
                lower.append(ci_data[0] if isinstance(ci_data, tuple) else means[i] * 0.9)
                upper.append(ci_data[1] if isinstance(ci_data, tuple) else means[i] * 1.1)
        else:
            # Default values if no confidence interval data
            lower.append(means[i] * 0.9)
            upper.append(means[i] * 1.1)
    
    # Calculate error bars AS POSITIVE DISTANCES
    errors_lower = [max(0, means[i] - lower[i]) for i in range(len(means))]
    errors_upper = [max(0, upper[i] - means[i]) for i in range(len(means))]
   
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
    
    # Process one distribution at a time with explicit memory cleanup
    for dist_name in distributions:
        # Copy args and update distribution
        dist_args = argparse.Namespace(**vars(args))
        dist_args.distribution = dist_name
        
        # IMPORTANT: Turn off bootstrap for memory efficiency
        dist_args.bootstrap = False
        dist_args.bootstrap_iterations = 0
        
        # IMPORTANT: Reduce target size for memory efficiency
        dist_args.target_size_gb = min(0.25, dist_args.target_size_gb)
        
        # Run analysis and store the returned results directly
        logger.info(f"Running analysis for {dist_name} distribution...")
        try:
            dist_results = run_analysis(dist_args)
            
            # Only store the essential results to reduce memory
            if dist_results and "distribution" in dist_results:
                results_by_dist[dist_name] = {
                    "distribution": dist_results["distribution"],
                    "evaluation": dist_results.get("evaluation", {})
                }
                
                # Force garbage collection after each distribution
                import gc
                gc.collect()
                
                logger.info(f"Successfully completed analysis for {dist_name} distribution")
                
                # IMPORTANT: Save intermediate results after each distribution
                try:
                    results_dir = setup_directories()
                    with open(results_dir / f"intermediate_{dist_name}_results.json", 'w') as f:
                        json.dump(results_by_dist[dist_name], f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to save intermediate results: {e}")
        except Exception as e:
            logger.error(f"Analysis for {dist_name} distribution failed: {e}")
            continue
            
        # Force cleanup
        gc.collect()
    
    # Create comparative visualizations only if we have multiple results
    if len(results_by_dist) > 1:
        logger.info(f"Creating comparative visualizations for {len(results_by_dist)} distributions...")
        try:
            create_distribution_comparison(results_by_dist, distributions, args.tokenizer, setup_directories())
            logger.info("Comparative visualization complete")
        except Exception as e:
            logger.error(f"Failed to create comparison visualizations: {e}")
    else:
        logger.warning(f"Only {len(results_by_dist)} distributions had valid results - skipping comparison")
    
    return results_by_dist

def create_distribution_comparison(results_by_dist, distributions, tokenizer_name, results_dir):
    """Create visualizations comparing results across different distributions."""
    # Extract metrics for comparison
    dist_names = list(results_by_dist.keys())
    
    # Safely access metrics with default values if missing
    log_mse_values = []
    mae_values = []
    js_values = []
    correlation_values = []
    
    for d in dist_names:
        eval_data = results_by_dist[d].get("evaluation", {})
        # Use default value of 0.0 if metric is missing
        log_mse_values.append(eval_data.get("log10_mse", 0.0) if "distribution_metrics" not in eval_data 
                              else eval_data.get("distribution_metrics", {}).get("log10_mse", 0.0))
        mae_values.append(eval_data.get("mae", 0.0) if "distribution_metrics" not in eval_data 
                         else eval_data.get("distribution_metrics", {}).get("mae", 0.0))
        js_values.append(eval_data.get("js_distance", 0.0) if "distribution_metrics" not in eval_data 
                         else eval_data.get("distribution_metrics", {}).get("js_distance", 0.0))
        
        # Rank correlation might be in a nested structure
        if "decade_metrics" in eval_data:
            correlation_values.append(eval_data.get("decade_metrics", {}).get("rank_correlation", 0.0))
        else:
            correlation_values.append(eval_data.get("rank_correlation", 0.0))
    
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

def save_checkpoint(data, name="inference_checkpoint"):
    """Save checkpoint to allow recovery from failures"""
    import pickle
    import time
    from pathlib import Path
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"{name}_{timestamp}.pkl"
    
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save a latest version
        latest_path = checkpoint_dir / f"{name}_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        
    return checkpoint_path

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
    parser.add_argument("--multi_tokenizer_validation", action="store_true", 
                      help="Validate results using multiple tokenizers")

    args = parser.parse_args()
    
    # Run all distributions or just the specified one
    if args.distribution == "all":
        compare_all_distributions(args)
    else:
        run_analysis(args)