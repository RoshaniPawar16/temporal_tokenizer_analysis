# src/config.py

"""
Project Configuration

This module defines the configuration for the temporal tokenizer analysis project.
It handles directory structure, analysis parameters, and dataset configurations.
The configuration is designed to support analysis of temporal patterns in language
model tokenizers across different time periods.

Key components:
- Project directory structure and path management
- Temporal period definitions
- Dataset sampling and processing parameters
- Analysis configuration
- Data source settings
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.
    This ensures consistent path resolution across different
    execution contexts.
    """
    return Path(__file__).resolve().parent.parent

# Project directory structure setup
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# Define all project directories that need to be created
PROJECT_DIRS: List[Path] = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    CACHE_DIR,
    LOGS_DIR,
    # Data source specific directories
    RAW_DATA_DIR / "british_library",
    RAW_DATA_DIR / "gutenberg",
    PROCESSED_DATA_DIR / "temporal_dataset",
    CACHE_DIR / "tokenizers",
]

# Create all necessary directories
for dir_path in PROJECT_DIRS:
    dir_path.mkdir(parents=True, exist_ok=True)

# Temporal analysis configuration
# Maps decade identifiers to their start and end years
TIME_PERIODS: Dict[str, Tuple[int, int]] = {
    "1850s": (1850, 1859),
    "1860s": (1860, 1869),
    "1870s": (1870, 1879),
    "1880s": (1880, 1889),
    "1890s": (1890, 1899),
    "1900s": (1900, 1909),  
    "1910s": (1910, 1919),
    "1920s": (1920, 1929),
    "1930s": (1930, 1939),
    "1940s": (1940, 1949),
    "1950s": (1950, 1959),
    "1960s": (1960, 1969),
    "1970s": (1970, 1979),
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2029),
}

# Dataset configuration
DATASET_CONFIG = {
    "sampling": {
        "texts_per_decade": 50,     # Target number of texts per decade
        "min_text_length": 1000,    # Minimum text length in characters
        "max_text_length": 10000,   # Maximum text length in characters
        "chunk_size": 5000,         # Size of text chunks for processing
        "overlap": 200,             # Overlap between chunks
    },
    "processing": {
        "clean_html": True,         # Remove HTML tags
        "normalize_whitespace": True,# Standardize spacing
        "fix_unicode": True,        # Handle unicode normalization
        "max_line_length": 80,      # Maximum line length for formatted text
    },
    "validation": {
        "split_ratio": 0.2,         # Validation set size
        "min_samples": 10,          # Minimum samples per category
        "random_seed": 42,          # Random seed for reproducibility
    }
}

# Analysis configuration for tokenizer investigation
ANALYSIS_CONFIG = {
    "tokenizer": {
        "merge_rules_sample": 3000, # Number of merge rules to analyze
        "vocab_size": 30000,        # Target vocabulary size
        "min_frequency": 2,         # Minimum token frequency
        "special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "sampling": {
        "sample_size": 1000,        # Number of samples for analysis
        "random_seed": 42,          # Random seed for reproducibility
        "stratify": True,           # Whether to stratify by decade
    },
    "metrics": {
        "track_merge_order": True,  # Track order of merge operations
        "save_frequencies": True,   # Save token frequency distributions
        "compute_efficiency": True, # Calculate tokenization efficiency
    }
}

# Data source specific configurations
DATA_SOURCES = {
    "british_library": {
        "collection_id": "britishlibrary",
        "cache_dir": CACHE_DIR / "british_library",
        "request_delay": 2.0,       # Delay between API requests
        "max_retries": 5,           # Maximum retry attempts
        "timeout": 30,              # Request timeout in seconds
        "batch_size": 50,           # Batch size for processing
    },
    "gutenberg": {
        "cache_dir": CACHE_DIR / "gutenberg",
        "sample_size": 20,          # Books per decade
        "languages": ["en"],        # Language filter
        "formats": ["txt"],         # Acceptable formats
    },
    "arxiv": {
        "cache_dir": CACHE_DIR / "arxiv",
        "categories": ["cs", "math", "physics"],
        "batch_size": 100,          # Papers per request
        "max_results": 1000,        # Maximum results to fetch
    }
}

# Model configuration for analysis
MODEL_CONFIG = {
    "reference_models": [
        "gpt2",
        "bert-base-uncased",
        "roberta-base",
        "t5-base"
    ],
    "tokenizer_params": {
        "add_prefix_space": True,
        "trim_offsets": True,
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_epochs": 10,
        "early_stopping": True,
    }
}

def verify_project_setup() -> bool:
    """
    Verify all required project directories exist and are writable.
    Also checks for basic configuration validity.
    
    Returns:
        bool: True if setup is valid, False otherwise
    """
    try:
        # Check directories exist and are writable
        for dir_path in PROJECT_DIRS:
            if not dir_path.exists():
                logger.error(f"Directory does not exist: {dir_path}")
                return False
            if not os.access(dir_path, os.W_OK):
                logger.error(f"Directory not writable: {dir_path}")
                return False
        
        # Verify time periods are properly ordered
        years = [(start, end) for start, end in TIME_PERIODS.values()]
        for i in range(len(years)-1):
            if years[i][1] >= years[i+1][0]:
                logger.error("Time periods overlap or are not properly ordered")
                return False
        
        logger.info("Project setup verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"Project setup verification failed: {e}")
        return False

# Verify setup on import
if not verify_project_setup():
    raise RuntimeError("Project setup verification failed. Check logs for details.")