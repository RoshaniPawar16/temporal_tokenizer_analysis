# src/data/fill_missing_decades.py

"""
Fill Missing Decades

This module provides functionality to fill gaps in the temporal dataset,
ensuring balanced representation across all decades.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from ..config import (
    PROCESSED_DATA_DIR,
    TIME_PERIODS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fill_missing_decades(num_texts: int = 20):
    """
    Fill missing decades in the dataset with historically plausible text.
    
    Args:
        num_texts: Number of texts to generate for each missing decade
    """
    logger.info(f"Filling missing decades with {num_texts} texts per decade")
    
    dataset_dir = PROCESSED_DATA_DIR / "temporal_dataset"
    metadata_path = dataset_dir / "dataset_metadata.json"
    
    # Check if metadata exists
    if not metadata_path.exists():
        logger.error("Dataset metadata not found. Please build dataset first.")
        return
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return
    
    # Identify missing or sparse decades
    missing_decades = []
    sparse_decades = []
    
    for decade in TIME_PERIODS.keys():
        decade_dir = dataset_dir / decade
        
        if not decade_dir.exists() or decade not in metadata.get("decades", {}):
            logger.warning(f"Decade {decade} is missing completely")
            missing_decades.append(decade)
        elif metadata["decades"][decade]["total"] < num_texts:
            shortfall = num_texts - metadata["decades"][decade]["total"]
            logger.warning(f"Decade {decade} is sparse: only {metadata['decades'][decade]['total']} texts")
            sparse_decades.append((decade, shortfall))
    
    if not missing_decades and not sparse_decades:
        logger.info("All decades have sufficient data. No filling required.")
        return
    
    # Process missing decades
    for decade in missing_decades:
        logger.info(f"Creating directory and synthetic data for missing decade: {decade}")
        
        # Create decade directory
        decade_dir = dataset_dir / decade
        decade_dir.mkdir(exist_ok=True)
        
        # Generate synthetic texts
        synthetic_texts = _generate_synthetic_texts(decade, num_texts)
        
        # Save texts and update metadata
        _save_synthetic_decade(decade, synthetic_texts, dataset_dir, metadata)
    
    # Process sparse decades
    for decade, shortfall in sparse_decades:
        logger.info(f"Adding {shortfall} synthetic texts to sparse decade: {decade}")
        
        # Generate additional synthetic texts
        synthetic_texts = _generate_synthetic_texts(decade, shortfall)
        
        # Add to existing decade data
        _add_to_existing_decade(decade, synthetic_texts, dataset_dir, metadata)
    
    # Save updated metadata
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("Updated dataset metadata with filled decades")
    except Exception as e:
        logger.error(f"Failed to save updated metadata: {e}")

def _generate_synthetic_texts(decade: str, count: int) -> List[Tuple[str, str]]:
    """
    Generate synthetic texts for a specific decade.
    
    Args:
        decade: Target decade
        count: Number of texts to generate
        
    Returns:
        List of (text, source) tuples
    """
    start_year, end_year = TIME_PERIODS[decade]
    synthetic_texts = []
    
    # Define decade-specific vocabulary and themes
    decade_vocab = {
        "1850s": ["railway", "industrial", "Victorian", "telegraph", "Empire"],
        "1860s": ["telegraph", "Civil War", "expedition", "workhouse", "colonies"],
        "1870s": ["phonograph", "telephone", "typewriter", "electric light"],
        "1880s": ["electricity", "modern", "scientific", "industrial"],
        "1890s": ["bicycle", "horseless carriage", "cinematograph", "photography"],
        "1900s": ["automobile", "aeroplane", "wireless", "gramophone"],
        "1910s": ["Great War", "aeroplane", "wireless", "cinema"],
        "1920s": ["wireless", "radio", "cinema", "automobile", "aeroplane"],
        "1930s": ["depression", "radio", "cinema", "modern", "automobile"],
        "1940s": ["war", "atomic", "radar", "radio", "modern"],
        "1950s": ["atomic", "television", "modern", "electric", "radio"],
        "1960s": ["television", "modern", "electronic", "space", "computer"],
        "1970s": ["computer", "electronic", "modern", "digital", "space"],
        "1980s": ["computer", "electronic", "modern", "digital", "personal"],
        "1990s": ["internet", "computer", "digital", "modern", "global"],
        "2000s": ["internet", "digital", "mobile", "online", "global"],
        "2010s": ["smartphone", "social media", "digital", "online", "app"],
        "2020s": ["pandemic", "digital", "AI", "remote", "virtual"]
    }
    
    decade_themes = {
        "1850s": ["Industrial progress", "Class divisions", "Empire"],
        "1860s": ["American Civil War", "Colonial expansion", "Literary societies"],
        "1870s": ["Scientific discovery", "Technological progress", "Imperial expansion"],
        "1880s": ["Social reform", "Industrial development", "Scientific advances"],
        "1890s": ["Modern innovations", "Social questions", "Imperial concerns"],
        "1900s": ["New century", "Social reform", "Imperial politics"],
        "1910s": ["The Great War", "Social change", "Political movements"],
        "1920s": ["Post-war society", "Modern entertainment", "Economic growth"],
        "1930s": ["Economic depression", "Political tensions", "Social welfare"],
        "1940s": ["World War II", "Post-war planning", "Atomic age"],
        "1950s": ["Post-war prosperity", "Cold War tensions", "Cultural changes"],
        "1960s": ["Cultural revolution", "Political change", "Space exploration"],
        "1970s": ["Economic challenges", "Technological advances", "Cultural shifts"],
        "1980s": ["Digital revolution", "Economic policies", "Cultural developments"],
        "1990s": ["Internet emergence", "Global politics", "Cultural trends"],
        "2000s": ["Digital transformation", "Terrorism", "Economic crises"],
        "2010s": ["Social media", "Smartphone era", "Political polarization"],
        "2020s": ["Pandemic", "Remote work", "AI advancement"]
    }
    
    # Get decade-specific elements or use defaults
    vocab = decade_vocab.get(decade, ["modern", "society", "development"])
    themes = decade_themes.get(decade, ["Society", "Technology", "Culture"])
    
    # Generate each synthetic text
    for i in range(count):
        # Pick a year from the decade
        year = random.randint(start_year, end_year)
        
        # Pick a theme and style
        theme = random.choice(themes)
        
        # Generate text with period-appropriate vocabulary and style
        text = f"[Synthetic text representing {decade}] "
        text += f"In {year}, {theme.lower()} was undergoing significant transformation. "
        
        # Add several paragraphs of synthetic content
        paragraphs = []
        
        # Main theme paragraph
        paragraphs.append(f"The development of {random.choice(vocab)} had profound implications for society. "
                         f"Many observers noted how it was changing everyday life in unexpected ways.")
        
        # Historical context paragraph
        paragraphs.append(f"This period was characterized by rapid changes in {theme.lower()}. "
                         f"The influence of {random.choice(vocab)} and {random.choice(vocab)} "
                         f"created new possibilities, while also generating significant debate.")
        
        # Add several more paragraphs with period vocabulary
        for _ in range(3):
            para = f"The question of how {random.choice(vocab)} relates to {theme.lower()} "
            para += f"was of particular interest during this time. "
            para += f"Some argued that {random.choice(vocab)} represented progress, "
            para += f"while others expressed concern about its broader implications."
            paragraphs.append(para)
        
        # Combine paragraphs
        text += " ".join(paragraphs)
        
        synthetic_texts.append((text, "synthetic"))
    
    return synthetic_texts

def _save_synthetic_decade(decade: str, synthetic_texts: List[Tuple[str, str]], 
                         dataset_dir: Path, metadata: Dict) -> None:
    """
    Save synthetic texts for a completely missing decade.
    
    Args:
        decade: Target decade
        synthetic_texts: List of (text, source) tuples
        dataset_dir: Dataset directory path
        metadata: Dataset metadata dictionary to update
    """
    decade_dir = dataset_dir / decade
    
    # Create metadata CSV entries
    rows = []
    
    # Process each synthetic text
    for i, (text, source) in enumerate(synthetic_texts):
        text_id = f"{decade}_synth_{i:04d}"
        text_path = decade_dir / f"{text_id}.txt"
        
        # Save text file
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Add metadata row
        rows.append({
            "id": text_id,
            "decade": decade,
            "source": source,
            "length": len(text),
            "path": str(text_path.relative_to(dataset_dir))
        })
    
    # Create metadata CSV
    import pandas as pd
    metadata_df = pd.DataFrame(rows)
    metadata_df.to_csv(decade_dir / "metadata.csv", index=False)
    
    # Update global metadata
    if "decades" not in metadata:
        metadata["decades"] = {}
    
    metadata["decades"][decade] = {
        "total": len(synthetic_texts),
        "british_library": 0,
        "gutenberg": 0,
        "synthetic": len(synthetic_texts)
    }
    
    metadata["total_texts"] = metadata.get("total_texts", 0) + len(synthetic_texts)
    
    logger.info(f"Created {len(synthetic_texts)} synthetic texts for {decade}")

def _add_to_existing_decade(decade: str, synthetic_texts: List[Tuple[str, str]], 
                          dataset_dir: Path, metadata: Dict) -> None:
    """
    Add synthetic texts to an existing but sparse decade.
    
    Args:
        decade: Target decade
        synthetic_texts: List of (text, source) tuples
        dataset_dir: Dataset directory path
        metadata: Dataset metadata dictionary to update
    """
    decade_dir = dataset_dir / decade
    metadata_csv = decade_dir / "metadata.csv"
    
    # Load existing metadata
    import pandas as pd
    try:
        metadata_df = pd.read_csv(metadata_csv)
        start_index = len(metadata_df)
    except Exception:
        metadata_df = pd.DataFrame(columns=["id", "decade", "source", "length", "path"])
        start_index = 0
    
    # New rows for synthetic texts
    new_rows = []
    
    # Process each synthetic text
    for i, (text, source) in enumerate(synthetic_texts):
        text_id = f"{decade}_synth_{start_index + i:04d}"
        text_path = decade_dir / f"{text_id}.txt"
        
        # Save text file
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Add metadata row
        new_rows.append({
            "id": text_id,
            "decade": decade,
            "source": source,
            "length": len(text),
            "path": str(text_path.relative_to(dataset_dir))
        })
    
    # Update metadata CSV
    new_rows_df = pd.DataFrame(new_rows)
    updated_df = pd.concat([metadata_df, new_rows_df], ignore_index=True)
    updated_df.to_csv(metadata_csv, index=False)
    
    # Update global metadata
    metadata["decades"][decade]["total"] += len(synthetic_texts)
    metadata["decades"][decade]["synthetic"] = metadata["decades"][decade].get("synthetic", 0) + len(synthetic_texts)
    metadata["total_texts"] += len(synthetic_texts)
    
    logger.info(f"Added {len(synthetic_texts)} synthetic texts to {decade}")

if __name__ == "__main__":
    # Allow command-line specification of text count
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    fill_missing_decades(num_texts=count)