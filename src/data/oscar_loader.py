# src/data/oscar_loader.py

import logging
import re
import random
from typing import Dict, List, Optional
from pathlib import Path
import os
from datetime import datetime
from tqdm import tqdm
import json

from datasets import load_dataset
from ..config import CACHE_DIR, RAW_DATA_DIR, TIME_PERIODS

logger = logging.getLogger(__name__)

class OscarLoader:
    """
    Loader for the Oscar corpus with temporal analysis support.
    """
    
    def __init__(self):
        """Initialize the loader with necessary paths."""
        self.cache_dir = CACHE_DIR / "oscar"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_data_dir = RAW_DATA_DIR / "oscar"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set which Oscar subset to use - "unshuffled_deduplicated_en" for English
        self.oscar_subset = "unshuffled_deduplicated_en"
        
        logger.info(f"Oscar loader initialized with subset: {self.oscar_subset}")
    
    def load_decade_samples(self, 
                          target_decades=None, 
                          texts_per_decade=1000) -> Dict[str, List[tuple]]:
        """
        Load samples of texts from Oscar with temporal attribution.
        
        Args:
            target_decades: List of decades to focus on
            texts_per_decade: Target number of texts per decade
            
        Returns:
            Dictionary mapping decades to lists of (text, source) tuples
        """
        if target_decades is None:
            target_decades = list(TIME_PERIODS.keys())
        
        # Focus on mid-century decades if not specified
        if not target_decades:
            target_decades = ["1930s", "1940s", "1950s", "1960s", "1970s", "1980s"]
        
        logger.info(f"Loading Oscar samples for decades: {target_decades}")
        
        # Check cache first
        cache_path = self.cache_dir / f"oscar_decade_texts_{texts_per_decade}.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    # Convert stored data back to tuples
                    cached_data = json.load(f)
                    decade_texts = {decade: [(text, source) for text, source in texts] 
                                  for decade, texts in cached_data.items()}
                    
                    # Filter to only requested decades
                    result = {decade: decade_texts.get(decade, []) 
                             for decade in target_decades}
                    
                    texts_count = sum(len(texts) for texts in result.values())
                    logger.info(f"Loaded {texts_count} Oscar texts from cache")
                    return result
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        # Initialize results dictionary
        decade_texts = {decade: [] for decade in target_decades}
        
        try:
            # Load Oscar dataset
            logger.info(f"Loading Oscar dataset from Hugging Face...")
            dataset = load_dataset("oscar", self.oscar_subset, streaming=True, split="train")
            
            # Process a sample of the dataset to find temporal information
            processed_count = 0
            assigned_count = 0
            
            # Determine how many samples to process per decade
            samples_to_process = texts_per_decade * 20  # Process more to ensure finding enough
            
            logger.info(f"Processing {samples_to_process} Oscar samples...")
            
            for i, example in enumerate(tqdm(dataset.take(samples_to_process), 
                                         total=samples_to_process,
                                         desc="Processing Oscar samples")):
                processed_count += 1
                
                # Extract text
                if 'text' not in example:
                    continue
                
                text = example['text']
                
                # Skip if too short
                if len(text) < 1000:
                    continue
                
                # Try to extract decade information from text
                decade = self._extract_decade_from_text(text, target_decades)
                
                if decade:
                    # Check if we need more texts for this decade
                    if len(decade_texts[decade]) < texts_per_decade:
                        decade_texts[decade].append((text, f"oscar_{i}"))
                        assigned_count += 1
                
                # Check if we have enough texts for all decades
                if all(len(texts) >= texts_per_decade for decade, texts in decade_texts.items() 
                       if decade in target_decades):
                    logger.info("Collected sufficient texts for all target decades")
                    break
                
                # Log progress periodically
                if i % 1000 == 0:
                    logger.info(f"Processed {i} samples, assigned {assigned_count} texts")
                    for decade in target_decades:
                        logger.info(f"  {decade}: {len(decade_texts[decade])} texts")
            
            # Log final stats
            logger.info(f"Oscar processing complete: processed {processed_count} records, assigned {assigned_count} texts")
            for decade in target_decades:
                logger.info(f"Final count for {decade}: {len(decade_texts[decade])} texts")
            
            # Cache the results
            try:
                # Convert tuples to lists for JSON serialization
                serializable_data = {decade: [[text, source] for text, source in texts] 
                                  for decade, texts in decade_texts.items()}
                
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f)
                logger.info(f"Cached Oscar decade texts to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache Oscar texts: {e}")
        
        except Exception as e:
            logger.error(f"Error loading Oscar dataset: {e}")
        
        return decade_texts
    
    def _extract_decade_from_text(self, text: str, target_decades: List[str]) -> Optional[str]:
        """
        Extract decade information from text content.
        
        This uses various heuristics to determine which decade a text belongs to:
        1. Look for explicit year mentions
        2. Look for decade names
        3. Look for period-specific vocabulary
        
        Args:
            text: Text content to analyze
            target_decades: List of decades to consider
            
        Returns:
            Detected decade or None
        """
        # 1. Look for explicit year mentions (19XX or 20XX)
        year_pattern = r'\b(19[3-9]\d|20[0-2]\d)\b'
        years = re.findall(year_pattern, text)
        
        # Convert found years to decades
        decade_counts = {}
        for year_str in years:
            try:
                year = int(year_str)
                for decade, (start_year, end_year) in TIME_PERIODS.items():
                    if start_year <= year <= end_year and decade in target_decades:
                        decade_counts[decade] = decade_counts.get(decade, 0) + 1
            except ValueError:
                continue
        
        # If we found years, use the most common decade
        if decade_counts:
            most_common_decade = max(decade_counts.items(), key=lambda x: x[1])[0]
            return most_common_decade
        
        # 2. Look for decade names ("1950s", "sixties", etc.)
        decade_patterns = {
            "1930s": [r'\b19[3]0s\b', r'\bthirties\b', r'\b30s\b'],
            "1940s": [r'\b19[4]0s\b', r'\bforties\b', r'\b40s\b'],
            "1950s": [r'\b19[5]0s\b', r'\bfifties\b', r'\b50s\b'],
            "1960s": [r'\b19[6]0s\b', r'\bsixties\b', r'\b60s\b'],
            "1970s": [r'\b19[7]0s\b', r'\bseventies\b', r'\b70s\b'],
            "1980s": [r'\b19[8]0s\b', r'\beighties\b', r'\b80s\b'],
            "1990s": [r'\b19[9]0s\b', r'\bnineties\b', r'\b90s\b'],
            "2000s": [r'\b20[0]0s\b', r'\btwo thousands\b', r'\b2000s\b'],
            "2010s": [r'\b20[1]0s\b', r'\btwenty tens\b', r'\b2010s\b'],
            "2020s": [r'\b20[2]0s\b', r'\btwenty twenties\b', r'\b2020s\b'],
        }
        
        for decade, patterns in decade_patterns.items():
            if decade in target_decades:
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        return decade
        
        # 3. Use period-specific vocabulary as a fallback
        decade_vocab = {
            "1930s": ["Great Depression", "New Deal", "Roosevelt", "Dust Bowl", "prohibition"],
            "1940s": ["World War II", "atomic bomb", "post-war", "GI Bill", "Roosevelt", "Truman"],
            "1950s": ["Cold War", "McCarthy", "Korean War", "Eisenhower", "civil rights", "rock and roll"],
            "1960s": ["Vietnam War", "Kennedy", "civil rights", "moon landing", "Beatles", "counterculture"],
            "1970s": ["Watergate", "oil crisis", "Nixon", "Carter", "disco", "inflation"],
            "1980s": ["Reagan", "Cold War", "Thatcher", "Gorbachev", "personal computer"],
            "1990s": ["Clinton", "dot-com", "Gulf War", "World Wide Web", "email"],
            "2000s": ["9/11", "Bush", "Iraq War", "Obama", "iPhone", "Facebook"],
            "2010s": ["Trump", "smartphone", "social media", "Brexit", "climate change"],
            "2020s": ["pandemic", "COVID", "coronavirus", "Biden", "TikTok"]
        }
        
        decade_scores = {decade: 0 for decade in target_decades if decade in decade_vocab}
        
        for decade, vocab in decade_vocab.items():
            if decade in target_decades:
                for term in vocab:
                    count = len(re.findall(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
                    decade_scores[decade] += count
        
        # Return the decade with the highest score, if any
        if decade_scores and max(decade_scores.values()) > 0:
            return max(decade_scores.items(), key=lambda x: x[1])[0]
        
        # If no decade detected, return None or random assignment
        if target_decades and random.random() < 0.1:  # 10% chance of random assignment
            return random.choice(target_decades)
        
        return None