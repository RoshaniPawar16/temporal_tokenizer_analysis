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
                      texts_per_decade=5000) -> Dict[str, List[tuple]]:
        """
        Load samples of texts from Oscar with improved temporal attribution.
        
        Args:
            target_decades: List of decades to focus on
            texts_per_decade: Target number of texts per decade
            
        Returns:
            Dictionary mapping decades to lists of (text, source) tuples
        """
        if target_decades is None:
            target_decades = list(TIME_PERIODS.keys())
        
        logger.info(f"Loading Oscar samples for decades: {target_decades}")
        
        # Check cache first with proper cache key that includes texts_per_decade
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
            # Try different approaches for loading Oscar, with fallbacks for different library versions
            logger.info(f"Loading Oscar dataset from Hugging Face...")
            
            # First try the modern approach with streaming for memory efficiency
            try:
                from datasets import load_dataset
                dataset = load_dataset(
                    "oscar", 
                    "unshuffled_deduplicated_en", 
                    split="train", 
                    streaming=True,  # Use streaming to handle large dataset
                    trust_remote_code=True
                )
                
                logger.info(f"Successfully loaded Oscar with streaming approach")
                modern_load_success = True
            except Exception as e1:
                logger.warning(f"Failed to load Oscar with streaming approach: {e1}")
                
                # Try without streaming
                try:
                    dataset = load_dataset(
                        "oscar", 
                        "unshuffled_deduplicated_en", 
                        split="train", 
                        trust_remote_code=True
                    )
                    logger.info(f"Successfully loaded Oscar without streaming")
                    modern_load_success = True
                except Exception as e2:
                    logger.warning(f"Failed to load Oscar without streaming: {e2}")
                    modern_load_success = False
                    
                    # Try older approach without trust_remote_code
                    try:
                        dataset = load_dataset("oscar", "unshuffled_deduplicated_en", split="train")
                        logger.info(f"Successfully loaded Oscar with older approach")
                        modern_load_success = True
                    except Exception as e3:
                        logger.warning(f"Failed to load Oscar with older approach: {e3}")
                        modern_load_success = False
            
            # If we couldn't load the dataset, try loading from local files or direct download
            if not modern_load_success:
                # Try loading from local files if available
                try:
                    local_path = self.raw_data_dir / "oscar_samples.json"
                    if local_path.exists():
                        with open(local_path, 'r', encoding='utf-8') as f:
                            local_samples = json.load(f)
                            
                        # Process samples
                        for sample in local_samples:
                            if "text" in sample:
                                text = sample["text"]
                                decade = self._extract_decade_from_text(text, target_decades)
                                if decade and decade in target_decades:
                                    decade_texts[decade].append((text, "oscar_local"))
                        
                        logger.info(f"Loaded {sum(len(texts) for texts in decade_texts.values())} texts from local Oscar samples")
                    else:
                        logger.warning("No local Oscar samples available")
                        
                        # Last resort - try to download a small sample directly
                        try:
                            import requests
                            url = "https://huggingface.co/datasets/oscar/resolve/main/unshuffled_deduplicated_en/train-sample.json"
                            response = requests.get(url, timeout=300)  # Increased timeout to 5 minutes
                            if response.status_code == 200:
                                samples = [json.loads(line) for line in response.text.splitlines() if line.strip()]
                                
                                for sample in samples:
                                    if "text" in sample:
                                        text = sample["text"]
                                        decade = self._extract_decade_from_text(text, target_decades)
                                        if decade and decade in target_decades:
                                            decade_texts[decade].append((text, "oscar_direct"))
                                
                                logger.info(f"Loaded {sum(len(texts) for texts in decade_texts.values())} texts from direct Oscar sample")
                            else:
                                logger.warning(f"Failed to download Oscar sample: HTTP {response.status_code}")
                        except Exception as e4:
                            logger.error(f"Failed to download Oscar sample: {e4}")
                except Exception as e5:
                    logger.error(f"Failed to load local Oscar samples: {e5}")
            
            # If we successfully loaded the dataset, process it
            if modern_load_success:
                # Process a sample of the dataset to find temporal information
                processed_count = 0
                assigned_count = 0
                
                # Use batch processing to avoid memory issues
                batch_size = 500  # Increased from 100 for efficiency
                samples_to_process = texts_per_decade * 20  # Increased from 10x to ensure we find enough
                
                logger.info(f"Processing {samples_to_process} Oscar samples in batches of {batch_size}...")
                
                for i in range(0, samples_to_process, batch_size):
                    current_batch = min(batch_size, samples_to_process - i)
                    if current_batch <= 0:
                        break
                    
                    try:
                        # For streaming dataset
                        if hasattr(dataset, 'take'):
                            batch = list(dataset.take(current_batch))
                        # For non-streaming dataset
                        else:
                            start_idx = i
                            end_idx = min(i + current_batch, len(dataset))
                            batch = dataset[start_idx:end_idx]
                            
                        processed_count += len(batch)
                        
                        for j, example in enumerate(batch):
                            if 'text' not in example:
                                continue
                            
                            text = example['text']
                            
                            # Relaxed minimum length requirement
                            if len(text) < 500:
                                continue
                            
                            # Try to extract decade information from text
                            decade = self._extract_decade_from_text(text, target_decades)
                            
                            if decade and decade in target_decades:
                                # Check if we need more texts for this decade
                                if len(decade_texts[decade]) < texts_per_decade * 2:  # Increased to 2x
                                    decade_texts[decade].append((text, f"oscar_{i+j}"))
                                    assigned_count += 1
                        
                        # Log progress
                        if (i // batch_size) % 5 == 0:
                            logger.info(f"Processed {processed_count} samples, assigned {assigned_count} texts")
                        
                        # Check if we have enough texts for all target decades
                        reached_targets = True
                        for decade in target_decades:
                            if decade in ["1930s", "1940s", "1950s", "1960s", "1970s", "1980s"]:  # Focus on mid-century
                                if len(decade_texts[decade]) < texts_per_decade:
                                    reached_targets = False
                                    break
                                    
                        if reached_targets:
                            logger.info("Collected sufficient texts for all target decades")
                            break
                    
                    except Exception as e:
                        logger.warning(f"Error processing batch {i // batch_size}: {e}")
                        continue
                
                # Log final stats
                logger.info(f"Oscar processing complete: processed {processed_count} records, assigned {assigned_count} texts")
                
                # If we still need more texts for some decades, try targeted approach using year pattern matching
                for decade in target_decades:
                    if len(decade_texts[decade]) < texts_per_decade // 2:
                        logger.info(f"Still need more texts for {decade}, trying targeted approach")
                        
                        # Generate decade-specific year patterns
                        decade_year = int(decade[:4])
                        year_patterns = [str(y) for y in range(decade_year, decade_year+10)]
                        
                        # Process more samples with explicit year filtering
                        added_count = 0
                        target_count = texts_per_decade - len(decade_texts[decade])
                        
                        for i in range(0, 20000, batch_size):  # Try more samples for sparse decades
                            if added_count >= target_count:
                                break
                                
                            try:
                                if hasattr(dataset, 'take'):
                                    additional_batch = list(dataset.take(batch_size))
                                else:
                                    start_idx = samples_to_process + i
                                    end_idx = min(start_idx + batch_size, len(dataset))
                                    additional_batch = dataset[start_idx:end_idx]
                                    
                                for example in additional_batch:
                                    if 'text' not in example:
                                        continue
                                        
                                    text = example['text']
                                    
                                    # Explicit year matching
                                    if any(f" {year} " in text or f"({year})" in text for year in year_patterns):
                                        decade_texts[decade].append((text, f"oscar_targeted_{added_count}"))
                                        added_count += 1
                                        
                                        if added_count >= target_count:
                                            break
                            except Exception as e:
                                logger.warning(f"Error in targeted processing for {decade}: {e}")
                                break
                        
                        logger.info(f"Added {added_count} additional texts for {decade} using targeted approach")
            
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
        
        # Final stats
        for decade in target_decades:
            logger.info(f"Final count for {decade}: {len(decade_texts[decade])} texts")
        
        return decade_texts

    def _extract_decade_from_text(self, text: str, target_decades: List[str]) -> Optional[str]:
        """
        Enhanced method to extract decade information from text content.
        Uses multiple strategies to determine time period.
        
        Args:
            text: Text content to analyze
            target_decades: List of decades to consider
            
        Returns:
            Detected decade or None
        """
        # 1. Look for explicit year mentions with expanded patterns
        year_patterns = [
            r'\b(19[0-9]{2}|20[0-2][0-9])\b',  # Full years (1900-2029)
            r'\bCopyright\s+[©]?\s*(?:c\.?)?\s*(19[0-9]{2}|20[0-2][0-9])\b',  # Copyright years
            r'\bPublished\s+(?:in\s+)?(19[0-9]{2}|20[0-2][0-9])\b',  # Publication years
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* [0-9]{1,2},? (19[0-9]{2}|20[0-2][0-9])\b',  # Full dates
            r'\b[0-9]{1,2}[/-][0-9]{1,2}[/-](19[0-9]{2}|20[0-2][0-9])\b'  # Numeric dates
        ]
        
        years = []
        for pattern in year_patterns:
            years.extend(re.findall(pattern, text))
        
        # Extract all potential years
        year_candidates = []
        for year_match in years:
            if isinstance(year_match, tuple):
                # Extract from tuple if needed (for regex groups)
                for item in year_match:
                    if isinstance(item, str) and re.match(r'^(19|20)\d{2}$', item):
                        year_candidates.append(int(item))
            elif isinstance(year_match, str) and re.match(r'^(19|20)\d{2}$', year_match):
                # Direct string match
                year_candidates.append(int(year_match))
        
        # Convert found years to decades with a weighted approach
        decade_weights = {}
        
        for year in year_candidates:
            for decade, (start_year, end_year) in TIME_PERIODS.items():
                if start_year <= year <= end_year and decade in target_decades:
                    # Weight by count and position in text (years mentioned earlier are more likely to be relevant)
                    # Find position in text
                    position_weight = 1.0
                    year_str = str(year)
                    year_pos = text.find(year_str)
                    if year_pos > 0:
                        # Higher weight for years mentioned in first 25% of text (likely publication info)
                        if year_pos < len(text) * 0.25:
                            position_weight = 2.0
                    
                    decade_weights[decade] = decade_weights.get(decade, 0) + position_weight
        
        # If we found years, use the most common decade
        if decade_weights:
            most_common_decade = max(decade_weights.items(), key=lambda x: x[1])[0]
            return most_common_decade
        
        # 2. Look for decade names with expanded patterns
        decade_patterns = {
            "1930s": [r'\b19[3]0s\b', r'\bthirties\b', r'\b30s\b', r'\bthirty[- ]?(?:ies|s)\b'],
            "1940s": [r'\b19[4]0s\b', r'\bforties\b', r'\b40s\b', r'\bforty[- ]?(?:ies|s)\b'],
            "1950s": [r'\b19[5]0s\b', r'\bfifties\b', r'\b50s\b', r'\bfifty[- ]?(?:ies|s)\b'],
            "1960s": [r'\b19[6]0s\b', r'\bsixties\b', r'\b60s\b', r'\bsixty[- ]?(?:ies|s)\b'],
            "1970s": [r'\b19[7]0s\b', r'\bseventies\b', r'\b70s\b', r'\bseventy[- ]?(?:ies|s)\b'],
            "1980s": [r'\b19[8]0s\b', r'\beighties\b', r'\b80s\b', r'\beighty[- ]?(?:ies|s)\b'],
            "1990s": [r'\b19[9]0s\b', r'\bnineties\b', r'\b90s\b', r'\bninety[- ]?(?:ies|s)\b'],
            "2000s": [r'\b20[0]0s\b', r'\btwo thousands\b', r'\b2000s\b', r'\b00s\b', r'\boughts\b'],
            "2010s": [r'\b20[1]0s\b', r'\btwenty tens\b', r'\b2010s\b', r'\b10s\b', r'\btens\b'],
            "2020s": [r'\b20[2]0s\b', r'\btwenty twenties\b', r'\b2020s\b', r'\b20s\b']
        }
        
        for decade, patterns in decade_patterns.items():
            if decade in target_decades:
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        return decade
        
        # 3. Use historical/cultural context clues
        decade_markers = {
            "1930s": ["great depression", "new deal", "roosevelt", "dust bowl", "prohibition", "nazi germany", "hoover"],
            "1940s": ["world war ii", "atomic bomb", "pearl harbor", "holocaust", "truman", "rosie the riveter", "radar"],
            "1950s": ["cold war", "korean war", "mccarthyism", "eisenhower", "civil rights", "rock and roll", "sputnik"],
            "1960s": ["vietnam war", "kennedy", "civil rights movement", "moon landing", "beatles", "woodstock", "martin luther king"],
            "1970s": ["watergate", "oil crisis", "nixon", "carter", "disco", "punk rock", "star wars", "vietnam"],
            "1980s": ["reagan", "thatcher", "cold war", "berlin wall", "personal computer", "mtv", "aids", "chernobyl"],
            "1990s": ["clinton", "gulf war", "world wide web", "dot-com", "internet", "napster", "y2k", "grunge"],
            "2000s": ["9/11", "iraq war", "bush", "obama", "iphone", "facebook", "myspace", "hurricane katrina"],
            "2010s": ["trump", "brexit", "social media", "instagram", "uber", "arab spring", "occupy", "smartphone"],
            "2020s": ["covid", "pandemic", "lockdown", "biden", "tiktok", "ukraine war", "black lives matter", "zoom"]
        }
        
        decade_scores = {decade: 0 for decade in target_decades if decade in decade_markers}
        
        for decade, markers in decade_markers.items():
            if decade in target_decades:
                for marker in markers:
                    # Look for exact phrase matches with word boundaries
                    count = len(re.findall(r'\b' + re.escape(marker) + r'\b', text.lower()))
                    decade_scores[decade] += count * 2  # Weight these stronger
        
        # Return the decade with the highest score, if significant
        max_score = max(decade_scores.values()) if decade_scores else 0
        if max_score > 1:  # Threshold to ensure it's not just random mentions
            return max(decade_scores.items(), key=lambda x: x[1])[0]
        
        # Don't randomly assign decades with no evidence - better to return None
        # than incorrect decade assignment
        return None