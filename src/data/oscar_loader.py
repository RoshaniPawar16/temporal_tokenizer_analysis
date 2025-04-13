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
            # Try different approaches for loading Oscar, with fallbacks for different library versions
            logger.info(f"Loading Oscar dataset from Hugging Face...")
            
            # First try the modern approach
            try:
                from datasets import load_dataset
                dataset = load_dataset("oscar", "unshuffled_deduplicated_en", split="train", trust_remote_code=True)
                
                logger.info(f"Successfully loaded Oscar with modern approach")
                modern_load_success = True
            except Exception as e1:
                logger.warning(f"Failed to load Oscar with modern approach: {e1}")
                modern_load_success = False
                
                # Try older approach without trust_remote_code
                try:
                    dataset = load_dataset("oscar", "unshuffled_deduplicated_en", split="train", trust_remote_code=True)
                    logger.info(f"Successfully loaded Oscar with older approach")
                except Exception as e2:
                    logger.warning(f"Failed to load Oscar with older approach: {e2}")
                    
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
                                response = requests.get(url, timeout=120)
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
                            except Exception as e3:
                                logger.error(f"Failed to download Oscar sample: {e3}")
                    except Exception as e3:
                        logger.error(f"Failed to load local Oscar samples: {e3}")
            
            # If we successfully loaded the dataset, process it
            if modern_load_success:
                # Process a sample of the dataset to find temporal information
                processed_count = 0
                assigned_count = 0
                
                # Use batch processing to avoid memory issues
                batch_size = 100
                samples_to_process = texts_per_decade * 10  # 10x to ensure we find enough
                
                logger.info(f"Processing {samples_to_process} Oscar samples in batches...")
                
                for i in range(0, samples_to_process, batch_size):
                    current_batch = min(batch_size, samples_to_process - i)
                    if current_batch <= 0:
                        break
                        
                    batch = list(dataset.take(current_batch))
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
                        
                        if decade:
                            # Check if we need more texts for this decade
                            if len(decade_texts[decade]) < texts_per_decade * 1.2:
                                decade_texts[decade].append((text, f"oscar_{i+j}"))
                                assigned_count += 1
                    
                    # Log progress
                    if (i // batch_size) % 10 == 0:
                        logger.info(f"Processed {processed_count} samples, assigned {assigned_count} texts")
                    
                    # Check if we have enough texts for all decades
                    if all(len(texts) >= texts_per_decade for decade, texts in decade_texts.items() 
                        if decade in target_decades):
                        logger.info("Collected sufficient texts for all target decades")
                        break
                
                # Log final stats
                logger.info(f"Oscar processing complete: processed {processed_count} records, assigned {assigned_count} texts")
            
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
                    if re.match(r'^(19|20)\d{2}$', str(item)):
                        year_candidates.append(int(item))
            else:
                # Direct string match
                if re.match(r'^(19|20)\d{2}$', str(year_match)):
                    year_candidates.append(int(year_match))
        
        # Convert found years to decades with a weighted approach
        decade_weights = {}
        
        for year in year_candidates:
            for decade, (start_year, end_year) in TIME_PERIODS.items():
                if start_year <= year <= end_year and decade in target_decades:
                    # Weight by count and position in text (years mentioned earlier are more likely to be relevant)
                    decade_weights[decade] = decade_weights.get(decade, 0) + 1
        
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
        
        decade_scores = {decade: 0 for decade in target_decades}
        
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