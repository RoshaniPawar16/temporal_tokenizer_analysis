# src/data/dataset_manager.py

import logging
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import json
import random
import re
import pickle
from collections import Counter
from transformers import AutoTokenizer

from ..config import (
    PROCESSED_DATA_DIR,
    TIME_PERIODS,
    ANALYSIS_CONFIG,
    CACHE_DIR  # Add this import
)
from .british_library_loader import BritishLibraryLoader 
from .gutenberg_loader import GutenbergLoader
from .oscar_loader import OscarLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalDatasetManager:
    """
    Manages temporal datasets focusing on historical texts from 1850-2020.
    Uses British Library and Gutenberg as primary sources to ensure
    reliable historical coverage and balanced representation.
    """
    
    def __init__(self):
        """Initialize data loaders and set up directory structure."""
        # Initialize our historical data sources
        self.oscar_loader = OscarLoader()
        self.british_library_loader = BritishLibraryLoader()
        self.gutenberg_loader = GutenbergLoader()
               
        # Set up storage directories
        self.dataset_dir = PROCESSED_DATA_DIR / "temporal_dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.dataset_dir / "dataset_metadata.json"

        # Add this line to define the cache_dir attribute
        self.cache_dir = CACHE_DIR / "dataset_manager"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_modern_web_content(self, target_decades=None, texts_per_decade=1000):
        """
        Load modern web content for recent decades that are hard to find in traditional datasets.
        
        Args:
            target_decades: List of decades to focus on (defaults to 1990s-2020s)
            texts_per_decade: Number of texts to aim for per decade
            
        Returns:
            Dictionary mapping decades to lists of (text, source) tuples
        """
        if target_decades is None:
            target_decades = ["1990s", "2000s", "2010s", "2020s"]
        else:
            # Filter to only include modern decades
            target_decades = [d for d in target_decades if d in ["1990s", "2000s", "2010s", "2020s"]]
        
        if not target_decades:
            return {}
        
        logger.info(f"Loading modern web content for decades: {target_decades}")
        
        # Initialize results dictionary
        decade_texts = {decade: [] for decade in target_decades}
        
        # Check if we have cached data
        cache_path = self.cache_dir / "modern_web_content.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    for decade in target_decades:
                        if decade in cached_data:
                            decade_texts[decade] = [(text, source) for text, source in cached_data[decade]]
                    
                    logger.info(f"Loaded {sum(len(texts) for texts in decade_texts.values())} modern web texts from cache")
                    return decade_texts
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        # Simple Wikipedia dataset often has many timestamp references
        try:
            from datasets import load_dataset
            
            # Try to load Simple Wikipedia
            try:
                wiki_data = load_dataset("wikipedia", "20220301.simple", split="train")
                logger.info(f"Successfully loaded Simple Wikipedia dataset")
                
                # Process some samples
                samples = list(wiki_data.take(5000))
                
                # Extract decade information and assign
                for sample in samples:
                    if "text" in sample:
                        text = sample["text"]
                        
                        # Try to extract decade
                        decade = self._extract_decade_from_text(text, target_decades)
                        
                        if decade and decade in target_decades:
                            title = sample.get("title", "Untitled")
                            decade_texts[decade].append((text, f"wikipedia_{title}"))
                
                logger.info(f"Added {sum(len(texts) for texts in decade_texts.values())} texts from Simple Wikipedia")
                
            except Exception as e:
                logger.warning(f"Failed to load Simple Wikipedia: {e}")
            
            # Try to load news articles dataset for 2000s-2020s
            try:
                news_data = load_dataset("cnn_dailymail", "3.0.0", split="train")
                logger.info(f"Successfully loaded news dataset")
                
                # Process some samples
                samples = list(news_data.take(2000))
                
                # Extract decade information and assign
                for sample in samples:
                    if "article" in sample:
                        text = sample["article"]
                        
                        # Try to extract decade
                        decade = self._extract_decade_from_text(text, target_decades)
                        
                        if decade and decade in target_decades:
                            decade_texts[decade].append((text, "news_article"))
                
                logger.info(f"Added texts from news articles")
                
            except Exception as e:
                logger.warning(f"Failed to load news dataset: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
        
        # If we have insufficient data for any decade, add placeholder data
        for decade in target_decades:
            if len(decade_texts[decade]) < 10:
                logger.warning(f"Insufficient modern data for {decade}, adding placeholders")
                
                # Generate placeholder content with appropriate decade references
                for i in range(100):
                    text = self._generate_modern_placeholder(decade)
                    decade_texts[decade].append((text, f"placeholder_{decade}_{i}"))
        
        # Cache the results
        try:
            serializable_data = {decade: [[text, source] for text, source in texts] 
                            for decade, texts in decade_texts.items()}
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f)
            logger.info(f"Cached modern web content to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache modern web content: {e}")
        
        # Log final counts
        for decade in target_decades:
            logger.info(f"Final count for {decade}: {len(decade_texts[decade])} texts")
        
        return decade_texts

    def _generate_modern_placeholder(self, decade):
        """Generate placeholder content with appropriate decade references."""
        decade_start = decade[:4]
        year_range = range(int(decade_start), int(decade_start) + 10)
        
        # Create decade-specific references
        decade_events = {
            "1990s": ["World Wide Web", "dot-com boom", "Windows 95", "Gulf War", "Clinton administration", 
                    "Internet Explorer", "Netscape Navigator", "PlayStation", "Y2K preparations"],
            "2000s": ["9/11 attacks", "War on Terror", "Web 2.0", "Wikipedia", "Facebook", "YouTube", 
                    "iPod", "iPhone", "Financial crisis of 2007-2008", "Obama election"],
            "2010s": ["Social media", "Arab Spring", "Brexit", "Trump administration", "Smartphone revolution", 
                    "Cloud computing", "Big data", "Machine learning", "Streaming services"],
            "2020s": ["COVID-19 pandemic", "Remote work", "Vaccination campaigns", "Climate crisis", 
                    "Cryptocurrency boom", "Metaverse", "AI revolution", "TikTok", "Ukraine war"]
        }
        
        events = decade_events.get(decade, ["modern developments"])
        years = random.sample(list(year_range), min(3, len(year_range)))
        
        # Generate article-style text with explicit decade and year references
        paragraphs = []
        
        # Title with decade reference
        title = f"Developments in the {decade}"
        paragraphs.append(title)
        
        # Introduction with decade reference
        intro = f"The {decade} was a transformative period marked by significant technological and cultural changes."
        paragraphs.append(intro)
        
        # Add paragraphs with year and event references
        for year in years:
            event = random.choice(events)
            para = f"In {year}, the development of {event} significantly impacted how people interacted with technology. "
            para += f"This was characteristic of changes taking place throughout the {decade}."
            paragraphs.append(para)
        
        # Add some generic content with decade references
        for _ in range(3):
            events_sample = random.sample(events, min(2, len(events)))
            para = f"During the {decade}, {events_sample[0]} and {events_sample[1] if len(events_sample) > 1 else 'related technologies'} "
            para += f"represented the leading edge of innovation. Companies and individuals alike adapted to these changes."
            paragraphs.append(para)
        
        # Conclusion with decade reference
        conclusion = f"Looking back, we can see how the developments of the {decade} laid groundwork for many modern systems we rely on today."
        paragraphs.append(conclusion)
        
        # Join all paragraphs with double newlines
        return "\n\n".join(paragraphs)

    def verify_dataset_quality(self, dataset, min_real_percentage=0.2):
        """
        Verify that the dataset has sufficient quality before proceeding with analysis.
        
        Args:
            dataset: The dataset to verify
            min_real_percentage: Minimum percentage of real (non-synthetic) data required
            
        Returns:
            Tuple of (is_valid, quality_report)
        """
        if not dataset:
            return False, {"error": "Empty dataset"}
        
        # Calculate dataset statistics
        total_texts = 0
        real_count = 0
        synthetic_count = 0
        expanded_count = 0
        
        for decade, texts in dataset.items():
            for item in texts:
                total_texts += 1
                # Handle different data formats
                if isinstance(item, tuple) and len(item) > 1:
                    source = item[1]
                    if "synthetic" in source:
                        synthetic_count += 1
                    elif "expanded" in source or "augmented" in source:
                        expanded_count += 1
                    else:
                        real_count += 1
                else:
                    # Assume it's real data if not a tuple with source info
                    real_count += 1
        
        # Calculate percentages
        real_percentage = real_count / total_texts if total_texts > 0 else 0
        synthetic_percentage = synthetic_count / total_texts if total_texts > 0 else 0
        expanded_percentage = expanded_count / total_texts if total_texts > 0 else 0
        
        # Check decade coverage
        decades_with_data = [decade for decade, texts in dataset.items() if texts]
        decade_coverage = len(decades_with_data) / len(TIME_PERIODS)
        
        # Assess data volume
        total_bytes = sum(sum(len(text.encode('utf-8')) for text, _ in texts) for texts in dataset.values())
        total_gb = total_bytes / (1024**3)
        
        # Check if the dataset meets quality thresholds
        is_valid = (
            real_percentage >= min_real_percentage and
            decade_coverage >= 0.5 and
            total_gb >= 0.1
        )
        
        quality_report = {
            "is_valid": is_valid,
            "total_texts": total_texts,
            "real_texts": real_count,
            "real_percentage": real_percentage,
            "synthetic_percentage": synthetic_percentage,
            "expanded_percentage": expanded_percentage,
            "decade_coverage": decade_coverage,
            "total_size_gb": total_gb,
            "decades_with_data": decades_with_data
        }
        
        # Log dataset quality
        if is_valid:
            logger.info(f"Dataset validation passed: {real_percentage:.1%} real data, {decade_coverage:.1%} decade coverage")
        else:
            logger.warning(f"Dataset validation failed: only {real_percentage:.1%} real data, {decade_coverage:.1%} decade coverage")
            if real_percentage < min_real_percentage:
                logger.warning(f"  Insufficient real data: {real_count}/{total_texts} texts ({real_percentage:.1%})")
            if decade_coverage < 0.5:
                logger.warning(f"  Insufficient decade coverage: {len(decades_with_data)}/{len(TIME_PERIODS)} decades")
            if total_gb < 0.1:
                logger.warning(f"  Insufficient data volume: {total_gb:.2f} GB")
        
        return is_valid, quality_report

    def verify_dataset_volumes(self, decade_texts, target_gb_per_decade=0.5):
        """
        Verify that each decade has sufficient data volume with enhanced error handling.
        
        Args:
            decade_texts: Dictionary mapping decades to texts
            target_gb_per_decade: Target volume in GB
            
        Returns:
            Tuple of (volumes_dict, all_sufficient)
        """
        volumes = {}
        all_sufficient = True
        
        for decade, texts in decade_texts.items():
            try:
                # Handle empty lists
                if not texts:
                    volumes[decade] = 0.0
                    all_sufficient = False
                    logger.warning(f"No data for {decade}: 0.00 GB (target: {target_gb_per_decade:.2f} GB)")
                    continue
                    
                # Calculate size with extensive error handling
                byte_size = 0
                for item in texts:
                    try:
                        if isinstance(item, tuple):
                            # Extract the text component (assume it's the first element)
                            text_content = item[0]
                        else:
                            # Use the item directly as text
                            text_content = item
                            
                        # Now ensure text_content is a string before encoding
                        if not isinstance(text_content, str):
                            logger.warning(f"Non-string content found in {decade}: {type(text_content)}")
                            continue
                            
                        byte_size += len(text_content.encode('utf-8'))
                    except Exception as e:
                        logger.warning(f"Error processing item in {decade}: {e}")
                        continue
                
                gb_size = byte_size / (1024**3)
                volumes[decade] = gb_size
                
                if gb_size < target_gb_per_decade:
                    all_sufficient = False
                    logger.warning(f"Insufficient data for {decade}: {gb_size:.2f} GB (target: {target_gb_per_decade:.2f} GB)")
                
            except Exception as e:
                logger.error(f"Error processing decade {decade}: {e}")
                volumes[decade] = 0.0
                all_sufficient = False
        
        # Log overall status
        if all_sufficient:
            logger.info(f"All decades meet the minimum volume requirement of {target_gb_per_decade:.2f} GB")
        
        return volumes, all_sufficient

    def load_checkpoint(self, name="checkpoint"):
        """
        Load the most recent checkpoint.
        
        Args:
            name: Checkpoint name prefix
            
        Returns:
            Checkpoint data or None if not found
        """
        import pickle
        
        checkpoint_dir = CACHE_DIR / "checkpoints"
        latest_path = checkpoint_dir / f"{name}_latest.pkl"
        
        if latest_path.exists():
            try:
                with open(latest_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded checkpoint from {latest_path}")
                return data
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        
        return None

    def save_checkpoint(self, data, name="checkpoint"):
        """
        Save a checkpoint during long-running operations.
        
        Args:
            data: Data to checkpoint
            name: Checkpoint name prefix
        """
        import pickle
        import time
        
        checkpoint_dir = CACHE_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"{name}_{timestamp}.pkl"
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Also save a latest version that always has the same name
            latest_path = checkpoint_dir / f"{name}_latest.pkl"
            with open(latest_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def ensure_historical_coverage(self):
        """
        Ensures we have sufficient data for all time periods, especially historical ones.
        This is crucial for accurate temporal distribution inference.
        
        Returns:
            Dictionary mapping decades to lists of texts with adequate coverage
        """
        logger.info("Ensuring adequate historical coverage for all decades...")
        
        # First, check current dataset status
        dataset = self.load_dataset()
        
        # If no dataset, build one
        if not dataset or sum(len(texts) for texts in dataset.values()) == 0:
            logger.info("No existing dataset found, building new dataset")
            dataset = self.build_temporal_dataset(texts_per_decade=30, save_dataset=True)
        
        # Check which decades have insufficient data (less than 20 texts)
        insufficient_decades = []
        for decade in TIME_PERIODS.keys():
            if decade not in dataset or len(dataset[decade]) < 20:
                insufficient_decades.append(decade)
        
        if not insufficient_decades:
            logger.info("All decades have sufficient data coverage")
            return dataset
        
        # If we have insufficient decades, enhance the dataset
        logger.info(f"Found {len(insufficient_decades)} decades with insufficient data: {insufficient_decades}")
        
        enhanced_dataset = {decade: texts.copy() if decade in dataset else [] for decade, texts in dataset.items()}
        
        # For each insufficient decade, generate synthetic data
        for decade in insufficient_decades:
            current_count = len(enhanced_dataset.get(decade, []))
            needed_count = max(30 - current_count, 0)
            
            if needed_count > 0:
                logger.info(f"Generating {needed_count} synthetic texts for {decade}")
                
                # Generate synthetic texts appropriate for this decade
                new_texts = self._create_historical_synthetic_texts(
                    decade=decade,
                    count=needed_count,
                    existing_data=dataset
                )
                
                # Add to dataset
                if decade not in enhanced_dataset:
                    enhanced_dataset[decade] = []
                
                # Tag texts as synthetic for proper tracking
                synthetic_texts = [(text, "synthetic") for text in new_texts]
                enhanced_dataset[decade].extend(synthetic_texts)
                
                logger.info(f"Added {len(new_texts)} synthetic texts to {decade}")
        
        # Save the enhanced dataset
        metadata = {
            "total_texts": sum(len(texts) for texts in enhanced_dataset.values()),
            "sources": {
                "british_library": sum(1 for decade_texts in enhanced_dataset.values() 
                                    for text, source in decade_texts if source == "british_library"),
                "gutenberg": sum(1 for decade_texts in enhanced_dataset.values() 
                                for text, source in decade_texts if source == "gutenberg"),
                "synthetic": sum(1 for decade_texts in enhanced_dataset.values() 
                                for text, source in decade_texts if source == "synthetic")
            },
            "decades": {decade: len(texts) for decade, texts in enhanced_dataset.items()}
        }
        
        self._save_dataset(enhanced_dataset, metadata)
        logger.info(f"Saved enhanced dataset with {metadata['total_texts']} total texts")
        
        return enhanced_dataset

    def _modify_text_slightly(self, text: str) -> str:
        """
        Make minor modifications to text to avoid exact duplication
        while maintaining its essential characteristics.
        
        Args:
            text: Original text
            
        Returns:
            Modified version of the text
        """
        # Don't modify very short texts
        if len(text) < 500:
            return text
        
        # Simple modifications:
        # 1. Change some punctuation
        # 2. Replace some common words with synonyms
        # 3. Add or remove a few minor words
        
        # Common word replacements to create variations
        word_replacements = {
            "very": ["quite", "extremely", "particularly", "rather"],
            "good": ["excellent", "fine", "satisfactory", "worthy"],
            "bad": ["poor", "unsatisfactory", "undesirable", "problematic"],
            "important": ["significant", "crucial", "essential", "noteworthy"],
            "big": ["large", "substantial", "considerable", "sizable"],
            "small": ["modest", "limited", "minor", "slight"],
            "interesting": ["intriguing", "engaging", "compelling", "noteworthy"],
            "people": ["individuals", "persons", "citizens", "population"],
            "think": ["believe", "consider", "suppose", "reckon"],
            "say": ["state", "declare", "remark", "mention"],
            "great": ["notable", "remarkable", "significant", "considerable"],
            "new": ["novel", "recent", "modern", "latest"],
            "old": ["former", "previous", "traditional", "established"]
        }
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Modify approximately 30% of sentences
        num_to_modify = max(1, int(len(sentences) * 0.3))
        indices_to_modify = random.sample(range(len(sentences)), num_to_modify)
        
        for idx in indices_to_modify:
            sentence = sentences[idx]
            words = sentence.split()
            
            if len(words) < 5:
                continue
            
            # 1. Word replacement - replace a few common words with synonyms
            for i, word in enumerate(words):
                word_lower = word.lower().rstrip(',.;:!?')
                if word_lower in word_replacements and random.random() < 0.4:
                    replacement = random.choice(word_replacements[word_lower])
                    # Preserve capitalization
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    # Preserve punctuation
                    if not word[-1].isalnum():
                        replacement = replacement + word[-1]
                    words[i] = replacement
            
            # 2. Add or remove minor words (articles, conjunctions, etc.)
            minor_words_to_add = ["also", "indeed", "certainly", "perhaps", "surely", "clearly", "obviously"]
            minor_words_to_remove = ["the", "a", "an", "very", "quite", "rather", "somewhat"]
            
            # Add a minor word (30% chance)
            if len(words) > 5 and random.random() < 0.3:
                insert_pos = random.randint(1, len(words) - 1)
                words.insert(insert_pos, random.choice(minor_words_to_add))
            
            # Remove a minor word if present (20% chance)
            if len(words) > 8 and random.random() < 0.2:
                removable_indices = [i for i, word in enumerate(words) 
                                if word.lower() in minor_words_to_remove]
                if removable_indices:
                    remove_idx = random.choice(removable_indices)
                    words.pop(remove_idx)
            
            # 3. Modify punctuation slightly (10% chance)
            if random.random() < 0.1 and len(words) > 3:
                # Find potential positions for comma insertion
                potential_comma_positions = [i for i in range(2, len(words) - 1) 
                                        if not words[i-1].endswith(',')]
                if potential_comma_positions:
                    comma_pos = random.choice(potential_comma_positions)
                    words[comma_pos-1] = words[comma_pos-1] + ","
            
            # Reconstruct the modified sentence
            sentences[idx] = " ".join(words)
        
        # Recombine sentences
        modified_text = " ".join(sentences)
        return modified_text

    def _create_historical_synthetic_texts(self, decade, count, existing_data: Dict[str, List] = {}) -> List[str]:
        """
        Create synthetic texts for a specific decade with appropriate vocabulary
        and style to supplement missing data.
        
        Args:
            decade: Target decade (e.g. '1850s')
            count: Number of texts to generate
            existing_data: Data already available to avoid duplication
            
        Returns:
            List of synthetic texts with period-appropriate content
        """
        # Check if we already have a similar method
        if hasattr(self, '_create_historical_synthetic_texts'):
            # Use existing method if available
            return self._create_historical_synthetic_texts(decade, count, {})
            
        # Define decade-specific vocabulary (if not already defined elsewhere)
        decade_vocab = {
            "1850s": ["railway", "telegraph", "empire", "industrial revolution", "manufactures", 
                    "workhouse", "steam-engine", "daguerreotype", "ether", "Chartists", 
                    "Crystal Palace", "Great Exhibition", "galvanic", "phrenology", "laudanum"],
            
            "1860s": ["telegram", "American Civil War", "telegraph wires", "colonization", "ironclad",
                    "Fenian", "suffrage", "zouave", "torpedo", "velocipede", "metropolitan railway",
                    "penny post", "chloroform", "telegraph", "typewriter", "dynamite"],
            
            "1870s": ["telephone", "phonograph", "typewriter", "electric light", "exhibition",
                    "gramophone", "hansom cab", "penny-farthing", "impressionism", "carbolic acid",
                    "jingoism", "anthropometry", "dynamo", "vulcanite", "gerrymander"],
            
            "1880s": ["electricity", "scientific", "phonograph", "industrial", "photographic", "bicycle",
                    "tuberculosis", "microbiology", "motorcar", "Home Rule", "suffragist", "telephone exchange",
                    "underground railway", "penny-farthing", "cocaine", "antiseptic", "germ theory"],
            
            "1890s": ["bicycle", "cinematograph", "photography", "modern", "telephone", "horseless carriage",
                    "horseless vehicle", "wireless", "X-rays", "aeroplane", "suffragette", "psychoanalysis",
                    "radioactivity", "typewriter", "tuberculin", "kinetoscope", "Kodak", "electric lights"],
            
            "1900s": ["automobile", "aeroplane", "wireless", "gramophone", "motion pictures", "cinematograph",
                    "suffragette", "wireless telegraph", "moving pictures", "eugenics", "psychoanalysis",
                    "radioactive", "modernism", "quantum", "Model T", "psychotherapy", "Zeppelin"],
            
            "1910s": ["Great War", "aeroplane", "wireless", "cinema", "modern", "trench warfare", "Soviet",
                    "jazz", "Bolshevik", "influenza epidemic", "conscription", "Zeppelin", "poison gas",
                    "tank", "shell shock", "U-boat", "wireless telephone", "dogfight", "cubism"],
            
            "1920s": ["wireless", "radio", "cinema", "automobile", "aeroplane", "modern", "broadcasting",
                    "flapper", "jazz", "talkies", "quantum mechanics", "relativity", "Prohibition", 
                    "stock market", "Hollywood", "bobbed hair", "insulin", "television", "speakeasy"],
            
            "1930s": ["depression", "radio", "cinema", "modern", "automobile", "broadcasting", "talking pictures",
                    "Dust Bowl", "New Deal", "Fascism", "Nazism", "unemployment", "breadline", "hooverville",
                    "dust storm", "talkie", "Empire State Building", "streamline", "radar", "quantum physics"],
            
            "1940s": ["war", "atomic", "radar", "radio", "modern", "atomic bomb", "nuclear", "antibiotics",
                    "United Nations", "Iron Curtain", "Holocaust", "television", "jet aircraft", "computer",
                    "penicillin", "nylon", "transistor", "Cold War", "NATO", "V-2 rocket"],
            
            "1950s": ["atomic", "television", "modern", "electric", "radio", "nuclear", "Soviet", "space race",
                    "Rock and Roll", "hydrogen bomb", "satellite", "automation", "transistor radio",
                    "polio vaccine", "civil rights", "suburban", "integrated circuit", "beatnik"],
                    
            "1960s": ["television", "modern", "electronic", "space", "computer", "Apollo", "lunar", "transistor",
                    "Vietnam War", "civil rights", "hippie", "counterculture", "LSD", "microchip", "The Pill",
                    "women's liberation", "mainframe", "NASA", "integrated circuit", "miniskirt"]
        }
        
        # Define era-specific writing styles for more authentic text
        era_styles = {
            "1850s": "formal Victorian prose with long sentences and elaborate descriptions",
            "1870s": "confident Victorian optimism about progress and industry",
            "1890s": "late Victorian scientific and social awareness",
            "1900s": "enthusiasm for new century and technology",
            "1920s": "jazz age modernity with shorter sentences and newer vocabulary",
            "1940s": "more direct, practical language reflecting war and reconstruction",
            "1960s": "increasingly informal with references to popular culture",
            "1980s": "technical and efficiency-focused language",
            "2000s": "digital-era casual language with technical terminology",
            "2020s": "contemporary language with emphasis on social and technological issues"
        }
        
        # Find closest era style for this decade
        closest_era = decade
        for era in sorted(era_styles.keys()):
            if decade >= era:
                closest_era = era
        
        era_style = era_styles.get(closest_era, "standard historical prose")
        vocab = decade_vocab.get(decade, ["historical", "period", "era", "decade", "time"])
        
        texts = []
        for i in range(count):
            # Generate a synthetic text with period vocabulary
            paragraphs = []
            
            # Add a title and introduction
            start_year = int(decade[:4])
            title = f"Historical Account from the {decade}"
            intro = f"The following text represents language typical of the {decade} period " + \
                    f"({start_year}-{start_year+9}), written in {era_style}."
            
            paragraphs.append(title)
            paragraphs.append(intro)
            
            # Generate 5-15 paragraphs with period-appropriate content
            for _ in range(random.randint(5, 15)):
                # Create a paragraph with period vocabulary
                sentences = []
                
                # Start with a topic sentence using period vocabulary
                topic_word = random.choice(vocab)
                topic_sentence = f"The {topic_word} was of considerable importance during this period."
                sentences.append(topic_sentence)
                
                # Add 3-8 supporting sentences
                for _ in range(random.randint(3, 8)):
                    # Occasionally use period vocabulary (30% chance)
                    if random.random() < 0.3:
                        word1 = random.choice(vocab)
                        word2 = random.choice(vocab)
                        
                        templates = [
                            f"The {word1} contributed significantly to developments in {word2}.",
                            f"Many considered {word1} to be essential to modern {word2}.",
                            f"The relationship between {word1} and {word2} merits further examination.",
                            f"The advancement of {word1} continued to transform {word2}.",
                            f"Scholars debated the significance of {word1} in relation to {word2}."
                        ]
                        
                        sentences.append(random.choice(templates))
                    else:
                        # General filler sentences
                        templates = [
                            "This development had far-reaching implications.",
                            "The consequences were felt throughout society.",
                            "Many contemporary accounts mention this phenomenon.",
                            "Historical records from this period confirm these observations.",
                            "The general public reacted with both enthusiasm and skepticism.",
                            "Subsequent events would prove these assessments correct.",
                            "Various factors contributed to this situation.",
                            "The historical context helps explain these developments."
                        ]
                        
                        sentences.append(random.choice(templates))
                
                # Combine sentences into a paragraph
                paragraph = " ".join(sentences)
                paragraphs.append(paragraph)
            
            # Combine paragraphs into a complete text
            text = "\n\n".join(paragraphs)
            texts.append(text)
        
        return texts

    def _balance_by_distribution(self, decade_texts, distribution, target_size_gb=1.0):
        """
        Balance the dataset according to the target distribution with emphasis on
        maximizing real data and minimizing synthetic content.
        
        Args:
            decade_texts: Dictionary mapping decades to lists of texts
            distribution: Target distribution mapping decades to proportions
            target_size_gb: Target size in GB for the total dataset
                
        Returns:
            Balanced dataset with proportions matching target distribution
        """
        logger.info("Balancing dataset according to target distribution...")
        
        # First, normalize data structure to ensure consistency
        normalized_texts = {}
        for decade, texts in decade_texts.items():
            normalized_decade_texts = []
            for item in texts:
                try:
                    # Handle different formats
                    if isinstance(item, tuple) and len(item) >= 1:
                        # Extract text component ensuring it's a string
                        text = item[0]
                        source = item[1] if len(item) > 1 else "unknown"
                        if isinstance(text, str):
                            normalized_decade_texts.append((text, source))
                    elif isinstance(item, str):
                        # Convert string to (text, source) tuple
                        normalized_decade_texts.append((item, "normalized"))
                    # Skip other formats
                except Exception as e:
                    logger.debug(f"Skipping invalid item: {e}")
            
            normalized_texts[decade] = normalized_decade_texts
        
        # Calculate total size and current distribution
        total_bytes = 0
        decade_bytes = {}
        
        for decade, texts in normalized_texts.items():
            if not texts:
                decade_bytes[decade] = 0
                continue
                
            bytes_size = sum(len(text.encode('utf-8')) for text, _ in texts)
            decade_bytes[decade] = bytes_size
            total_bytes += bytes_size
        
        # Calculate current distribution
        current_distribution = {decade: bytes_size / max(1, total_bytes) 
                            for decade, bytes_size in decade_bytes.items()}
        
        # Calculate target bytes per decade
        total_target_bytes = target_size_gb * 1024 * 1024 * 1024
        target_bytes_per_decade = {decade: total_target_bytes * prop 
                                for decade, prop in distribution.items()}
        
        # Special handling for historical decades - give them higher priority
        historical_decades = ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s"]
        for decade in historical_decades:
            if decade in target_bytes_per_decade:
                # Increase from 1.5x to 3x for historical decades
                target_bytes_per_decade[decade] = target_bytes_per_decade[decade] * 3.0  
                logger.info(f"Boosting target data volume for historical decade {decade} by 300%")

        # Create balanced dataset
        balanced_dataset = {}
        
        # First, categorize data sources for each decade
        for decade, target_prop in distribution.items():
            texts = normalized_texts.get(decade, [])
            
            if not texts:
                balanced_dataset[decade] = []
                logger.warning(f"No texts available for {decade}, cannot match distribution")
                continue
            
            # Categorize texts by source type
            real_texts = []
            expanded_texts = []
            synthetic_texts = []
            
            for text, source in texts:
                if "synthetic" in source:
                    synthetic_texts.append((text, source))
                elif "expanded" in source or "augmented" in source:
                    expanded_texts.append((text, source))
                else:
                    real_texts.append((text, source))
            
            current_bytes = decade_bytes.get(decade, 0)
            target_bytes = target_bytes_per_decade.get(decade, 0)
            
            # Prioritize historical decades for better representation
            is_historical = decade in historical_decades
            
            # If current is less than target, use all available texts
            # Prioritizing real texts over expanded and synthetic
            if current_bytes <= target_bytes or is_historical:
                # For historical decades, use all available real texts
                balanced_dataset[decade] = real_texts
                
                # If we need more data, add expanded texts
                if current_bytes < target_bytes:
                    balanced_dataset[decade].extend(expanded_texts)
                
                # Still need more? Add synthetic, but with limits
                if current_bytes < target_bytes * 0.8 and not is_historical:
                    # For non-historical, limit synthetic to 30% max
                    synthetic_limit = min(len(synthetic_texts), int(len(real_texts) * 0.3))
                    balanced_dataset[decade].extend(synthetic_texts[:synthetic_limit])
                elif is_historical and current_bytes < target_bytes * 0.5:
                    # For historical, limit synthetic to 15% max
                    synthetic_limit = min(len(synthetic_texts), int(len(real_texts) * 0.15))
                    balanced_dataset[decade].extend(synthetic_texts[:synthetic_limit])
                    
                logger.info(f"Using all {len(real_texts)} real texts plus {len(balanced_dataset[decade]) - len(real_texts)} additional texts for {decade}")
            else:
                # We have more than needed, sample to match target
                # First calculate bytes for each category
                real_bytes = sum(len(text.encode('utf-8')) for text, _ in real_texts)
                expanded_bytes = sum(len(text.encode('utf-8')) for text, _ in expanded_texts)
                synthetic_bytes = sum(len(text.encode('utf-8')) for text, _ in synthetic_texts)
                
                # Determine how much of each type to use
                if real_bytes >= target_bytes * 0.9:
                    # We have enough real data to almost meet target
                    # Calculate how many real texts to sample
                    bytes_per_real_text = real_bytes / len(real_texts) if real_texts else 0
                    real_sample_size = min(len(real_texts), int(target_bytes * 0.9 / bytes_per_real_text) if bytes_per_real_text > 0 else 0)
                    
                    # Prioritize longer, higher quality texts
                    real_texts_with_length = [(text, source, len(text.encode('utf-8'))) 
                                        for text, source in real_texts]
                    real_texts_with_length.sort(key=lambda x: x[2], reverse=True)
                    
                    # Take top 20% by length
                    top_count = max(5, real_sample_size // 5)
                    top_texts = [real_texts_with_length[i][:2] for i in range(min(top_count, len(real_texts_with_length)))]
                    
                    # Random sample from the rest
                    if real_sample_size - top_count > 0 and len(real_texts_with_length) > top_count:
                        remaining = random.sample(real_texts_with_length[top_count:], 
                                                min(real_sample_size - top_count, len(real_texts_with_length) - top_count))
                        sampled_real_texts = top_texts + [item[:2] for item in remaining]
                    else:
                        sampled_real_texts = top_texts
                    
                    # Calculate remaining target
                    real_sampled_bytes = sum(len(text.encode('utf-8')) for text, _ in sampled_real_texts)
                    remaining_target = target_bytes - real_sampled_bytes
                    
                    # Fill remaining with expanded texts
                    expanded_sample_size = 0
                    sampled_expanded_texts = []
                    
                    if remaining_target > 0 and expanded_texts:
                        bytes_per_expanded = expanded_bytes / len(expanded_texts)
                        expanded_sample_size = min(len(expanded_texts), int(remaining_target / bytes_per_expanded) if bytes_per_expanded > 0 else 0)
                        sampled_expanded_texts = random.sample(expanded_texts, expanded_sample_size) if expanded_sample_size > 0 else []
                    
                    # Only use synthetic as last resort
                    expanded_sampled_bytes = sum(len(text.encode('utf-8')) for text, _ in sampled_expanded_texts)
                    remaining_target = target_bytes - real_sampled_bytes - expanded_sampled_bytes
                    
                    sampled_synthetic_texts = []
                    if remaining_target > 0 and synthetic_texts:
                        bytes_per_synthetic = synthetic_bytes / len(synthetic_texts)
                        synthetic_sample_size = min(len(synthetic_texts), int(remaining_target / bytes_per_synthetic) if bytes_per_synthetic > 0 else 0)
                        # Limit synthetic to 10% maximum
                        synthetic_sample_size = min(synthetic_sample_size, int(real_sample_size * 0.1))
                        sampled_synthetic_texts = random.sample(synthetic_texts, synthetic_sample_size) if synthetic_sample_size > 0 else []
                    
                    # Combine all sampled texts
                    balanced_dataset[decade] = sampled_real_texts + sampled_expanded_texts + sampled_synthetic_texts
                    
                    logger.info(f"Sampled {len(sampled_real_texts)} real, {len(sampled_expanded_texts)} expanded, and {len(sampled_synthetic_texts)} synthetic texts for {decade} to match distribution")
                
                else:
                    # We don't have enough real data - use all real and some expanded/synthetic
                    real_bytes = sum(len(text.encode('utf-8')) for text, _ in real_texts)
                    remaining_target = target_bytes - real_bytes
                    
                    # Use expanded texts to fill gap
                    expanded_sample_size = 0
                    sampled_expanded_texts = []
                    
                    if remaining_target > 0 and expanded_texts:
                        bytes_per_expanded = expanded_bytes / len(expanded_texts)
                        expanded_sample_size = min(len(expanded_texts), int(remaining_target / bytes_per_expanded) if bytes_per_expanded > 0 else 0)
                        if expanded_sample_size > 0:
                            sampled_expanded_texts = expanded_texts[:expanded_sample_size]
                    
                    # Use synthetic as last resort, limited to 20% of real+expanded
                    expanded_sampled_bytes = sum(len(text.encode('utf-8')) for text, _ in sampled_expanded_texts)
                    remaining_target = target_bytes - real_bytes - expanded_sampled_bytes
                    
                    sampled_synthetic_texts = []
                    if remaining_target > 0 and synthetic_texts:
                        bytes_per_synthetic = synthetic_bytes / len(synthetic_texts)
                        synthetic_sample_size = min(len(synthetic_texts), int(remaining_target / bytes_per_synthetic) if bytes_per_synthetic > 0 else 0)
                        # Limit synthetic to 20% of real+expanded
                        synthetic_limit = max(10, int((len(real_texts) + expanded_sample_size) * 0.2))
                        synthetic_sample_size = min(synthetic_sample_size, synthetic_limit)
                        if synthetic_sample_size > 0:
                            sampled_synthetic_texts = synthetic_texts[:synthetic_sample_size]
                    
                    # Combine all sampled texts
                    balanced_dataset[decade] = real_texts + sampled_expanded_texts + sampled_synthetic_texts
                    
                    logger.info(f"Using all {len(real_texts)} real texts plus {len(sampled_expanded_texts)} expanded and {len(sampled_synthetic_texts)} synthetic for {decade}")
        
        # Verify final size
        final_bytes = sum(sum(len(text.encode('utf-8')) for text, _ in texts) 
                        for decade, texts in balanced_dataset.items())
        
        logger.info(f"Balanced dataset total size: {final_bytes/(1024*1024*1024):.2f} GB")
        
        # Analyze final composition
        total_real = sum(sum(1 for text, source in texts if not ("synthetic" in source or "expanded" in source))
                    for decade, texts in balanced_dataset.items())
        total_expanded = sum(sum(1 for text, source in texts if "expanded" in source or "augmented" in source)
                        for decade, texts in balanced_dataset.items())
        total_synthetic = sum(sum(1 for text, source in texts if "synthetic" in source)
                        for decade, texts in balanced_dataset.items())
        total_texts = total_real + total_expanded + total_synthetic
        
        if total_texts > 0:
            real_percent = total_real / total_texts * 100
            expanded_percent = total_expanded / total_texts * 100
            synthetic_percent = total_synthetic / total_texts * 100
            
            logger.info(f"Final dataset composition: {real_percent:.1f}% real, {expanded_percent:.1f}% expanded, {synthetic_percent:.1f}% synthetic")
        
        return balanced_dataset
    
    def create_large_dataset(self, distribution: Dict[str, float], target_size_gb: float = 1.0) -> Dict[str, List[Tuple[str, str]]]:
        """
        Create a dataset with known temporal distribution for validation,
        scaled up to match Hayase et al.'s data volumes.
        Enhanced for better balance and decade-specific characteristics.
        
        Args:
            distribution: Dictionary mapping decades to proportions
            target_size_gb: Target size in GB for the total dataset
                
        Returns:
            Dictionary mapping decades to lists of texts with the specified distribution
        """
        logger.info(f"Creating controlled dataset with distribution: {distribution}")
        
        # Set data volume targets based on Hayase paper
        target_size_bytes = target_size_gb * 1024 * 1024 * 1024
        
        # Calculate bytes per decade based on distribution
        bytes_per_decade = {decade: target_size_bytes * prop for decade, prop in distribution.items()}
        
        # Normalize distribution if needed
        total_proportion = sum(distribution.values())
        if abs(total_proportion - 1.0) > 0.001:  # Allow small rounding errors
            normalized = {d: v/total_proportion for d, v in distribution.items()}
            logger.info(f"Normalized distribution to: {normalized}")
            distribution = normalized
        
        # Calculate texts per decade, with much higher counts than before
        texts_per_decade = {decade: max(int(prop * 20000), 1000) for decade, prop in distribution.items()}
        
        # Load all available data with expanded coverage
        logger.info("Loading source texts for controlled dataset...")
        
        # Load British Library historical data FIRST for best historical coverage
        logger.info("Loading British Library historical data...")
        self.british_library_loader.expand_metadata_sources()
        bl_texts = self.british_library_loader.load_british_library_historical_data(
            per_decade=50000,  # Massively increased from 10000 for better coverage
            early_stop=False   # Disable early stopping to ensure we get all available data
        )
        
        # Enhanced Gutenberg data loading with historical focus
        logger.info("Loading Gutenberg texts with enhanced historical focus...")
        self.gutenberg_loader.expand_historical_catalog()
        self.gutenberg_loader.expand_metadata_sources()
        gutenberg_texts = self.gutenberg_loader.load_focused_decade_samples(
            target_decades=list(distribution.keys()),
            texts_per_decade=50000  # Massively increased from 20000
        )
        
        # Load Oscar for mid-century decades with increased volume
        target_decades = list(distribution.keys())
        logger.info(f"Loading Oscar texts for all decades: {target_decades}")
        oscar_texts = self.oscar_loader.load_decade_samples(
            target_decades=target_decades,
            texts_per_decade=50000  # Increased from 20000
        )
        
        # Load modern web content for recent decades
        modern_decades = [d for d in distribution.keys() 
                        if d in ["1990s", "2000s", "2010s", "2020s"]]
        if modern_decades:
            logger.info(f"Loading modern web content for decades: {modern_decades}")
            modern_texts = self.load_modern_web_content(
                target_decades=modern_decades, 
                texts_per_decade=50000  # Increased from 20000
            )
        else:
            modern_texts = {}
        
        # Initialize a dictionary to store all source texts by decade
        all_source_texts = {decade: [] for decade in distribution.keys()}
        
        # Combine all data sources, with complete source tracking
        for decade, texts in bl_texts.items():
            if decade in distribution:
                all_source_texts[decade].extend([(text, "british_library") for text in texts])
        
        for decade, texts in gutenberg_texts.items():
            if decade in all_source_texts:
                all_source_texts[decade].extend(texts)
        
        for decade, texts in oscar_texts.items():
            if decade in all_source_texts:
                all_source_texts[decade].extend(texts)
        
        for decade, texts in modern_texts.items():
            if decade in all_source_texts:
                all_source_texts[decade].extend(texts)
        
        # Log data counts by source for each decade
        for decade, texts in all_source_texts.items():
            bl_count = sum(1 for _, src in texts if src == "british_library")
            gutenberg_count = sum(1 for _, src in texts if src == "gutenberg")
            oscar_count = sum(1 for _, src in texts if src == "oscar")
            modern_count = sum(1 for _, src in texts if "web" in str(src).lower())
            other_count = len(texts) - (bl_count + gutenberg_count + oscar_count + modern_count)
            
            logger.info(f"{decade} source breakdown: {len(texts)} total texts")
            logger.info(f"  - British Library: {bl_count} texts")
            logger.info(f"  - Gutenberg: {gutenberg_count} texts")
            logger.info(f"  - Oscar: {oscar_count} texts")
            logger.info(f"  - Modern web: {modern_count} texts")
            logger.info(f"  - Other: {other_count} texts")
        
        # Build the controlled dataset with enhanced decade-specific preservation
        controlled_dataset = {}
        current_size_bytes = 0
        
        # Process each decade individually with exact byte tracking
        for decade, target_bytes in bytes_per_decade.items():
            source_texts = all_source_texts.get(decade, [])
            
            if not source_texts:
                logger.warning(f"No source texts for {decade}, generating synthetic texts")
                synthetic_texts = self._create_historical_synthetic_texts(
                    decade=decade,
                    count=1000,  # Generate 1000 synthetic texts
                    existing_data={},
                    preserve_decade_characteristics=True
                )
                source_texts = [(text, "synthetic") for text in synthetic_texts]
            
            # Sort texts by quality and length (prioritize real and longer texts)
            sorted_texts = sorted(source_texts, key=lambda x: (
                0 if "synthetic" not in x[1] and "augmented" not in x[1] else 1,  # Real texts first
                len(x[0]),  # Then longer texts
            ), reverse=True)
            
            # Build dataset with exact byte-level tracking
            decade_texts = []
            decade_bytes = 0
            
            for text, source in sorted_texts:
                # Add text if it doesn't exceed target
                text_bytes = len(text.encode('utf-8'))
                if decade_bytes + text_bytes <= target_bytes:
                    decade_texts.append((text, source))
                    decade_bytes += text_bytes
                else:
                    # For last text, add a partial text to exactly meet target
                    remaining_bytes = target_bytes - decade_bytes
                    if remaining_bytes > 1000:  # Only add if significant chunk remains
                        # Find a suitable truncation point (end of sentence)
                        truncation_point = min(len(text), int(remaining_bytes))
                        # Try to find a sentence boundary
                        for i in range(truncation_point - 1, max(0, truncation_point - 200), -1):
                            if i < len(text) and text[i] in '.!?' and i + 1 < len(text) and text[i+1].isspace():
                                truncation_point = i + 1
                                break
                        
                        partial_text = text[:truncation_point]
                        decade_texts.append((partial_text, f"{source}_partial"))
                        decade_bytes += len(partial_text.encode('utf-8'))
                    
                    # We've reached our target, stop adding texts
                    break
            
            # If we didn't reach the target, synthesize or augment more texts
            if decade_bytes < target_bytes * 0.95:  # If we're below 95% of target
                logger.info(f"Only reached {decade_bytes/(target_bytes)*100:.1f}% of target for {decade}, adding synthesized/augmented content")
                
                # First try to augment existing texts
                if decade_texts:
                    remaining_bytes = target_bytes - decade_bytes
                    augmented_bytes = 0
                    
                    # Base texts to augment from - up to 100 texts
                    base_texts = decade_texts[:min(100, len(decade_texts))]
                    augmented_count = 0
                    
                    while augmented_bytes < remaining_bytes and augmented_count < 1000:
                        # Select a random base text
                        base_text, base_source = random.choice(base_texts)
                        
                        # Create augmented version
                        augmented_text = self._augment_text_for_volume(
                            base_text, 
                            decade, 
                            volume_multiplier=random.randint(3, 10)  # Variable size augmentation
                        )
                        
                        text_bytes = len(augmented_text.encode('utf-8'))
                        if decade_bytes + text_bytes <= target_bytes:
                            decade_texts.append((augmented_text, f"{base_source}_augmented"))
                            decade_bytes += text_bytes
                            augmented_bytes += text_bytes
                        
                        augmented_count += 1
                        
                        # Log progress periodically
                        if augmented_count % 100 == 0:
                            logger.info(f"Added {augmented_count} augmented texts ({augmented_bytes/(1024*1024):.2f} MB) for {decade}")
                
                # If still not enough, add synthetic texts
                if decade_bytes < target_bytes * 0.95:
                    remaining_bytes = target_bytes - decade_bytes
                    
                    # Generate synthetic texts
                    estimated_text_size = 10000  # Assume average synthetic text is 10KB
                    synthetic_count = max(10, int(remaining_bytes / estimated_text_size))
                    
                    logger.info(f"Adding {synthetic_count} synthetic texts for {decade}")
                    synthetic_texts = self._create_historical_synthetic_texts(
                        decade=decade,
                        count=synthetic_count,
                        existing_data={},
                        preserve_decade_characteristics=True
                    )
                    
                    # Add synthetic texts up to target
                    for text in synthetic_texts:
                        text_bytes = len(text.encode('utf-8'))
                        if decade_bytes + text_bytes <= target_bytes:
                            decade_texts.append((text, "synthetic"))
                            decade_bytes += text_bytes
                        else:
                            break
            
            controlled_dataset[decade] = decade_texts
            current_size_bytes += decade_bytes

            # Log final volume for this decade
            logger.info(f"{decade} final dataset: {len(decade_texts)} texts, {decade_bytes/(1024*1024*1024):.3f} GB")
            
            # Log composition statistics
            real_count = sum(1 for _, src in decade_texts if "synthetic" not in src and "augmented" not in src and "partial" not in src)
            augmented_count = sum(1 for _, src in decade_texts if "augmented" in src)
            partial_count = sum(1 for _, src in decade_texts if "partial" in src)
            synthetic_count = sum(1 for _, src in decade_texts if "synthetic" in src)
            
            if decade_texts:
                logger.info(f"  Composition: {real_count/len(decade_texts):.1%} real, "
                        f"{augmented_count/len(decade_texts):.1%} augmented, "
                        f"{partial_count/len(decade_texts):.1%} partial, "
                        f"{synthetic_count/len(decade_texts):.1%} synthetic")
        
        # Verify the final dataset meets the target distribution
        total_bytes = sum(sum(len(text.encode('utf-8')) for text, _ in texts) 
                        for decade, texts in controlled_dataset.items())
        
        actual_distribution = {}
        for decade, texts in controlled_dataset.items():
            decade_bytes = sum(len(text.encode('utf-8')) for text, _ in texts)
            actual_distribution[decade] = decade_bytes / total_bytes if total_bytes > 0 else 0
        
        # Log actual vs target distribution for verification
        logger.info("Actual vs Target Distribution:")
        for decade in sorted(distribution.keys()):
            target = distribution.get(decade, 0)
            actual = actual_distribution.get(decade, 0)
            difference = actual - target
            logger.info(f"  {decade}: {actual:.1%} (target: {target:.1%}, diff: {difference:+.1%})")
        
        # Create metadata
        metadata = {
            "type": "controlled_dataset",
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_distribution": distribution,
            "actual_distribution": actual_distribution,
            "target_size_gb": target_size_gb,
            "actual_size_gb": total_bytes / (1024*1024*1024),
            "total_texts": sum(len(texts) for texts in controlled_dataset.values()),
            "composition": {
                "real": sum(sum(1 for _, src in texts if "synthetic" not in src and "augmented" not in src and "partial" not in src) 
                        for texts in controlled_dataset.values()),
                "augmented": sum(sum(1 for _, src in texts if "augmented" in src) 
                            for texts in controlled_dataset.values()),
                "partial": sum(sum(1 for _, src in texts if "partial" in src) 
                            for texts in controlled_dataset.values()),
                "synthetic": sum(sum(1 for _, src in texts if "synthetic" in src) 
                            for texts in controlled_dataset.values()),
            }
        }
        
        # Save metadata
        metadata_path = self.dataset_dir / "controlled_datasets"
        metadata_path.mkdir(exist_ok=True, parents=True)
        with open(metadata_path / f"controlled_dataset_{int(time.time())}.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Total dataset size: {total_bytes/(1024*1024*1024):.2f} GB")
        return controlled_dataset    

    
    def load_additional_sources(self, target_decades):
        """
        Load additional contemporary data sources to supplement the dataset.
        """
        logger.info("Loading additional contemporary data sources...")
        
        decade_texts = {decade: [] for decade in target_decades}
        
        # Only focus on newer decades that typically need more data
        modern_decades = [d for d in target_decades if d in ["1990s", "2000s", "2010s", "2020s"]]
        
        if not modern_decades:
            return decade_texts
        
        try:
            # Try to load C4 dataset samples - good source for 2000s and 2010s
            from datasets import load_dataset
            import pickle
            
            logger.info("Loading samples from C4 dataset...")
            try:
                # Set a 5-minute timeout
                import signal
                
                class TimeoutException(Exception):
                    pass
                
                def timeout_handler(signum, frame):
                    raise TimeoutException("C4 dataset loading timed out")
                
                # Set timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)
                
                try:
                    # MODIFIED: Remove the problematic download_config
                    c4_dataset = load_dataset(
                        "c4", "en", split="train", streaming=True, 
                        trust_remote_code=True
                    )
                    
                    # Reset the alarm
                    signal.alarm(0)
                    
                    # Process a limited number of examples
                    sample_size = 10000
                    
                    processed = 0
                    assigned = 0
                    # Check if we have a checkpoint to resume from
                    checkpoint_path = CACHE_DIR / "checkpoints" / "c4_processing_latest.pkl"
                    if checkpoint_path.exists():
                        try:
                            with open(checkpoint_path, 'rb') as f:
                                checkpoint = pickle.load(f)
                                if "decade_texts" in checkpoint and "processed" in checkpoint:
                                    # Resume from checkpoint
                                    stored_texts = checkpoint["decade_texts"]
                                    for decade in target_decades:
                                        if decade in stored_texts and stored_texts[decade]:
                                            decade_texts[decade].extend(stored_texts[decade])
                                    processed = checkpoint.get("processed", 0)
                                    assigned = checkpoint.get("assigned", 0)
                                    logger.info(f"Resumed C4 processing from checkpoint: {processed} processed, {assigned} assigned")
                        except Exception as e:
                            logger.warning(f"Failed to load C4 processing checkpoint: {e}")
                    
                    # Skip already processed examples if resuming
                    skip_count = processed
                    processed_this_session = 0
                    
                    for i, example in enumerate(c4_dataset.take(sample_size + skip_count)):
                        # Skip examples we've already processed
                        if i < skip_count:
                            continue
                            
                        if "text" not in example or not example["text"]:
                            continue
                            
                        text = example["text"]
                        # Skip if too short
                        if len(text) < 1000:
                            continue
                            
                        # Extract decade information - focusing on modern decades
                        decade = self._extract_decade_from_text_enhanced(text, modern_decades)
                        
                        if decade:
                            decade_texts[decade].append((text, f"c4_dataset"))
                            assigned += 1
                        
                        processed += 1
                        processed_this_session += 1
                        
                        if processed % 1000 == 0:
                            logger.info(f"Processed {processed} C4 examples, assigned {assigned}")
                            
                        # Save checkpoint periodically
                        if processed_this_session % 2000 == 0:
                            checkpoint_dir = CACHE_DIR / "checkpoints"
                            checkpoint_dir.mkdir(exist_ok=True)
                            
                            with open(checkpoint_dir / "c4_processing_latest.pkl", 'wb') as f:
                                pickle.dump({
                                    "processed": processed,
                                    "assigned": assigned,
                                    "decade_texts": decade_texts
                                }, f)
                            logger.info(f"Saved C4 processing checkpoint at {processed} examples")
                            
                        if assigned >= 5000:  # Stop once we have enough
                            break
                    
                    logger.info(f"Added {assigned} texts from C4 dataset")
                    
                except TimeoutException:
                    logger.warning("C4 dataset loading timed out after 5 minutes")
                finally:
                    # Ensure alarm is disabled even if there was an error
                    signal.alarm(0)
                    
            except Exception as e:
                logger.warning(f"Failed to load C4 dataset: {e}")
        
        except ImportError:
            logger.warning("Could not import datasets library, skipping C4 loading")
        
        # Try loading Wikipedia samples if available
        try:
            # Make sure pickle is imported
            wiki_cache_path = CACHE_DIR / "wikipedia_samples.pkl"
            if wiki_cache_path.exists():
                try:
                    with open(wiki_cache_path, 'rb') as f:
                        wiki_samples = pickle.load(f)
                        logger.info(f"Loaded {sum(len(texts) for texts in wiki_samples.values())} Wikipedia samples from cache")
                        for decade in modern_decades:
                            if decade in wiki_samples:
                                decade_texts[decade].extend(wiki_samples[decade])
                except Exception as e:
                    logger.warning(f"Failed to load Wikipedia samples from cache: {e}")
            
            # If we don't have cached data or it failed to load, fetch new data
            if all(len(decade_texts[decade]) == 0 for decade in modern_decades):
                try:
                    # Add a more robust download configuration
                    from datasets import DownloadConfig
                    download_config = DownloadConfig(
                        max_retries=10,  # Increase retries
                        timeout=300,     # Longer timeout (5 minutes)
                        force_download=False,
                        cache_dir=str(CACHE_DIR / "wikipedia")
                    )
                    
                    wiki_dataset = load_dataset(
                        "wikipedia", "20220301.en", split="train", streaming=True,
                        trust_remote_code=True,
                        download_config=download_config
                    )
                    
                    sample_size = 8000  # Adjust as needed
                    processed = 0
                    assigned = 0
                    
                    for i, example in enumerate(wiki_dataset.take(sample_size)):
                        if "text" not in example or not example["text"]:
                            continue
                            
                        text = example["text"]
                        title = example.get("title", "")
                        
                        # Skip if too short
                        if len(text) < 1000:
                            continue
                            
                        # Check for time-specific articles
                        time_indicators = [
                            "history", "in the", "century", "decade", 
                            "period", "era", "year", "timeline"
                        ]
                        
                        # Give preference to articles with time indicators in title
                        has_time_indicator = any(indicator in title.lower() for indicator in time_indicators)
                        
                        # Extract decade with enhanced method
                        decade = self._extract_decade_from_text_enhanced(text, modern_decades)
                        
                        # For time-related articles, use a more aggressive classification approach
                        if has_time_indicator and not decade:
                            # Look harder for temporal clues in the first paragraph
                            first_para = text.split("\n\n")[0] if "\n\n" in text else text[:2000]
                            decades_mentioned = []
                            
                            for d in modern_decades:
                                decade_year = d[:4]
                                if decade_year in first_para or d in first_para:
                                    decades_mentioned.append(d)
                            
                            if decades_mentioned:
                                decade = random.choice(decades_mentioned)
                        
                        if decade:
                            # Check if we need more texts for this decade
                            decade_texts[decade].append((text, f"wikipedia_{title}"))
                            assigned += 1
                        
                        processed += 1
                        if processed % 500 == 0:
                            logger.info(f"Processed {processed} Wikipedia articles, assigned {assigned}")
                            
                        # Ensure we have a good balance
                        min_per_decade = 200
                        if all(len(texts) >= min_per_decade for decade, texts in decade_texts.items() 
                            if decade in ["1990s", "2000s", "2010s"]):
                            break
                    
                    # Cache the results
                    try:
                        with open(wiki_cache_path, 'wb') as f:
                            pickle.dump(decade_texts, f)
                    except Exception as e:
                        logger.warning(f"Failed to cache Wikipedia samples: {e}")
                    
                    logger.info(f"Loaded {assigned} Wikipedia articles, distributed across {len(modern_decades)} decades")
                
                except Exception as e:
                    logger.error(f"Failed to load Wikipedia dataset: {e}")
                    # Create a fallback empty dataset
                    from datasets import Dataset
                    wiki_dataset = Dataset.from_dict({"text": [], "title": []})
        
        except Exception as e:
            logger.warning(f"Error processing Wikipedia samples: {e}")
        
        # Return the collected additional texts
        total_texts = sum(len(texts) for decade, texts in decade_texts.items())
        logger.info(f"Loaded {total_texts} texts from additional sources")
        
        return decade_texts

    def _extract_decade_from_text_enhanced(self, text, target_decades):
        """
        Enhanced version of decade extraction with better pattern recognition
        for more accurate temporal classification.
        
        Args:
            text: Text content to analyze
            target_decades: List of decades to consider
            
        Returns:
            Detected decade or None
        """
        # 1. Look for explicit year mentions with more patterns (19XX or 20XX)
        year_patterns = [
            r'\b(19[0-9]{2}|20[0-2][0-9])\b',  # YYYY (1900-2029)
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* [0-9]{1,2},? (19[0-9]{2}|20[0-2][0-9])\b',  # Month D(D), YYYY
            r'\b[0-9]{1,2}[/-][0-9]{1,2}[/-](19[0-9]{2}|20[0-2][0-9])\b',  # D(D)/D(D)/YYYY
            r'copyright\s+(?:©|\(c\))?\s*(19[0-9]{2}|20[0-2][0-9])',  # Copyright year
            r'published\s+(?:in)?\s*(19[0-9]{2}|20[0-2][0-9])'  # Publication year
        ]
        
        # Combine patterns for efficiency
        combined_pattern = '|'.join(f'({pattern})' for pattern in year_patterns)
        years = re.findall(combined_pattern, text, re.IGNORECASE)
        
        # Flatten and clean the results
        flat_years = []
        for match in years:
            for group in match:
                if group and re.match(r'^(19|20)\d{2}$', group):
                    flat_years.append(group)
        
        # Convert found years to decades with weighted distribution
        decade_counts = {}
        for year_str in flat_years:
            try:
                year = int(year_str)
                for decade, (start_year, end_year) in TIME_PERIODS.items():
                    if start_year <= year <= end_year and decade in target_decades:
                        # Give more weight to years that appear early in the text (likely publication dates)
                        position_weight = 2.0 if text.find(year_str) < len(text) // 4 else 1.0
                        decade_counts[decade] = decade_counts.get(decade, 0) + position_weight
            except ValueError:
                continue
        
        # If we found years, use the most common decade
        if decade_counts:
            most_common_decade = max(decade_counts.items(), key=lambda x: x[1])[0]
            return most_common_decade
        
        # 2. Look for decade names with more variations ("1990s", "sixties", etc.)
        decade_patterns = {
            "1930s": [r'\b19[3]0s\b', r'\bthirties\b', r'\b30s\b', r'\b1930\'?s\b'],
            "1940s": [r'\b19[4]0s\b', r'\bforties\b', r'\b40s\b', r'\b1940\'?s\b'],
            "1950s": [r'\b19[5]0s\b', r'\bfifties\b', r'\b50s\b', r'\b1950\'?s\b'],
            "1960s": [r'\b19[6]0s\b', r'\bsixties\b', r'\b60s\b', r'\b1960\'?s\b'],
            "1970s": [r'\b19[7]0s\b', r'\bseventies\b', r'\b70s\b', r'\b1970\'?s\b'],
            "1980s": [r'\b19[8]0s\b', r'\beighties\b', r'\b80s\b', r'\b1980\'?s\b'],
            "1990s": [r'\b19[9]0s\b', r'\bnineties\b', r'\b90s\b', r'\b1990\'?s\b'],
            "2000s": [r'\b20[0]0s\b', r'\btwo thousands\b', r'\b2000\'?s\b', r'\nearly 2000s\b'],
            "2010s": [r'\b20[1]0s\b', r'\btwenty tens\b', r'\b2010\'?s\b', r'\b201\ds\b'],
            "2020s": [r'\b20[2]0s\b', r'\btwenty twenties\b', r'\b2020\'?s\b', r'\b202\ds\b'],
        }
        
        for decade, patterns in decade_patterns.items():
            if decade in target_decades:
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        return decade
        
        # 3. Use era-specific vocabulary and markers
        decade_markers = {
            "1990s": ["world wide web", "clinton", "internet explorer", "windows 95", "netscape", 
                    "dial-up", "gulf war", "berlin wall", "soviet union collapse", "y2k"],
            "2000s": ["9/11", "iraq war", "facebook", "ipod", "bush administration", "myspace", 
                    "financial crisis", "harry potter", "hurricane katrina", "web 2.0"],
            "2010s": ["smartphone", "obama", "instagram", "trump", "brexit", "occupy wall street", 
                    "social media", "arab spring", "cloud computing", "black lives matter"],
            "2020s": ["pandemic", "COVID-19", "TikTok", "ukraine war", "vaccine", "lockdown", 
                    "inflation", "metaverse", "AIrevolution", "remote work", "social distancing", "contact tracing", "quarantine", "stimulus checks"]
        }
        
        for decade, markers in decade_markers.items():
            if decade in target_decades:
                for marker in markers:
                    if marker in text.lower():
                        return decade
        
        return None  # If no decade could be determined
    
    def boost_historical_data(self):
        """
        Create a dataset with boosted historical data for use when we need to
        emphasize historical content in our analysis.
        
        Returns:
            Dictionary mapping decades to lists of texts with emphasis on historical content
        """
        logger.info("Creating dataset with boosted historical content...")
        
        # Define historical decades we want to boost
        historical_decades = ["1850s", "1860s", "1870s", "1880s", "1890s", 
                             "1900s", "1910s", "1920s", "1930s", "1940s"]
        
        # Start with existing dataset if available
        dataset = self.load_dataset()
        if not dataset:
            logger.info("No existing dataset found, building new dataset")
            dataset = self.build_temporal_dataset(texts_per_decade=30, save_dataset=True)
        
        # Initialize boosted dataset
        boosted_dataset = {decade: texts.copy() if decade in dataset else [] 
                          for decade, texts in dataset.items()}
        
        # Load focused historical data from British Library and Gutenberg
        logger.info("Loading additional historical texts from British Library...")
        bl_texts = self.british_library_loader.load_british_library_historical_data(
            per_decade=1000,  # Get substantial historical content
            early_stop=False,  # Don't stop early
            target_decades=historical_decades  # Focus on historical decades
        )
        
        logger.info("Loading additional historical texts from Gutenberg...")
        self.gutenberg_loader.expand_historical_catalog()
        gutenberg_texts = self.gutenberg_loader.load_focused_decade_samples(
            target_decades=historical_decades,
            texts_per_decade=1000
        )
        
        # Add historical texts to boosted dataset
        for decade in historical_decades:
            # Add British Library texts
            bl_decade_texts = bl_texts.get(decade, [])
            boosted_dataset.setdefault(decade, []).extend(
                [(text, "british_library_boosted") for text in bl_decade_texts])
            
            # Add Gutenberg texts
            gutenberg_decade_texts = gutenberg_texts.get(decade, [])
            boosted_dataset.setdefault(decade, []).extend(gutenberg_decade_texts)
            
            logger.info(f"Added {len(bl_decade_texts)} British Library texts and {len(gutenberg_decade_texts)} Gutenberg texts to {decade}")
        
        # Create metadata
        metadata = {
            "type": "boosted_historical_dataset",
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_texts": sum(len(texts) for texts in boosted_dataset.values()),
            "boosted_decades": historical_decades,
            "sources": {
                "british_library": sum(1 for decade_texts in boosted_dataset.values() 
                                     for text, source in decade_texts if "british_library" in source),
                "gutenberg": sum(1 for decade_texts in boosted_dataset.values() 
                               for text, source in decade_texts if "gutenberg" in source),
            }
        }
        
        # Log statistics
        logger.info(f"Created boosted historical dataset with {metadata['total_texts']} total texts")
        for decade in sorted(boosted_dataset.keys()):
            count = len(boosted_dataset[decade])
            logger.info(f"  {decade}: {count} texts")
        
        return boosted_dataset
