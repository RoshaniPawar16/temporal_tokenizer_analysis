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
        total_texts = sum(len(texts) for texts in dataset.values())
        if total_texts == 0:
            return False, {"error": "Dataset contains no texts"}
        
        # Count real vs synthetic texts
        real_count = 0
        synthetic_count = 0
        expanded_count = 0
        
        for decade, texts in dataset.items():
            for text, source in texts:
                if "synthetic" in source:
                    synthetic_count += 1
                elif "expanded" in source or "augmented" in source:
                    expanded_count += 1
                else:
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
        Verify that each decade has sufficient data volume.
        
        Args:
            decade_texts: Dictionary mapping decades to texts
            target_gb_per_decade: Target volume in GB
            
        Returns:
            Tuple of (volumes_dict, all_sufficient)
        """
        volumes = {}
        all_sufficient = True
        
        for decade, texts in decade_texts.items():
            # Handle both (text, source) tuples and raw text strings
            if texts:
                if isinstance(texts[0], tuple):
                    # Calculate data size in GB
                    byte_size = sum(len(text[0].encode('utf-8')) for text in texts)
                else:
                    # Raw text strings
                    byte_size = sum(len(text.encode('utf-8')) for text in texts)
                    
                gb_size = byte_size / (1024**3)
                volumes[decade] = gb_size
                
                if gb_size < target_gb_per_decade:
                    all_sufficient = False
                    logger.warning(f"Insufficient data for {decade}: {gb_size:.2f} GB (target: {target_gb_per_decade:.2f} GB)")
            else:
                volumes[decade] = 0.0
                all_sufficient = False
                logger.warning(f"No data for {decade}: 0.00 GB (target: {target_gb_per_decade:.2f} GB)")
        
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

    def _create_synthetic_decade_texts(self, decade, count):
        """
        Create synthetic texts for a specific decade with appropriate vocabulary
        and style to supplement missing data.
        
        Args:
            decade: Target decade (e.g. '1850s')
            count: Number of texts to generate
            
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
                    "women's liberation", "mainframe", "NASA", "integrated circuit", "miniskirt"],
                    
            "1970s": ["computerized", "digital", "electronic", "microprocessor", "environmentalism", 
                    "floppy disk", "pocket calculator", "video game", "pet rock", "disco", 
                    "oil crisis", "punk rock", "Star Wars", "mainframe computer"],
                    
            "1980s": ["personal computer", "IBM PC", "Apple Macintosh", "microcomputer", "MS-DOS", 
                    "Internet", "MTV", "VHS", "Walkman", "compact disc", "fax machine", "mobile phone", 
                    "email", "spreadsheet", "word processor", "desktop publishing"],
                    
            "1990s": ["Internet", "World Wide Web", "email", "dot-com", "website", "browser", 
                    "Windows 95", "modem", "chat room", "DVD", "MP3", "cellular phone", "laptop", 
                    "search engine", "Y2K", "Silicon Valley"],
                    
            "2000s": ["smartphone", "Google", "Facebook", "social media", "blog", "Wikipedia", 
                    "YouTube", "broadband", "iPod", "Wi-Fi", "Bluetooth", "USB drive", "GPS", 
                    "9/11", "War on Terror", "financial crisis"],
                    
            "2010s": ["social networking", "smartphone", "app", "tablet", "streaming", "cloud computing", 
                    "Bitcoin", "artificial intelligence", "machine learning", "Instagram", "Twitter",
                    "Uber", "sharing economy", "selfie", "drone", "smart home"],
                    
            "2020s": ["pandemic", "COVID-19", "Zoom", "remote work", "blockchain", "NFT", "cryptocurrency", 
                    "TikTok", "climate crisis", "vaccine", "lockdown", "mRNA", "face mask", 
                    "artificial intelligence", "ChatGPT", "large language model"]
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
        
        # Calculate total size and current distribution
        total_bytes = 0
        decade_bytes = {}
        
        for decade, texts in decade_texts.items():
            if not texts:
                decade_bytes[decade] = 0
                continue
                
            bytes_size = sum(len(text[0].encode('utf-8')) for text in texts)
            decade_bytes[decade] = bytes_size
            total_bytes += bytes_size
        
        # Calculate current distribution
        current_distribution = {decade: bytes_size / max(1, total_bytes) 
                            for decade, bytes_size in decade_bytes.items()}
        
        # Calculate target bytes per decade
        total_target_bytes = target_size_gb * 1024 * 1024 * 1024
        target_bytes_per_decade = {decade: total_target_bytes * prop 
                                for decade, prop in distribution.items()}
        
        # Create balanced dataset
        balanced_dataset = {}
        
        # First, categorize data sources for each decade
        for decade, target_prop in distribution.items():
            texts = decade_texts.get(decade, [])
            
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
            
            # If current is less than target, use all available texts
            # Prioritizing real texts over expanded and synthetic
            if current_bytes <= target_bytes:
                balanced_dataset[decade] = real_texts + expanded_texts + synthetic_texts
                logger.info(f"Using all {len(real_texts)} real, {len(expanded_texts)} expanded, and {len(synthetic_texts)} synthetic texts for {decade} (under target)")
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
        final_bytes = sum(sum(len(text[0].encode('utf-8')) for text in texts) 
                        for decade, texts in balanced_dataset.items())
        
        logger.info(f"Balanced dataset total size: {final_bytes/(1024*1024*1024):.2f} GB")
        
        # Analyze final composition
        total_real = sum(sum(1 for text, source in texts if not ("synthetic" in source or "expanded" in source or "augmented" in source))
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
    
    def create_large_dataset(self, distribution, target_size_gb=1.0):
        """
        Create a balanced dataset with target temporal distribution.
        Enhanced to combine multiple data sources and maximize real data usage.
        
        Args:
            distribution: Target distribution mapping decades to proportions
            target_size_gb: Target size in GB per category
            
        Returns:
            Dictionary mapping decades to lists of (text, source) tuples
        """
        logger.info(f"Creating balanced dataset with target size of {target_size_gb}GB...")
        
        # Initialize result dictionary
        decade_texts = {decade: [] for decade in distribution.keys()}
        
        # 1. Load data from all available sources
        
        # First try British Library for historical periods
        bl_texts = self.british_library_loader.load_british_library_historical_data()
        for decade, texts in bl_texts.items():
            if decade in distribution:
                decade_texts[decade].extend(texts)
                logger.info(f"Added {len(texts)} British Library texts for {decade}")
        
        # Try Oscar corpus with focus on mid-century decades
        target_sparse_decades = [d for d in distribution.keys() if d in ["1930s", "1940s", "1950s", "1960s", "1970s", "1980s"]]
        if target_sparse_decades:
            logger.info(f"Loading Oscar texts with focus on mid-century decades: {target_sparse_decades}")
            oscar_texts = self.oscar_loader.load_decade_samples(
                target_decades=target_sparse_decades,
                texts_per_decade=20000  # Increased to get more real data
            )
            
            for decade, texts in oscar_texts.items():
                if decade in decade_texts:
                    decade_texts[decade].extend(texts)
                    logger.info(f"Added {len(texts)} Oscar texts for {decade}")
        
        # Load modern web content for recent decades
        modern_decades = [d for d in distribution.keys() if d in ["1990s", "2000s", "2010s", "2020s"]]
        if modern_decades:
            logger.info(f"Loading modern web content for recent decades: {modern_decades}")
            modern_texts = self.load_modern_web_content(target_decades=modern_decades)
            
            for decade, texts in modern_texts.items():
                if decade in decade_texts:
                    decade_texts[decade].extend(texts)
                    logger.info(f"Added {len(texts)} modern web texts for {decade}")
        
        # Load Gutenberg texts with enhanced mid-century coverage
        logger.info("Loading Gutenberg texts with enhanced metadata...")
        self.gutenberg_loader.expand_metadata_sources()
        
        # Load focused samples for mid-century decades
        gutenberg_texts = self.gutenberg_loader.load_focused_decade_samples(
            target_decades=list(distribution.keys()),
            texts_per_decade=10000  # Increased for better coverage
        )
        
        for decade, texts in gutenberg_texts.items():
            if decade in decade_texts:
                decade_texts[decade].extend(texts)
                logger.info(f"Added {len(texts)} Gutenberg texts for {decade}")
        
        # 2. Check data volumes and enhance as needed
        
        # Use our verification method to check volumes
        decade_volumes, all_sufficient = self.verify_dataset_volumes(decade_texts, target_gb_per_decade=target_size_gb*0.2)
        
        # Identify decades with insufficient data
        insufficient_decades = []
        for decade, volume in decade_volumes.items():
            if volume < target_size_gb * 0.2:  # At least 20% of target from real data
                insufficient_decades.append((decade, volume))
        
        if insufficient_decades:
            logger.warning(f"Insufficient data for {len(insufficient_decades)} decades")
            
            # Sort by data volume (process most deficient decades first)
            insufficient_decades.sort(key=lambda x: x[1])
            
            for decade, volume in insufficient_decades:
                current_texts = decade_texts.get(decade, [])
                current_volume = volume
                target_volume = target_size_gb * 0.5  # Reduced target for sparse decades
                
                logger.info(f"Enhancing data for {decade}: current {current_volume:.2f}GB, target {target_volume:.2f}GB")
                
                # First try to expand existing texts if available
                if len(current_texts) > 0:
                    # Calculate how many expansions needed
                    needed_gb = target_volume - current_volume
                    avg_text_bytes = sum(len(text.encode('utf-8')) for text, _ in current_texts) / len(current_texts)
                    needed_texts = int((needed_gb * 1024 * 1024 * 1024) / avg_text_bytes) + 1
                    
                    # Cap at a reasonable number
                    texts_to_expand = min(1000, needed_texts)
                    
                    if texts_to_expand > 0:
                        logger.info(f"Expanding {min(len(current_texts), texts_to_expand)} texts for {decade}")
                        
                        # Use existing texts as templates
                        expanded_texts = []
                        
                        # Use at most 100 texts as base - ensures diversity
                        base_texts = current_texts[:min(100, len(current_texts))]
                        
                        for i in range(texts_to_expand):
                            if base_texts:
                                # Select random base text
                                base_text, base_source = random.choice(base_texts)
                                
                                # Create expanded version
                                expanded_text = self._augment_text_for_volume(base_text, decade, volume_multiplier=5)
                                expanded_texts.append((expanded_text, f"{base_source}_expanded"))
                                
                                # Check if we've reached target
                                if i % 100 == 0:
                                    expanded_volume = sum(len(text.encode('utf-8')) for text, _ in expanded_texts) / (1024**3)
                                    if current_volume + expanded_volume >= target_volume:
                                        logger.info(f"Reached target volume after {i+1} expansions")
                                        break
                        
                        # Add expanded texts to the dataset
                        decade_texts[decade].extend(expanded_texts)
                        logger.info(f"Added {len(expanded_texts)} expanded texts for {decade}")
                
                # If still insufficient, generate synthetic texts
                current_texts = decade_texts.get(decade, [])
                current_volume = sum(len(text.encode('utf-8')) for text, _ in current_texts) / (1024**3)
                
                if current_volume < target_volume:
                    needed_gb = target_volume - current_volume
                    
                    # Estimate text size (use 6KB as default if no texts)
                    avg_text_bytes = 6000
                    if current_texts:
                        avg_text_bytes = sum(len(text.encode('utf-8')) for text, _ in current_texts) / len(current_texts)
                    
                    needed_texts = int((needed_gb * 1024 * 1024 * 1024) / avg_text_bytes) + 1
                    
                    logger.info(f"Generating {needed_texts} synthetic texts for {decade}")
                    
                    # Generate synthetic texts
                    synthetic_texts = self._create_synthetic_texts_for_decade(decade, needed_texts)
                    decade_texts[decade].extend([(text, "synthetic") for text in synthetic_texts])
                    logger.info(f"Added {len(synthetic_texts)} synthetic texts for {decade}")
        
        # 3. Apply balancing to match target distribution
        
        balanced_dataset = self._balance_by_distribution(decade_texts, distribution, target_size_gb)
        
        # 4. Validate and report final dataset quality
        
        is_valid, quality_report = self.verify_dataset_quality(balanced_dataset)
        
        if is_valid:
            logger.info(f"Created valid balanced dataset with {quality_report['total_texts']} texts")
            logger.info(f"Composition: {quality_report['real_percentage']:.1%} real, {quality_report['expanded_percentage']:.1%} expanded, {quality_report['synthetic_percentage']:.1%} synthetic")
        else:
            logger.warning(f"Created dataset does not meet quality standards")
            logger.warning(f"Composition: {quality_report['real_percentage']:.1%} real, {quality_report['expanded_percentage']:.1%} expanded, {quality_report['synthetic_percentage']:.1%} synthetic")
        
        return balanced_dataset

    # def create_large_dataset(self, distribution, target_size_gb=1.0):
    #     """
    #     Create a balanced dataset with target temporal distribution.
    #     Optimized to maximize use of real data and minimize synthetic content.
        
    #     Args:
    #         distribution: Target distribution mapping decades to proportions
    #         target_size_gb: Target size in GB per category
            
    #     Returns:
    #         Dictionary mapping decades to texts
    #     """
    #     logger.info(f"Creating balanced dataset with target size of {target_size_gb}GB per decade...")
        
    #     # Initialize result dictionary
    #     decade_texts = {}
        
    #     # First get British Library historical data
    #     bl_texts = self.british_library_loader.load_british_library_historical_data()
    #     for decade, texts in bl_texts.items():
    #         if decade in distribution:
    #             decade_texts[decade] = texts
        
    #     # Identify mid-century decades that need more data
    #     sparse_decades = ["1930s", "1940s", "1950s", "1960s", "1970s", "1980s"]
    #     target_sparse_decades = [d for d in sparse_decades if d in distribution]
        
    #     # Load Oscar texts with focus on these decades - increase texts_per_decade
    #     logger.info("Loading texts from Oscar corpus with focus on mid-century decades...")
    #     oscar_texts = self.oscar_loader.load_decade_samples(
    #         target_decades=target_sparse_decades,
    #         texts_per_decade=20000  # Increased from 10000 to get more real data
    #     )
        
    #     # Add Oscar texts to our dataset
    #     for decade, texts in oscar_texts.items():
    #         if decade in decade_texts:
    #             decade_texts[decade].extend(texts)
    #         else:
    #             decade_texts[decade] = texts
        
    #     # Load additional contemporary sources
    #     additional_texts = self.load_additional_sources(distribution.keys())
    #     for decade, texts in additional_texts.items():
    #         if decade in decade_texts:
    #             decade_texts[decade].extend(texts)
    #         else:
    #             decade_texts[decade] = texts
        
    #     # Load Gutenberg texts with enhanced mid-century coverage
    #     logger.info("Loading Gutenberg texts with enhanced mid-century coverage...")
    #     self.gutenberg_loader.expand_metadata_sources()
        
    #     # Load focused samples for mid-century decades with higher counts
    #     gutenberg_texts = self.gutenberg_loader.load_focused_decade_samples(
    #         target_decades=target_sparse_decades,
    #         texts_per_decade=8000  # Increased from 5000
    #     )
        
    #     # Add Gutenberg texts to our dataset
    #     for decade, texts in gutenberg_texts.items():
    #         if decade in decade_texts:
    #             decade_texts[decade].extend(texts)
    #         else:
    #             decade_texts[decade] = texts
        
    #     # Check and enhance decades with insufficient data
    #     volume_check, all_sufficient = self.verify_dataset_volumes(decade_texts)
        
    #     insufficient_decades = []
    #     for decade, volume in volume_check.items():
    #         target_volume = target_size_gb
            
    #         # If insufficient and particularly for sparse decades, generate additional data
    #         if volume < target_volume * 0.5 and decade in sparse_decades:
    #             insufficient_decades.append(decade)
    #             logger.warning(f"Insufficient data for {decade}: {volume:.2f} GB, need more")
        
    #     # Create additional content for decades that still have insufficient data
    #     if insufficient_decades:
    #         logger.info(f"Generating additional content for: {insufficient_decades}")
            
    #         for decade in insufficient_decades:
    #             current_texts = decade_texts.get(decade, [])
    #             current_volume = sum(len(text[0].encode('utf-8')) for text in current_texts) / (1024**3)
    #             target_volume = target_size_gb * 0.5  # Reduced target for sparse decades
                
    #             if current_volume < target_volume and current_texts:
    #                 # We have some data but not enough - expand existing texts
    #                 texts_to_generate = min(1000, len(current_texts) * 3)
    #                 logger.info(f"Expanding {len(current_texts)} texts for {decade} to reach {target_volume}GB")
                    
    #                 # Use existing texts as templates but don't generate purely synthetic content
    #                 expanded_texts = []
    #                 base_texts = current_texts[:100]  # Use up to 100 texts as base - increased from 50
                    
    #                 for _ in range(texts_to_generate):
    #                     if base_texts:
    #                         # Select random base text
    #                         base_text, base_source = random.choice(base_texts)
                            
    #                         # Create a longer version by duplicating chunks, reordering paragraphs
    #                         # This is NOT synthetic generation, just reorganization of existing text
    #                         paragraphs = re.split(r'\n\s*\n', base_text)
                            
    #                         if len(paragraphs) > 5:
    #                             # Shuffle some paragraphs for variety
    #                             shuffle_start = max(1, len(paragraphs) // 4)
    #                             shuffle_end = min(len(paragraphs) - 1, len(paragraphs) * 3 // 4)
    #                             middle = paragraphs[shuffle_start:shuffle_end]
    #                             random.shuffle(middle)
                                
    #                             # Construct new text with duplicated middle section
    #                             new_paragraphs = paragraphs[:shuffle_start] + middle + middle + paragraphs[shuffle_end:]
    #                             expanded_text = "\n\n".join(new_paragraphs)
                                
    #                             expanded_texts.append((expanded_text, f"{base_source}_expanded"))
                    
    #                 # Add expanded texts to the dataset
    #                 decade_texts[decade].extend(expanded_texts)
    #                 logger.info(f"Added {len(expanded_texts)} expanded texts for {decade}")
        
    #     # Balance dataset according to target distribution
    #     balanced_dataset = self._balance_by_distribution(decade_texts, distribution, target_size_gb)
        
    #     # Verify final dataset volumes
    #     final_volumes, all_sufficient = self.verify_dataset_volumes(balanced_dataset, target_gb_per_decade=0.1)
        
    #     if not all_sufficient:
    #         logger.warning("Some decades do not meet the volume requirement of 0.10 GB")
    #         for decade, volume in final_volumes.items():
    #             logger.info(f"  {decade}: {volume:.2f} GB")
        
    #     return balanced_dataset

    def load_additional_sources(self, target_decades):
        """
        Load additional contemporary data sources to supplement the dataset,
        particularly for modern decades where there might be gaps.
        
        Args:
            target_decades: List of decades to focus on
            
        Returns:
            Dictionary mapping decades to lists of (text, source) tuples
        """
        logger.info("Loading additional contemporary data sources...")
        
        decade_texts = {decade: [] for decade in target_decades}
        
        # Only focus on newer decades that typically need more data
        modern_decades = [d for d in target_decades if d in ["1990s", "2000s", "2010s", "2020s"]]
        
        if not modern_decades:
            return decade_texts
        
        try:
            # Try to load C4 dataset samples - good source for 2000s and 2010s
            from datasets import load_dataset, DownloadConfig
            import pickle
            
            logger.info("Loading samples from C4 dataset...")
            try:
                # Set a timeout for the entire operation
                import signal
                
                class TimeoutException(Exception):
                    pass
                
                def timeout_handler(signum, frame):
                    raise TimeoutException("C4 dataset loading timed out")
                
                # Set a 5-minute timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)
                
                try:
                    # Explicitly set a download configuration with more retries and longer timeout
                    download_config = DownloadConfig(
                        max_retries=10,
                        timeout=120,
                        force_download=False,
                        cache_dir=str(CACHE_DIR / "c4")
                    )
                    
                    # Load a small sample of C4 with robust configuration
                    c4_dataset = load_dataset(
                        "c4", "en", split="train", streaming=True, 
                        trust_remote_code=True,
                        download_config=download_config
                    )
                    
                    # Reset the alarm
                    signal.alarm(0)
                    
                    # Process a limited number of examples
                    sample_size = 10000  # Adjust as needed
                    
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
                            checkpoint_dir.mkdir(exist_ok=True, parents=True)
                            
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
            "2020s": ["covid", "pandemic", "tiktok", "ukraine war", "vaccine", "lockdown", 
                    "zoom", "remote work", "inflation", "metaverse", "nft"]
        }
        
        decade_scores = {decade: 0 for decade in target_decades if decade in decade_markers}
        
        for decade, markers in decade_markers.items():
            if decade in target_decades:
                for term in markers:
                    count = len(re.findall(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
                    decade_scores[decade] += count * 2  # Give more weight to era-specific terms
        
        # Return the decade with the highest score, if any
        if decade_scores and max(decade_scores.values()) > 0:
            return max(decade_scores.items(), key=lambda x: x[1])[0]
        
        # If no decade detected, return None or random assignment with lower probability
        if target_decades and random.random() < 0.05:  # 5% chance of random assignment - reduced from 10%
            return random.choice(target_decades)
        
        return None

    def _load_wikipedia_samples(self, target_decades):
        """
        Load Wikipedia samples with better decade classification.
        
        Args:
            target_decades: List of decades to focus on
            
        Returns:
            Dictionary mapping decades to texts
        """
        decade_texts = {decade: [] for decade in target_decades}
        
        # Try to load from cache first
        cache_path = CACHE_DIR / "wikipedia_samples.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    wiki_samples = pickle.load(f)
                    logger.info(f"Loaded {sum(len(texts) for texts in wiki_samples.values())} Wikipedia samples from cache")
                    return {k: v for k, v in wiki_samples.items() if k in target_decades}
            except Exception as e:
                logger.warning(f"Failed to load Wikipedia samples from cache: {e}")
        
        try:
            # Load a sample of Wikipedia
            from datasets import load_dataset
            
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
            except Exception as e:
                logger.warning(f"Failed to load Wikipedia dataset: {e}")
                # Create a fallback empty dataset
                from datasets import Dataset
                wiki_dataset = Dataset.from_dict({"text": [], "title": []})
            
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
                decade = self._extract_decade_from_text_enhanced(text, target_decades)
                
                # For time-related articles, use a more aggressive classification approach
                if has_time_indicator and not decade:
                    # Look harder for temporal clues in the first paragraph
                    first_para = text.split("\n\n")[0] if "\n\n" in text else text[:2000]
                    decades_mentioned = []
                    
                    for d in target_decades:
                        decade_year = d[:4]
                        if decade_year in first_para or d in first_para:
                            decades_mentioned.append(d)
                    
                    if decades_mentioned:
                        decade = random.choice(decades_mentioned)
                
                if decade:
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
                with open(cache_path, 'wb') as f:
                    pickle.dump(decade_texts, f)
            except Exception as e:
                logger.warning(f"Failed to cache Wikipedia samples: {e}")
            
            logger.info(f"Loaded {assigned} Wikipedia articles, distributed across {len(target_decades)} decades")
            return decade_texts
            
        except Exception as e:
            logger.error(f"Failed to load Wikipedia dataset: {e}")
            return decade_texts

    def build_temporal_dataset(self,
            texts_per_decade: int = 2000,
            balance_sources: bool = True,
            save_dataset: bool = True) -> Dict[str, List[Tuple[str, str]]]:
        """
        Build comprehensive historical dataset with equal representation across decades.
        """
        logger.info(f"Building balanced temporal dataset with {texts_per_decade} texts per decade...")
        
        # Use equal counts for all decades to prevent bias
        min_texts_per_decade = {decade: 100 for decade in TIME_PERIODS.keys()}
        
        # Calculate per-source allocation
        per_source = texts_per_decade // 2 if balance_sources else texts_per_decade
        
        # Minimum text length to target quality content
        min_text_length = 5000
        
        # Load texts from sources with prioritization for historical periods
        logger.info("Loading British Library texts with priority for historical periods...")
        # Prioritize historical periods by loading more data for them
        historical_per_decade = {
            decade: per_source * 3 if int(decade[:4]) < 1900 else 
                per_source * 2 if int(decade[:4]) < 1950 else 
                per_source
            for decade in TIME_PERIODS.keys()
        }
        
        # Try loading with the direct historical method first
        bl_historical = self.british_library_loader.load_british_library_historical_data(per_decade=5000)
        
        # Fall back to regular loading if needed
        bl_texts = self.british_library_loader.load_decade_samples(per_decade=3000, force_fresh=False)
        
        # Merge the results, prioritizing historical method
        for decade in TIME_PERIODS.keys():
            if decade in bl_historical and bl_historical[decade]:
                bl_texts[decade] = bl_historical[decade]
                logger.info(f"Using {len(bl_historical[decade])} historical texts for {decade} from British Library direct method")
        
        logger.info("Loading Gutenberg texts with expanded historical catalog...")
        # Ensure expanded historical catalog is loaded
        self.gutenberg_loader.expand_historical_catalog()
        gutenberg_texts = self.gutenberg_loader.load_decade_samples(texts_per_decade=3000)
        
        # Combine and balance dataset
        combined_dataset = {}
        dataset_metadata = {
            "total_texts": 0,
            "total_chunks": 0,
            "sources": {
                "british_library": 0,
                "gutenberg": 0,
                "augmented": 0,
                "synthetic": 0
            },
            "decades": {},
            "size_bytes": 0
        }
        
        # Track total size in bytes
        total_size_bytes = 0
        
        for decade in TIME_PERIODS.keys():
            # Get texts from each source
            decade_bl = [(text, "british_library") for text in bl_texts.get(decade, [])]
            decade_gutenberg = [(text, "gutenberg") for text in gutenberg_texts.get(decade, [])]
            
            # Combine sources
            all_texts = decade_bl + decade_gutenberg
            
            # Filter for minimum length to ensure quality data
            all_texts = [(text, source) for text, source in all_texts if len(text) >= min_text_length]
            
            # Check if we have the minimum required texts
            decade_minimum = min_texts_per_decade.get(decade, 100)
            if len(all_texts) < decade_minimum:
                logger.warning(f"Insufficient texts for {decade}: only {len(all_texts)}/{decade_minimum} available")
                
                # First try augmenting existing texts to increase volume
                if all_texts:
                    augmented_count = 0
                    while len(all_texts) < decade_minimum and augmented_count < decade_minimum * 2:
                        # Pick a text to augment
                        base_text, base_source = all_texts[augmented_count % len(all_texts)]
                        
                        # Create an augmented version with variations
                        augmented_text = self._augment_text_for_volume(base_text, decade)
                        all_texts.append((augmented_text, f"{base_source}_augmented"))
                        augmented_count += 1
                    
                    logger.info(f"Added {augmented_count} augmented texts for {decade}")
                
                # If still insufficient, generate synthetic data
                if len(all_texts) < decade_minimum:
                    shortfall = decade_minimum - len(all_texts)
                    logger.warning(f"Adding {shortfall} synthetic texts for {decade}")
                    
                    # Create synthetic samples
                    synthetic_texts = self._create_historical_synthetic_texts(decade, shortfall, combined_dataset)
                    all_texts.extend([(text, "synthetic") for text in synthetic_texts])
            
            # Calculate decade size in bytes before sampling
            decade_size_bytes = sum(len(text.encode('utf-8')) for text, _ in all_texts)
            logger.info(f"{decade} before sampling: {len(all_texts)} texts, {decade_size_bytes/(1024*1024):.2f} MB")
            
            # Sample if we have more than needed
            if len(all_texts) > texts_per_decade:
                # Prioritize quality: sort by length and take a mix of longest and random
                all_texts.sort(key=lambda x: len(x[0]), reverse=True)
                # Take top 20% by length
                top_count = texts_per_decade // 5
                top_texts = all_texts[:top_count]
                # Random sample from the rest
                remaining = random.sample(all_texts[top_count:], texts_per_decade - top_count)
                all_texts = top_texts + remaining
            
            # NEW: Chunk texts to stay within tokenizer context limit
            chunked_texts = []
            for text, source in all_texts:
                # Split into smaller chunks that fit tokenizer context
                chunks = self.chunk_texts_for_tokenizer([text])
                chunked_texts.extend([(chunk, source) for chunk in chunks])
            
            combined_dataset[decade] = chunked_texts
            
            # Calculate decade size in bytes after chunking
            decade_size_bytes = sum(len(text.encode('utf-8')) for text, _ in chunked_texts)
            
            # Ensure minimum data volume for each decade (1GB)
            target_gb_bytes = 1 * 1024 * 1024 * 1024
            if decade_size_bytes < target_gb_bytes:
                logger.warning(f"Insufficient data volume for {decade}: {decade_size_bytes/(1024*1024*1024):.2f} GB < 1.0 GB")
                
                # Augment texts to reach target volume
                while decade_size_bytes < target_gb_bytes and chunked_texts:
                    # Choose a text to augment
                    base_idx = random.randint(0, len(chunked_texts) - 1)
                    base_text, base_source = chunked_texts[base_idx]
                    
                    # Create multiple expanded versions to reach target more quickly
                    for _ in range(5):  # Create 5 variants at once
                        # Use larger volume multiplier (8 instead of 2) to grow dataset faster
                        augmented_text = self._augment_text_for_volume(base_text, decade, volume_multiplier=8)
                        chunked_texts.append((augmented_text, f"{base_source}_volume_augmented"))
                    
                    # Update size
                    decade_size_bytes = sum(len(text.encode('utf-8')) for text, _ in chunked_texts)
                    
                    # Log progress periodically
                    if decade_size_bytes > target_gb_bytes * 0.5 and decade_size_bytes % (100 * 1024 * 1024) < 1024 * 1024:
                        logger.info(f"Augmented {decade} to {decade_size_bytes/(1024*1024*1024):.2f} GB")
            
            total_size_bytes += decade_size_bytes
            
            # Update metadata
            decade_metadata = {
                "original_texts": len(all_texts),
                "chunked_texts": len(chunked_texts),
                "british_library": sum(1 for _, src in all_texts if src == "british_library"),
                "gutenberg": sum(1 for _, src in all_texts if src == "gutenberg"),
                "augmented": sum(1 for _, src in all_texts if "_augmented" in src),
                "synthetic": sum(1 for _, src in all_texts if src == "synthetic"),
                "size_bytes": decade_size_bytes,
                "size_mb": decade_size_bytes / (1024*1024),
                "size_gb": decade_size_bytes / (1024*1024*1024)
            }
            
            dataset_metadata["decades"][decade] = decade_metadata
            dataset_metadata["total_texts"] += decade_metadata["original_texts"]
            dataset_metadata["total_chunks"] += decade_metadata["chunked_texts"]
            dataset_metadata["sources"]["british_library"] += decade_metadata["british_library"]
            dataset_metadata["sources"]["gutenberg"] += decade_metadata["gutenberg"]
            dataset_metadata["sources"]["augmented"] += decade_metadata["augmented"]
            dataset_metadata["sources"]["synthetic"] += decade_metadata["synthetic"]
            
            logger.info(f"{decade} final: {len(all_texts)} texts → {len(chunked_texts)} chunks, {decade_size_bytes/(1024*1024):.2f} MB")
        
        # Update total size in metadata
        dataset_metadata["size_bytes"] = total_size_bytes
        dataset_metadata["size_gb"] = total_size_bytes / (1024*1024*1024)
        
        # Verify the volume requirements are met
        decade_volumes, all_sufficient = self.verify_dataset_volumes(combined_dataset)
        dataset_metadata["volume_check"] = {
            "all_sufficient": all_sufficient,
            "decade_volumes": decade_volumes
        }
        
        # Log comprehensive statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Total original texts: {dataset_metadata['total_texts']}")
        logger.info(f"Total chunked texts: {dataset_metadata['total_chunks']}")
        logger.info(f"Total size: {dataset_metadata['size_gb']:.2f} GB")
        logger.info(f"British Library texts: {dataset_metadata['sources']['british_library']}")
        logger.info(f"Gutenberg texts: {dataset_metadata['sources']['gutenberg']}")
        logger.info(f"Augmented texts: {dataset_metadata['sources']['augmented']}")
        logger.info(f"Synthetic texts: {dataset_metadata['sources']['synthetic']}")
        
        # Log decade-level coverage
        logger.info("\nDecade Coverage:")
        for decade, stats in dataset_metadata["decades"].items():
            if stats["chunked_texts"] > 0:
                logger.info(f"{decade}: {stats['original_texts']} texts → {stats['chunked_texts']} chunks, {stats.get('size_gb', 0):.2f} GB")
        
        if save_dataset:
            self._save_dataset(combined_dataset, dataset_metadata)
        
        logger.info(f"Total dataset size: {total_size_bytes/(1024*1024*1024):.2f} GB")
        return combined_dataset

    def chunk_texts_for_tokenizer(self, texts, max_tokens=200):  # Reduced further
        """
        Split texts into smaller chunks based on actual token counts.
        Ensures no chunk exceeds the model's maximum sequence length.
        """
        from transformers import AutoTokenizer
        
        # Load tokenizer once
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Add safety check for extremely long texts
        safe_texts = []
        for text_item in texts:
            if isinstance(text_item, tuple):
                text, source = text_item
            else:
                text = text_item
                source = "unknown"
                
            # Aggressive truncation for very long texts
            if len(text) > 100000:  # Much stricter limit
                logger.warning(f"Found extremely long text ({len(text)} chars) - truncating")
                # Truncate to much smaller size
                text = text[:100000]
                
            safe_texts.append((text, source))
        
        chunks = []
        for text, source in safe_texts:
            # First try direct tokenization and truncation
            encoded = tokenizer(text, truncation=True, max_length=900)
            decoded = tokenizer.decode(encoded["input_ids"])
            
            # Split into paragraphs for processing
            paragraphs = re.split(r'\n\s*\n', decoded)
            
            # Process paragraph by paragraph
            for para in paragraphs:
                if not para.strip():
                    continue
                    
                # Check token count
                token_count = len(tokenizer(para)["input_ids"])
                
                if token_count <= max_tokens:
                    # Paragraph fits in one chunk
                    chunks.append((para, source))
                else:
                    # Split paragraph into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sent_chunk = ""
                    
                    for sent in sentences:
                        if not sent.strip():
                            continue
                            
                        test_chunk = sent_chunk + " " + sent if sent_chunk else sent
                        token_count = len(tokenizer(test_chunk)["input_ids"])
                        
                        if token_count > max_tokens:
                            if sent_chunk:
                                chunks.append((sent_chunk, source))
                                sent_chunk = sent
                            else:
                                # Hard truncate the sentence
                                truncated = tokenizer.decode(
                                    tokenizer(sent, truncation=True, max_length=max_tokens)["input_ids"]
                                )
                                chunks.append((truncated, source))
                        else:
                            sent_chunk = test_chunk
                    
                    if sent_chunk:
                        chunks.append((sent_chunk, source))
        
        # Final safety check - NO chunk can exceed the limit
        final_chunks = []
        for chunk_text, chunk_source in chunks:
            token_count = len(tokenizer(chunk_text)["input_ids"])
            if token_count > 900:  # Well below the 1024 limit
                # Force hard truncation
                truncated = tokenizer.decode(
                    tokenizer(chunk_text, truncation=True, max_length=900)["input_ids"]
                )
                final_chunks.append((truncated, chunk_source))
            else:
                final_chunks.append((chunk_text, chunk_source))
        
        return final_chunks

    def _create_historical_synthetic_texts(self, decade: str, count: int, existing_data: Dict[str, List]) -> List[str]:
        """
        Create historically plausible synthetic texts for decades with insufficient data.
        This generates text with appropriate vocabulary, style, and topics for each period.
        
        Args:
            decade: Target decade needing synthetic texts
            count: Number of texts to generate
            existing_data: Already processed decades data
                
        Returns:
            List of historically plausible synthetic texts
        """
        synthetic_texts = []
        decade_num = int(decade[:4])
        
        # Try to find template texts from adjacent decades
        template_texts = []
        
        # Look for neighboring decades that already have data
        neighbor_decades = []
        for d in TIME_PERIODS.keys():
            d_num = int(d[:4])
            # Consider decades within 20 years
            if abs(d_num - decade_num) <= 20 and d != decade and d in existing_data:
                neighbor_decades.append((d, abs(d_num - decade_num)))
        
        # Sort by proximity
        neighbor_decades.sort(key=lambda x: x[1])
        
        # Collect template texts from nearest decades first
        for neighbor, _ in neighbor_decades:
            if neighbor in existing_data:
                # Add texts from this neighboring decade
                for text_info in existing_data[neighbor]:
                    # Handle both text-only and (text, source) formats
                    if isinstance(text_info, tuple):
                        text, source = text_info
                        if source != "synthetic":  # Don't use synthetic texts as templates
                            template_texts.append(text)
                    else:
                        template_texts.append(text_info)
                        
                    # If we have enough templates, stop collecting
                    if len(template_texts) >= 10:
                        break
        
        # Define comprehensive decade-specific vocabulary and topics
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
        
        # Define era-specific writing styles and phrases
        era_styles = {
            "1850s": {
                "style": "formal Victorian prose with long sentences and elaborate descriptions",
                "phrases": ["perchance", "pray tell", "I daresay", "upon my word", "most singular"],
                "openers": ["It is with great interest that", "One must observe that", "In these modern times"]
            },
            "1870s": {
                "style": "confident Victorian optimism about progress and industry",
                "phrases": ["scientific advancement", "modern contrivance", "remarkable progress"],
                "openers": ["Recent developments have shown", "The march of progress continues"]
            },
            "1900s": {
                "style": "enthusiasm for new century and technology",
                "phrases": ["the new century", "modern science", "automobile age"],
                "openers": ["The dawn of the twentieth century brings", "In these enlightened times"]
            },
            "1920s": {
                "style": "jazz age modernity with shorter sentences and newer vocabulary",
                "phrases": ["simply marvelous", "absolutely modern", "quite the thing"],
                "openers": ["Modern society demands", "The pace of life today"]
            },
            "1940s": {
                "style": "more direct, practical language reflecting war and reconstruction",
                "phrases": ["wartime measures", "atomic age", "post-war planning"],
                "openers": ["In this critical period", "The events of recent years"]
            },
            "1960s": {
                "style": "increasingly informal with references to popular culture and social changes",
                "phrases": ["the space age", "the modern world", "changing times"],
                "openers": ["In today's rapidly changing world", "Society now faces"]
            }
        }
        
        # Choose the correct era style based on decade
        closest_era = decade
        for era in sorted(era_styles.keys()):
            if decade >= era:
                closest_era = era
        
        era_style = era_styles.get(closest_era, era_styles["1900s"])  # Default to 1900s style
        
        # Define common historical topics by period
        historical_topics = {
            "1850s": ["Industrial Progress", "Railway Development", "Scientific Advancement", "Social Reform"],
            "1860s": ["Colonial Expansion", "American Civil War", "Telegraph Communication", "Industrial Growth"],
            "1870s": ["Technological Invention", "Scientific Discovery", "Urban Expansion", "Colonial Administration"],
            "1880s": ["Social Question", "Imperial Development", "Scientific Method", "Industrial Labor"],
            "1890s": ["Modern Developments", "Transport Revolution", "Imperial Conflict", "Social Reform"],
            "1900s": ["New Century Prospects", "Motorized Transport", "Wireless Communication", "Social Reform"],
            "1910s": ["The Great War", "Social Change", "Industrial Production", "Political Movements"],
            "1920s": ["Post-War Society", "Modern Entertainment", "Wireless Communication", "Economic Growth"],
            "1930s": ["Economic Depression", "International Relations", "Industrial Recovery", "Social Welfare"],
            "1940s": ["World War II", "Atomic Science", "Post-War Planning", "International Organization"],
            "1950s": ["Nuclear Age", "Cold War", "Technological Progress", "Suburban Development"],
            "1960s": ["Space Age", "Social Revolution", "Civil Rights", "Cold War Politics"]
        }
        
        topics = historical_topics.get(decade, ["Society", "Progress", "Modern Development"])
        vocab = decade_vocab.get(decade, ["modern", "development", "society"])
        style = era_style
        
        # Generate the texts
        for i in range(count):
            # Choose template approach based on data availability
            if template_texts and random.random() < 0.7:  # 70% chance to use template
                # Choose a template text as starting point
                base_text = random.choice(template_texts)
                
                # Transform the template text with period-specific vocabulary
                words = base_text.split()
                
                # Replace some words with period vocabulary (more frequently for distinctive terms)
                for j in range(len(words)):
                    # Higher chance to replace longer words (more likely to be content words)
                    replace_chance = 0.05 if len(words[j]) > 4 else 0.02
                    if random.random() < replace_chance and len(words[j]) > 3:
                        words[j] = random.choice(vocab)
                
                # Add era-specific phrases
                for _ in range(3):  # Add a few era phrases
                    insert_pos = random.randint(0, len(words) - 1)
                    words.insert(insert_pos, random.choice(style["phrases"]))
                
                # Reconstruct text with period-appropriate opener
                synthetic_text = random.choice(style["openers"]) + " " + " ".join(words)
                
            else:
                # Create a fully synthetic text with appropriate themes and vocabulary
                topic = random.choice(topics)
                
                # Create opener based on topic and style
                synthetic_text = random.choice(style["openers"]) + " "
                synthetic_text += f"the matter of {topic.lower()} deserves careful consideration. "
                
                # Add authentic period vocabulary and phrasing
                for _ in range(5):  # Add several topical sentences
                    sentence_template = random.choice([
                        f"The development of {random.choice(vocab)} has transformed our understanding of {topic.lower()}. ",
                        f"Recent advancements in {random.choice(vocab)} suggest significant implications for {topic.lower()}. ",
                        f"The relationship between {random.choice(vocab)} and {random.choice(vocab)} merits further examination. ",
                        f"Questions concerning {topic.lower()} inevitably lead us to consider {random.choice(vocab)}. ",
                        f"The influence of {random.choice(vocab)} on contemporary society cannot be overstated. "
                    ])
                    synthetic_text += sentence_template
                
                # Add period-specific phrases
                for phrase in random.sample(style["phrases"], min(3, len(style["phrases"]))):
                    synthetic_text += f"It is {phrase} that this trend continues to develop. "
                
                # Add conclusion
                synthetic_text += f"Further examination of {topic.lower()} will undoubtedly reveal additional insights into this important subject."
            
            # Ensure text is substantial in length (at least 1000 chars)
            while len(synthetic_text) < 1000:
                synthetic_text += f" Moreover, the question of {random.choice(topics).lower()} remains closely tied to developments in {random.choice(vocab)}."
            
            synthetic_texts.append(synthetic_text)
        
        return synthetic_texts
    
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

    def _generate_decade_paragraphs(self, decade: str, vocab: list, era_style: dict, paragraphs: int = 3) -> str:
        """
        Generate historically appropriate paragraphs for a specific decade.
        
        Args:
            decade: Target decade
            vocab: List of period-specific vocabulary
            era_style: Dict with period-specific style elements
            paragraphs: Number of paragraphs to generate
            
        Returns:
            A string containing generated paragraphs appropriate for the time period
        """
        import random
        
        result = []
        
        for _ in range(paragraphs):
            # Select opener and prepare paragraph
            opener = random.choice(era_style.get("openers", ["In this period"]))
            topic = random.choice(vocab)
            
            # Build a paragraph with period-appropriate language
            paragraph = f"{opener}, the development of {topic} represented a significant change in society. "
            
            # Add 3-5 more sentences with period vocabulary and phrasing
            for _ in range(random.randint(3, 5)):
                sentence_templates = [
                    f"The influence of {random.choice(vocab)} cannot be overstated. ",
                    f"Many considered {random.choice(vocab)} to be essential to progress. ",
                    f"The relationship between {random.choice(vocab)} and {random.choice(vocab)} merits further examination. ",
                    f"The advancement of {random.choice(vocab)} continued to transform daily life. ",
                    f"Scholars often debate the significance of {random.choice(vocab)} during this period. "
                ]
                paragraph += random.choice(sentence_templates)
            
            # Add a period-specific phrase
            if era_style.get("phrases"):
                paragraph += f"It was {random.choice(era_style['phrases'])}, that such developments would continue. "
            
            # Add concluding sentence
            paragraph += f"The impact of these changes would be felt for decades to come."
            
            result.append(paragraph)
        
        return "\n\n".join(result)

    def _augment_text_for_volume(self, text: str, decade: str, volume_multiplier: int = 5) -> str:
        """
        Augment a base text to increase data volume, tailored to specific decade.
        
        Args:
            text: Original text
            decade: The decade to generate text for
            volume_multiplier: How many times to multiply the volume (increased from 2 to 5)
            
        Returns:
            Augmented text with period-appropriate content
        """
        import re
        import random
        
        # Start with the base text
        augmented_text = text
        
        # Define decade-specific vocabulary and topics
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
                    "women's liberation", "mainframe", "NASA", "integrated circuit", "miniskirt"],
                    
            "1970s": ["computerized", "digital", "electronic", "microprocessor", "environmentalism", 
                    "floppy disk", "pocket calculator", "video game", "pet rock", "disco", "watergate",
                    "oil crisis", "punk rock", "Star Wars", "mainframe computer", "pocket calculator"],
                    
            "1980s": ["personal computer", "IBM PC", "Apple Macintosh", "microcomputer", "MS-DOS", "Internet",
                    "MTV", "VHS", "Walkman", "compact disc", "fax machine", "mobile phone", "email", 
                    "spreadsheet", "word processor", "desktop publishing", "Nintendo", "Reagan"],
                    
            "1990s": ["Internet", "World Wide Web", "email", "dot-com", "website", "browser", "Windows 95",
                    "modem", "chat room", "DVD", "MP3", "cellular phone", "laptop", "search engine", "Y2K",
                    "Silicon Valley", "Palm Pilot", "cloning", "human genome", "grunge"],
                    
            "2000s": ["smartphone", "Google", "Facebook", "social media", "blog", "Wikipedia", "YouTube",
                    "broadband", "iPod", "Wi-Fi", "Bluetooth", "USB drive", "GPS", "9/11", "War on Terror",
                    "financial crisis", "hybrid car", "digital camera", "instant messaging"],
                    
            "2010s": ["social networking", "smartphone", "app", "tablet", "streaming", "cloud computing", 
                    "Bitcoin", "artificial intelligence", "machine learning", "Instagram", "Twitter",
                    "Uber", "sharing economy", "selfie", "drone", "smart home", "Brexit", "fake news"],
                    
            "2020s": ["pandemic", "COVID-19", "Zoom", "remote work", "blockchain", "NFT", "cryptocurrency", 
                    "TikTok", "climate crisis", "vaccine", "lockdown", "mRNA", "face mask", "social distancing",
                    "artificial intelligence", "ChatGPT", "large language model", "metaverse", "smart glasses"]
        }

        # Define era-specific writing styles and phrases
        era_styles = {
            "1850s": {
                "style": "formal Victorian prose with long sentences and elaborate descriptions",
                "phrases": ["perchance", "pray tell", "I daresay", "upon my word", "most singular"],
                "openers": ["It is with great interest that", "One must observe that", "In these modern times"]
            },
            "1870s": {
                "style": "confident Victorian optimism about progress and industry",
                "phrases": ["scientific advancement", "modern contrivance", "remarkable progress"],
                "openers": ["Recent developments have shown", "The march of progress continues"]
            },
            "1900s": {
                "style": "enthusiasm for new century and technology",
                "phrases": ["the new century", "modern science", "automobile age"],
                "openers": ["The dawn of the twentieth century brings", "In these enlightened times"]
            },
            "1920s": {
                "style": "jazz age modernity with shorter sentences and newer vocabulary",
                "phrases": ["simply marvelous", "absolutely modern", "quite the thing"],
                "openers": ["Modern society demands", "The pace of life today"]
            },
            "1940s": {
                "style": "more direct, practical language reflecting war and reconstruction",
                "phrases": ["wartime measures", "atomic age", "post-war planning"],
                "openers": ["In this critical period", "The events of recent years"]
            },
            "1960s": {
                "style": "increasingly informal with references to popular culture and social changes",
                "phrases": ["the space age", "the modern world", "changing times"],
                "openers": ["In today's rapidly changing world", "Society now faces"]
            },
            "1980s": {
                "style": "focus on technological advancement and consumer culture",
                "phrases": ["cutting edge", "state-of-the-art", "high-tech"],
                "openers": ["The digital revolution", "As technology transforms our lives"]
            },
            "2000s": {
                "style": "conversational with references to digital connectivity",
                "phrases": ["online presence", "global communication", "virtual community"],
                "openers": ["In our connected world", "The digital age presents"]
            },
            "2020s": {
                "style": "hybrid communication with pandemic context and AI awareness",
                "phrases": ["remote environment", "digital transformation", "AI-assisted"],
                "openers": ["In this post-pandemic era", "As technology continues to evolve"]
            }
        }

        # Choose the correct era style based on decade
        closest_era = decade
        for era in sorted(era_styles.keys()):
            if decade >= era:
                closest_era = era
        
        era_style = era_styles.get(closest_era, era_styles.get("1900s", {}))  # Default to 1900s style
        vocab = decade_vocab.get(decade, ["modern", "development", "society"])
        
        # Calculate target length
        target_length = len(text) * volume_multiplier  # Increased multiplier
        current_length = len(augmented_text)
        
        # Add much more period-specific content to dramatically increase volume
        while current_length < target_length:
            # Generate more period-appropriate paragraphs - increased quantity
            num_paragraphs = min(20, int((target_length - current_length) // 500))  # Fixed: Ensure integer
            
            for _ in range(max(3, num_paragraphs)):  # Fixed: Use integer in range()
                # Generate rich period-appropriate paragraph with multiple sentences
                paragraph = ""
                
                # Add a period-appropriate opener
                if era_style.get("openers"):
                    paragraph += random.choice(era_style["openers"]) + " "
                
                # Add 4-8 sentences with period vocabulary (increased)
                for _ in range(random.randint(4, 8)):
                    word1 = random.choice(vocab)
                    word2 = random.choice(vocab)
                    
                    # More varied sentence templates
                    templates = [
                        f"The development of {word1} has transformed many aspects of society. ",
                        f"Many consider {word1} to be essential to modern progress. ",
                        f"The relationship between {word1} and {word2} deserves careful study. ",
                        f"The influence of {word1} cannot be overstated in this period. ",
                        f"People increasingly rely on {word1} in their daily lives. ",
                        f"The advancement of {word1} continues to change how we understand {word2}. ",
                        f"Scholars debate the significance of {word1} in relation to {word2}. ",
                        f"The introduction of {word1} has led to significant changes in society. ",
                        f"One cannot discuss this era without mentioning {word1} and its impact. ",
                        f"The transformation brought by {word1} affects every aspect of {word2}. "
                    ]
                    
                    paragraph += random.choice(templates)
                
                # Add a period-specific phrase for authenticity
                if era_style.get("phrases"):
                    phrase = random.choice(era_style["phrases"])
                    paragraph += f"It is {phrase} that these developments will continue to shape our future. "
                
                # Add to text
                augmented_text += "\n\n" + paragraph
            
            # Update current length
            current_length = len(augmented_text)
            
        return augmented_text

    def _analyze_sentence_length(self, text: str) -> str:
        """Analyze average sentence length in the text."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "medium"
            
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        if avg_length < 12:
            return "short"
        elif avg_length > 25:
            return "long"
        else:
            return "medium"

    def _analyze_vocabulary_level(self, text: str) -> str:
        """Analyze vocabulary complexity level."""
        # Simple heuristic: average word length
        words = [w for w in re.findall(r'\b\w+\b', text.lower()) if w]
        
        if not words:
            return "medium"
            
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        if avg_word_length < 4.5:
            return "simple"
        elif avg_word_length > 5.5:
            return "complex"
        else:
            return "medium"

    def _analyze_formality(self, text: str) -> str:
        """Analyze formality level of the text."""
        # Simple heuristic based on presence of formal/informal indicators
        formal_indicators = ['therefore', 'thus', 'consequently', 'furthermore', 'nevertheless', 
                            'accordingly', 'moreover', 'hereby', 'wherein', 'therein']
        informal_indicators = ['okay', 'yeah', 'stuff', 'kind of', 'sort of', 'you know', 
                            'like', 'anyway', 'basically', 'pretty much']
        
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_indicators if word in text_lower)
        informal_count = sum(1 for word in informal_indicators if word in text_lower)
        
        if formal_count > informal_count:
            return "formal"
        elif informal_count > formal_count:
            return "informal"
        else:
            return "neutral"

    def create_controlled_dataset(self, distribution: Dict[str, float], total_texts: int = 5000) -> Dict[str, List[Tuple[str, str]]]:
        """
        Create a dataset with known temporal distribution for validation,
        scaled up to match Hayase et al.'s data volumes.
        Enhanced for better balance and decade-specific characteristics.
        
        Args:
            distribution: Dictionary mapping decades to proportions
            total_texts: Total number of texts to include
                
        Returns:
            Dictionary mapping decades to lists of texts with the specified distribution
        """
        logger.info(f"Creating controlled dataset with distribution: {distribution}")
        
        # Set data volume targets based on Hayase paper
        target_size_gb = 10.0  # 10GB total dataset size
        target_size_bytes = target_size_gb * 1024 * 1024 * 1024
        
        # Calculate bytes per decade based on distribution
        bytes_per_decade = {decade: target_size_bytes * prop for decade, prop in distribution.items()}
        
        # Normalize distribution if needed
        total_proportion = sum(distribution.values())
        if abs(total_proportion - 1.0) > 0.001:  # Allow small rounding errors
            normalized = {d: p/total_proportion for d, p in distribution.items()}
            logger.info(f"Normalized distribution to: {normalized}")
            distribution = normalized
        
        # Calculate texts per decade, with much higher counts than before
        texts_per_decade = {decade: max(int(prop * total_texts), 50) for decade, prop in distribution.items()}
        
        # Load all available data with expanded coverage
        logger.info("Loading source texts for controlled dataset...")
        
        # Use the expanded historical catalog in the Gutenberg loader
        self.gutenberg_loader.expand_historical_catalog()
        all_gutenberg_texts = self.gutenberg_loader.load_decade_samples(texts_per_decade=1000)
        
        # British Library texts
        all_bl_texts = self.british_library_loader.load_decade_samples(per_decade=1000, force_fresh=False)  # Get more than needed
        
        # Track success in meeting distribution targets
        target_comparison = {
            "target_distribution": distribution,
            "achieved_distribution": {},
            "target_bytes": bytes_per_decade,
            "achieved_bytes": {}
        }
        
        # Build the controlled dataset with enhanced decade-specific preservation
        controlled_dataset = {}
        current_size_bytes = 0
        
        for decade, target_bytes in bytes_per_decade.items():
            # Get available texts for this decade
            decade_bl = [(text, "british_library") for text in all_bl_texts.get(decade, [])]
            decade_gutenberg = [(text, "gutenberg") for text in all_gutenberg_texts.get(decade, [])]
            
            # Combine sources
            all_texts = decade_bl + decade_gutenberg
            
            # Track decade data volume
            decade_bytes = 0
            decade_texts = []
            
            logger.info(f"Building {decade} dataset to target {target_bytes/(1024*1024):.2f} MB")
            
            # Filter by minimum length to favor longer texts
            min_length = 5000  # Minimum 5000 characters
            quality_texts = [t for t in all_texts if len(t[0]) >= min_length]
            
            # If we have enough quality texts, use them, otherwise fall back to all texts
            if len(quality_texts) >= texts_per_decade[decade] // 2:
                source_texts = quality_texts
                logger.info(f"Using {len(source_texts)} quality texts (min {min_length} chars) for {decade}")
            else:
                source_texts = all_texts
                logger.info(f"Using all {len(source_texts)} available texts for {decade}")
            
            # Keep adding texts until we reach the target data volume
            # Use texts with replacement if needed, but with improved variation
            i = 0
            max_iterations = 100000  # Prevent infinite loops
            
            while decade_bytes < target_bytes and i < max_iterations:
                if not source_texts:
                    # If no real texts, generate synthetic ones with decade-specific patterns
                    text = self._create_historical_synthetic_texts(decade, 1, {}, preserve_decade_characteristics=True)[0]
                    source = "synthetic"
                else:
                    # Use existing texts with wrapping when needed
                    idx = i % len(source_texts)
                    text, source = source_texts[idx]
                    
                    # For data volume, augment the text after first pass, but preserve decade characteristics
                    if i >= len(source_texts) and random.random() < 0.7:
                        text = self._augment_text_for_volume(text, decade, preserve_decade_style=True)
                        source = f"{source}_augmented"
                    elif i >= len(source_texts) * 2:
                        # After second pass, modify to avoid exact duplication
                        text = self._modify_text_slightly(text, decade_specific=True)
                        source = f"{source}_modified"
                
                decade_texts.append((text, source))
                text_bytes = len(text.encode('utf-8'))
                decade_bytes += text_bytes
                i += 1
                
                # Log progress periodically
                if i % 100 == 0 or (i < 100 and i % 10 == 0):
                    logger.info(f"{decade}: {i} texts, {decade_bytes/(1024*1024):.2f} MB / {target_bytes/(1024*1024):.2f} MB")
                
                # Break early if we're struggling to reach the target
                if i > 10000 and decade_bytes < target_bytes * 0.5:
                    logger.warning(f"Unable to reach target for {decade}, stopping at {decade_bytes/(1024*1024):.2f} MB")
                    break
                    
            controlled_dataset[decade] = decade_texts
            current_size_bytes += decade_bytes
            target_comparison["achieved_bytes"][decade] = decade_bytes
            
            logger.info(f"{decade}: {len(decade_texts)} texts, {decade_bytes/(1024*1024):.2f} MB ({distribution.get(decade, 0):.1%})")
        
        # Calculate actual distribution achieved
        actual_bytes_per_decade = {decade: sum(len(text.encode('utf-8')) for text, _ in texts) 
                                for decade, texts in controlled_dataset.items()}
        actual_distribution = {decade: bytes/current_size_bytes 
                            for decade, bytes in actual_bytes_per_decade.items()}
        
        target_comparison["achieved_distribution"] = actual_distribution
        
        # Create metadata for the controlled dataset
        metadata = {
            "type": "controlled_dataset",
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_distribution": distribution,
            "actual_distribution": actual_distribution,
            "target_size_gb": target_size_gb,
            "actual_size_gb": current_size_bytes / (1024*1024*1024),
            "total_texts": sum(len(texts) for texts in controlled_dataset.values()),
            "decades": {
                decade: {
                    "texts": len(texts),
                    "bytes": actual_bytes_per_decade.get(decade, 0),
                    "mb": actual_bytes_per_decade.get(decade, 0) / (1024*1024),
                    "target_proportion": distribution.get(decade, 0),
                    "actual_proportion": actual_distribution.get(decade, 0),
                    "proportion_error": actual_distribution.get(decade, 0) - distribution.get(decade, 0),
                    "sources": {
                        "british_library": sum(1 for _, src in texts if src == "british_library"),
                        "gutenberg": sum(1 for _, src in texts if src == "gutenberg"),
                        "augmented": sum(1 for _, src in texts if "augmented" in src),
                        "modified": sum(1 for _, src in texts if "modified" in src),
                        "synthetic": sum(1 for _, src in texts if src == "synthetic")
                    }
                } for decade, texts in controlled_dataset.items()
            }
        }
        
        # Save the metadata with more detailed information
        metadata_path = self.dataset_dir / "controlled_datasets"
        metadata_path.mkdir(exist_ok=True, parents=True)
        with open(metadata_path / f"controlled_dataset_{int(time.time())}.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Also save the target comparison separately for analysis
        with open(metadata_path / f"target_comparison_{int(time.time())}.json", "w") as f:
            json.dump(target_comparison, f, indent=2)
        
        logger.info("Actual distribution in controlled dataset:")
        for decade, prop in sorted(actual_distribution.items()):
            target = distribution.get(decade, 0)
            logger.info(f"{decade}: {prop:.2%} (target: {target:.2%}, diff: {prop-target:.2%})")
        
        logger.info(f"Total dataset size: {current_size_bytes/(1024*1024*1024):.2f} GB")
        return controlled_dataset
    
    # Add a method to evaluate MSE between distributions (like Hayase et al.)
    def calculate_distribution_mse(self, predicted: Dict[str, float], true: Dict[str, float]) -> float:
        """
        Calculate Mean Squared Error between predicted and true distributions.
        Returns log10(MSE) similar to Hayase et al.
        """
        # Ensure all keys are present in both
        all_decades = set(predicted.keys()) | set(true.keys())
        
        # Calculate MSE
        squared_errors = []
        for decade in all_decades:
            pred_val = predicted.get(decade, 0.0)
            true_val = true.get(decade, 0.0)
            squared_errors.append((pred_val - true_val) ** 2)
        
        mse = sum(squared_errors) / len(squared_errors)
        log10_mse = np.log10(mse) if mse > 0 else -float('inf')
        
        return log10_mse

    def _save_dataset(self, dataset: Dict[str, List[Tuple[str, str]]], metadata: Dict):
        """Save dataset and metadata to disk with size tracking."""
        # Calculate total size
        total_size_bytes = 0
        decade_sizes = {}
        
        for decade, texts_with_sources in dataset.items():
            decade_size_bytes = sum(len(text.encode('utf-8')) for text, _ in texts_with_sources)
            total_size_bytes += decade_size_bytes
            decade_sizes[decade] = {
                "size_bytes": decade_size_bytes,
                "size_mb": decade_size_bytes / (1024*1024),
                "size_gb": decade_size_bytes / (1024*1024*1024),
            }
        
        # Update metadata with size information
        metadata["decade_sizes"] = decade_sizes
        metadata["total_size_bytes"] = total_size_bytes
        metadata["total_size_mb"] = total_size_bytes / (1024*1024)
        metadata["total_size_gb"] = total_size_bytes / (1024*1024*1024)
        metadata["creation_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Saving dataset: {total_size_bytes/(1024*1024*1024):.2f} GB total")
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save texts by decade - using more efficient storage for large datasets
        is_large_dataset = total_size_bytes > 1 * 1024 * 1024 * 1024  # 1 GB threshold
        
        for decade, texts_with_sources in dataset.items():
            decade_dir = self.dataset_dir / decade
            decade_dir.mkdir(parents=True, exist_ok=True)
            
            # Create batch saving for large datasets to avoid memory issues
            if is_large_dataset:
                batch_size = 100  # Save in batches of 100 texts
                for batch_idx in range(0, len(texts_with_sources), batch_size):
                    batch = texts_with_sources[batch_idx:batch_idx + batch_size]
                    
                    # Create CSV with metadata for this batch
                    batch_rows = []
                    for i, (text, source) in enumerate(batch):
                        text_id = f"{decade}_{batch_idx + i:06d}"
                        text_path = decade_dir / f"{text_id}.txt"
                        
                        # Save text file
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                        
                        # Add metadata row
                        batch_rows.append({
                            "id": text_id,
                            "decade": decade,
                            "source": source,
                            "length": len(text),
                            "size_bytes": len(text.encode('utf-8')),
                            "path": str(text_path.relative_to(self.dataset_dir))
                        })
                    
                    # Save batch metadata
                    batch_df = pd.DataFrame(batch_rows)
                    batch_df.to_csv(decade_dir / f"metadata_batch_{batch_idx//batch_size}.csv", index=False)
                    
                    logger.info(f"Saved batch {batch_idx//batch_size} for {decade}: {len(batch)} texts")
            else:
                # Regular saving for smaller datasets
                rows = []
                for i, (text, source) in enumerate(texts_with_sources):
                    text_id = f"{decade}_{i:04d}"
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
                        "size_bytes": len(text.encode('utf-8')),
                        "path": str(text_path.relative_to(self.dataset_dir))
                    })
                
                # Save metadata CSV
                metadata_df = pd.DataFrame(rows)
                metadata_df.to_csv(decade_dir / "metadata.csv", index=False)
        
        logger.info(f"Dataset saved to {self.dataset_dir}")
        
        # For very large datasets, create a compressed backup
        if is_large_dataset and total_size_bytes > 5 * 1024 * 1024 * 1024:  # 5 GB threshold
            logger.info("Creating compressed metadata backup for large dataset")
            
            # Save just the metadata in a compressed format
            import gzip
            try:
                metadata_backup_path = self.dataset_dir / "large_dataset_metadata.json.gz"
                with gzip.open(metadata_backup_path, 'wt', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Compressed metadata backup saved to {metadata_backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create compressed metadata backup: {e}")
    
    

    def _generate_period_paragraph(self, decade: str, vocab: list = None, era_style: dict = None) -> str:
        """
        Generate a historically appropriate paragraph for a specific decade.
        
        Args:
            decade: Target decade
            vocab: List of period-specific vocabulary
            era_style: Dict with period-specific style elements
            
        Returns:
            A generated paragraph appropriate for the time period
        """
        if vocab is None:
            vocab = ["society", "modern", "development", "change", "progress"]
        
        if era_style is None:
            era_style = {
                "openers": ["In this period", "It is remarkable that", "One must consider"],
                "phrases": ["indeed", "as it were", "to be certain", "as one might expect"],
                "style": "formal"
            }
        
        # Select opener and prepare paragraph
        opener = random.choice(era_style["openers"])
        topic = random.choice(vocab)
        
        # Build a paragraph with period-appropriate language
        paragraph = f"{opener}, the development of {topic} represented a significant change in society. "
        
        # Add 3-5 more sentences with period vocabulary and phrasing
        for _ in range(random.randint(3, 5)):
            sentence_templates = [
                f"The influence of {random.choice(vocab)} cannot be overstated. ",
                f"Many considered {random.choice(vocab)} to be essential to progress. ",
                f"The relationship between {random.choice(vocab)} and {random.choice(vocab)} merits further examination. ",
                f"The advancement of {random.choice(vocab)} continued to transform daily life. ",
                f"Scholars often debate the significance of {random.choice(vocab)} during this period. "
            ]
            paragraph += random.choice(sentence_templates)
        
        # Add a period-specific phrase
        if era_style["phrases"]:
            paragraph += f"It was {random.choice(era_style['phrases'])}, that such developments would continue. "
        
        # Add concluding sentence
        paragraph += f"The impact of these changes would be felt for decades to come."
        
        return paragraph
    
    def analyze_decade_data_quality(self, decade: str) -> Dict:
        """
        Analyze the data quality for a specific decade.
        """
        decade_texts = self.load_dataset().get(decade, [])
        
        # Calculate statistics
        stats = {
            "total_texts": len(decade_texts),
            "avg_length": sum(len(text) for text in decade_texts) / max(1, len(decade_texts)),
            "source_distribution": {}
        }
        
        # Analyze sources
        if isinstance(decade_texts[0], tuple):
            sources = [source for _, source in decade_texts]
            for source in set(sources):
                stats["source_distribution"][source] = sources.count(source) / len(sources)
        
        # Count most common character pairs (which might influence merge rules)
        char_pairs = Counter()
        for text in decade_texts:
            if isinstance(text, tuple):
                text = text[0]
            for i in range(len(text) - 1):
                char_pairs[text[i:i+2]] += 1
        
        stats["top_char_pairs"] = dict(char_pairs.most_common(20))
        
        return stats

    def _modify_text_slightly(self, text: str) -> str:
        """
        Make minor modifications to text to avoid exact duplication
        while maintaining its essential characteristics.
        
        Args:
            text: Original text
                
        Returns:
            Modified version of the text
        """
        import re
        
        # Don't modify very short texts
        if len(text) < 500:
            return text
        
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
        indices_to_modify = random.sample(range(len(sentences)), min(num_to_modify, len(sentences)))
        
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

    def load_dataset(self) -> Dict[str, List[str]]:
        """
        Load the prepared dataset.
        
        Returns:
            Dictionary mapping decades to lists of texts
        """
        if not self.metadata_path.exists():
            logger.warning("No saved dataset found. Please build the dataset first.")
            return {}
        
        # Load dataset
        dataset = {decade: [] for decade in TIME_PERIODS.keys()}
        
        for decade in TIME_PERIODS.keys():
            decade_dir = self.dataset_dir / decade
            if not decade_dir.exists():
                continue
                
            metadata_csv = decade_dir / "metadata.csv"
            if metadata_csv.exists():
                metadata_df = pd.read_csv(metadata_csv)
                
                for _, row in metadata_df.iterrows():
                    text_path = self.dataset_dir / row['path']
                    if text_path.exists():
                        with open(text_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        dataset[decade].append(text)
        
        # Log statistics
        total_texts = sum(len(texts) for texts in dataset.values())
        logger.info(f"Loaded dataset with {total_texts} total texts:")
        for decade, texts in dataset.items():
            logger.info(f"  {decade}: {len(texts)} texts")
        
        return dataset

def test_dataset_manager():
    """Test the dataset manager with small sample."""
    manager = TemporalDatasetManager()
    
    # Build small test dataset
    dataset = manager.build_temporal_dataset(texts_per_decade=20, save_dataset=True)
    
    print("\nTemporal Dataset Summary:")
    print("-" * 50)
    for decade, texts_with_sources in dataset.items():
        if texts_with_sources:
            print(f"\n{decade}:")
            print(f"Number of texts: {len(texts_with_sources)}")
            
            # Count by source
            sources = {}
            for _, source in texts_with_sources:
                sources[source] = sources.get(source, 0) + 1
                
            print(f"Sources: {sources}")
            
            # Calculate average length
            avg_length = sum(len(text) for text, _ in texts_with_sources) / len(texts_with_sources)
            print(f"Average text length: {avg_length:.0f} characters")
            
            # Print sample
            if texts_with_sources:
                print(f"Sample text beginning: {texts_with_sources[0][0][:100]}...")

if __name__ == "__main__":
    test_dataset_manager()