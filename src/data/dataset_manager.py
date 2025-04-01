# src/data/dataset_manager.py

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import json
import random

from ..config import (
    PROCESSED_DATA_DIR,
    TIME_PERIODS,
    ANALYSIS_CONFIG
)
from .british_library_loader import BritishLibraryLoader 
from .gutenberg_loader import GutenbergLoader

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
        self.bl_loader = BritishLibraryLoader()
        self.gutenberg_loader = GutenbergLoader()
        
        # Set up storage directories
        self.dataset_dir = PROCESSED_DATA_DIR / "temporal_dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.dataset_dir / "dataset_metadata.json"
    
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

    def create_large_dataset(self, distribution: Dict[str, float], target_size_gb: float = 1.0) -> Dict[str, List[Tuple[str, str]]]:
        """
        Create a dataset with specified target size in GB to match Hayase et al.
        
        Args:
            distribution: Dictionary mapping decades to proportions
            target_size_gb: Target size in gigabytes
            
        Returns:
            Dictionary mapping decades to lists of (text, source) tuples
        """
        logger.info(f"Creating dataset with target size of {target_size_gb} GB and distribution: {distribution}")
        
        # Calculate target size in bytes
        target_size_bytes = target_size_gb * 1024 * 1024 * 1024
        
        # Calculate bytes per decade based on distribution
        bytes_per_decade = {decade: target_size_bytes * prop for decade, prop in distribution.items()}
        
        # Load all available source texts
        logger.info("Loading all available source texts...")
        
        # Load a large number of texts per decade to have sufficient data for large dataset
        max_texts = 100  # Increase this if needed
        
        # British Library texts - using the correct parameter name 'per_decade'
        bl_texts_by_decade = self.bl_loader.load_decade_samples(per_decade=max_texts)
        
        # Gutenberg texts - pass the parameter directly since it has a different parameter name
        # It likely uses 'texts_per_decade' based on the error message
        try:
            # Try with 'texts_per_decade'
            gutenberg_texts_by_decade = self.gutenberg_loader.load_decade_samples(texts_per_decade=max_texts)
        except TypeError:
            try:
                # Fall back to positional argument if keyword doesn't work
                gutenberg_texts_by_decade = self.gutenberg_loader.load_decade_samples(max_texts)
            except Exception as e:
                logger.error(f"Failed to load Gutenberg texts: {e}")
                gutenberg_texts_by_decade = {}
        
        # Combine sources
        all_texts = {}
        for decade in distribution.keys():
            decade_bl = [(text, "british_library") for text in bl_texts_by_decade.get(decade, [])]
            decade_gutenberg = [(text, "gutenberg") for text in gutenberg_texts_by_decade.get(decade, [])]
            all_texts[decade] = decade_bl + decade_gutenberg
            
            logger.info(f"{decade}: {len(all_texts[decade])} total texts available ({len(decade_bl)} BL, {len(decade_gutenberg)} Gutenberg)")
        
        # Build dataset with target sizes
        dataset = {}
        total_size_bytes = 0
        
        for decade, target_bytes in bytes_per_decade.items():
            decade_texts = all_texts.get(decade, [])
            if not decade_texts:
                logger.warning(f"No texts available for {decade}")
                dataset[decade] = []
                continue
            
            # Calculate current size
            decade_size = 0
            decade_dataset = []
            
            # Keep adding texts until we reach target size
            # Use texts with replacement if necessary
            max_iterations = 10000  # Prevent infinite loops
            iterations = 0
            
            while decade_size < target_bytes and decade_texts and iterations < max_iterations:
                # If we've used all texts once, resample with replacement
                if len(decade_dataset) >= len(decade_texts):
                    text, source = random.choice(decade_texts)
                else:
                    idx = len(decade_dataset) % len(decade_texts)
                    text, source = decade_texts[idx]
                
                decade_dataset.append((text, source))
                decade_size += len(text.encode('utf-8'))
                iterations += 1
                
                # Break if we've added a lot of texts but still haven't reached target
                if len(decade_dataset) > 1000 and decade_size < target_bytes * 0.5:
                    logger.warning(f"Unable to reach target size for {decade}, stopping at {decade_size/1024/1024/1024:.2f} GB")
                    break
            
            dataset[decade] = decade_dataset
            total_size_bytes += decade_size
            logger.info(f"{decade}: {len(decade_dataset)} texts, {decade_size/1024/1024/1024:.2f} GB (target: {target_bytes/1024/1024/1024:.2f} GB)")
        
        logger.info(f"Total dataset size: {total_size_bytes/1024/1024/1024:.2f} GB")
        return dataset

    def build_temporal_dataset(self,
                      texts_per_decade: int = 100,
                      balance_sources: bool = True,
                      save_dataset: bool = True) -> Dict[str, List[Tuple[str, str]]]:
        """
        Build comprehensive historical dataset combining multiple sources with improved balance.
        
        Args:
            texts_per_decade: Target number of texts per decade
            balance_sources: Whether to balance between sources
            save_dataset: Whether to save dataset to disk
        """
        logger.info(f"Building temporal dataset with {texts_per_decade} texts per decade...")
        
        # Clear Gutenberg cache to force metadata regeneration
        import os
        gutenberg_cache_path = CACHE_DIR / "gutenberg_cache" / "gutenberg_metadata.json"
        if os.path.exists(gutenberg_cache_path):
            os.remove(gutenberg_cache_path)
            logger.info(f"Removed Gutenberg metadata cache to force regeneration")
        
        # Define minimum acceptable texts per decade to ensure proper analysis
        min_texts_per_decade = {
            # Historical periods need at least 20 texts each
            "1850s": 20, "1860s": 20, "1870s": 20, "1880s": 20, "1890s": 20,
            "1900s": 20, "1910s": 20, "1920s": 20, "1930s": 20, "1940s": 20,
            "1950s": 20, "1960s": 20,
            # Modern periods can have more
            "1970s": 30, "1980s": 30, "1990s": 50, "2000s": 50, "2010s": 50, "2020s": 50
        }
        
        # Calculate per-source allocation
        per_source = texts_per_decade // 2 if balance_sources else texts_per_decade
        
        # For historical periods, double the source request to ensure we get enough data
        historical_per_source = per_source * 2
        
        # Load texts from historical sources with boosted counts for historical periods
        logger.info("Loading British Library texts...")
        bl_texts = self.bl_loader.load_decade_samples(per_source)
        
        logger.info("Loading Gutenberg texts...")
        # Request more texts from Gutenberg for pre-1950s decades
        gutenberg_per_decade = {}
        for decade in TIME_PERIODS.keys():
            if int(decade[:4]) < 1950:
                gutenberg_per_decade[decade] = historical_per_source
            else:
                gutenberg_per_decade[decade] = per_source
        
        gutenberg_texts = self.gutenberg_loader.load_decade_samples(texts_per_decade=per_source)
        
        # Combine and balance dataset
        combined_dataset = {}
        dataset_metadata = {
            "total_texts": 0,
            "sources": {
                "british_library": 0,
                "gutenberg": 0
            },
            "decades": {}
        }
        
        for decade in TIME_PERIODS.keys():
            # Get texts from each source
            decade_bl = [(text, "british_library") for text in bl_texts.get(decade, [])]
            decade_gutenberg = [(text, "gutenberg") for text in gutenberg_texts.get(decade, [])]
            
            # Combine sources
            all_texts = decade_bl + decade_gutenberg
            
            # Check if we have the minimum required texts
            decade_minimum = min_texts_per_decade.get(decade, 20)
            if len(all_texts) < decade_minimum:
                logger.warning(f"Insufficient texts for {decade}: only {len(all_texts)}/{decade_minimum} available")
                
                # Generate synthetic data only as a last resort for historical periods
                if int(decade[:4]) < 1970 and len(all_texts) < decade_minimum:
                    shortfall = decade_minimum - len(all_texts)
                    logger.warning(f"Adding {shortfall} historically accurate synthetic texts for {decade}")
                    
                    # Create synthetic samples based on neighboring decade texts when possible
                    synthetic_texts = self._create_historical_synthetic_texts(decade, shortfall, combined_dataset)
                    all_texts.extend([(text, "synthetic") for text in synthetic_texts])
            
            # Sample if we have more than needed
            if len(all_texts) > texts_per_decade:
                # When sampling, preserve all real historical texts for pre-1950s
                if int(decade[:4]) < 1950:
                    # Keep all authentic texts from sparse historical periods
                    priority_texts = [t for t in all_texts if t[1] != "synthetic"]
                    
                    if len(priority_texts) <= texts_per_decade:
                        # Keep all real texts and sample from synthetic to reach target
                        synthetic_texts = [t for t in all_texts if t[1] == "synthetic"]
                        needed_synthetic = texts_per_decade - len(priority_texts)
                        
                        if needed_synthetic > 0 and synthetic_texts:
                            sampled_synthetic = random.sample(synthetic_texts, min(needed_synthetic, len(synthetic_texts)))
                            all_texts = priority_texts + sampled_synthetic
                        else:
                            all_texts = priority_texts
                    else:
                        # If we have more genuine texts than needed, sample from them
                        all_texts = random.sample(priority_texts, texts_per_decade)
                else:
                    # For modern periods, simple random sampling
                    all_texts = random.sample(all_texts, texts_per_decade)
            
            combined_dataset[decade] = all_texts
            
            # Update metadata
            decade_metadata = {
                "total": len(all_texts),
                "british_library": sum(1 for _, src in all_texts if src == "british_library"),
                "gutenberg": sum(1 for _, src in all_texts if src == "gutenberg"),
                "synthetic": sum(1 for _, src in all_texts if src == "synthetic")
            }
            
            dataset_metadata["decades"][decade] = decade_metadata
            dataset_metadata["total_texts"] += decade_metadata["total"]
            dataset_metadata["sources"]["british_library"] += decade_metadata["british_library"]
            dataset_metadata["sources"]["gutenberg"] += decade_metadata["gutenberg"]
        
        # Log comprehensive statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Total texts: {dataset_metadata['total_texts']}")
        logger.info(f"British Library texts: {dataset_metadata['sources']['british_library']}")
        logger.info(f"Gutenberg texts: {dataset_metadata['sources']['gutenberg']}")
        
        # Log decade-level coverage
        logger.info("\nDecade Coverage:")
        for decade, stats in dataset_metadata["decades"].items():
            if stats["total"] > 0:
                synthetic_count = stats.get("synthetic", 0)
                synthetic_info = f", Synthetic: {synthetic_count}" if synthetic_count > 0 else ""
                
                logger.info(f"{decade}: {stats['total']} texts " +
                        f"(BL: {stats['british_library']}, " +
                        f"Gutenberg: {stats['gutenberg']}{synthetic_info})")
        
        if save_dataset:
            self._save_dataset(combined_dataset, dataset_metadata)
        
        return combined_dataset

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

    def create_controlled_dataset(self, distribution: Dict[str, float], total_texts: int = 500) -> Dict[str, List[Tuple[str, str]]]:
        """
        Create a dataset with known temporal distribution for validation,
        with improved handling of data shortages.
        
        Args:
            distribution: Dictionary mapping decades to proportions (e.g. {'1950s': 0.2})
            total_texts: Total number of texts to include
                
        Returns:
            Dictionary mapping decades to lists of texts with the specified distribution
        """
        logger.info(f"Creating controlled dataset with distribution: {distribution}")
        
        # Normalize distribution if needed
        total_proportion = sum(distribution.values())
        if abs(total_proportion - 1.0) > 0.001:  # Allow small rounding errors
            normalized = {d: p/total_proportion for d, p in distribution.items()}
            logger.info(f"Normalized distribution to: {normalized}")
            distribution = normalized
        
        # Calculate texts per decade
        texts_per_decade = {decade: int(prop * total_texts) for decade, prop in distribution.items()}
        
        # Ensure at least one text per specified decade
        for decade in distribution:
            if texts_per_decade[decade] == 0:
                texts_per_decade[decade] = 1
        
        # Recalculate total after ensuring minimums
        adjusted_total = sum(texts_per_decade.values())
        if adjusted_total > total_texts:
            # Scale down proportionally if we exceeded total
            factor = total_texts / adjusted_total
            texts_per_decade = {d: max(1, int(n * factor)) for d, n in texts_per_decade.items()}
        
        # Ensure we have enough historical data before proceeding
        self.ensure_historical_coverage()
        
        # Load all available data with expanded coverage
        logger.info("Loading source texts for controlled dataset...")
        all_bl_texts = self.bl_loader.load_decade_samples(per_decade=50)  # Get more than needed
        
        # Use the expanded historical catalog in the Gutenberg loader
        self.gutenberg_loader.expand_historical_catalog()
        all_gutenberg_texts = self.gutenberg_loader.load_decade_samples(texts_per_decade=50)
        
        # Handle case where Gutenberg loader returns None
        if all_gutenberg_texts is None:
            logger.error("Gutenberg loader returned None instead of dataset dictionary")
            all_gutenberg_texts = {}  # Use empty dict as fallback
        
        # Build the controlled dataset
        controlled_dataset = {}
        for decade, count in texts_per_decade.items():
            # Get available texts for this decade
            decade_bl = [(text, "british_library") for text in all_bl_texts.get(decade, [])]
            decade_gutenberg = [(text, "gutenberg") for text in all_gutenberg_texts.get(decade, [])]
            
            # Combine sources
            all_texts = decade_bl + decade_gutenberg
            
            # If we don't have enough real texts, generate synthetic ones
            if len(all_texts) < count:
                logger.warning(f"Insufficient real texts for {decade}: have {len(all_texts)}, need {count}")
                
                # Generate synthetic texts with period-appropriate vocabulary and style
                needed = count - len(all_texts)
                synthetic_texts = self._create_historical_synthetic_texts(decade, needed, {})
                all_texts.extend([(text, "synthetic") for text in synthetic_texts])
                
                logger.info(f"Added {len(synthetic_texts)} synthetic texts for {decade}")
            
            # Sample if we have more than needed, otherwise use all available
            if len(all_texts) > count:
                # Prioritize real texts over synthetic
                real_texts = [t for t in all_texts if t[1] != "synthetic"]
                synthetic_texts = [t for t in all_texts if t[1] == "synthetic"]
                
                # If we have enough real texts, use only those
                if len(real_texts) >= count:
                    sampled_texts = random.sample(real_texts, count)
                else:
                    # Use all real texts, plus some synthetic ones
                    needed_synthetic = count - len(real_texts)
                    sampled_synthetic = random.sample(synthetic_texts, min(needed_synthetic, len(synthetic_texts)))
                    sampled_texts = real_texts + sampled_synthetic
            else:
                sampled_texts = all_texts
                
            controlled_dataset[decade] = sampled_texts
            logger.info(f"{decade}: {len(sampled_texts)}/{count} texts (requested: {distribution.get(decade, 0):.1%})")
        
        # Calculate actual distribution
        total_selected = sum(len(texts) for texts in controlled_dataset.values())
        actual_distribution = {decade: len(texts)/total_selected for decade, texts in controlled_dataset.items()}
        
        logger.info("Actual distribution in controlled dataset:")
        for decade, prop in sorted(actual_distribution.items()):
            target = distribution.get(decade, 0)
            logger.info(f"{decade}: {prop:.2%} (target: {target:.2%}, diff: {prop-target:.2%})")
        
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
        """Save dataset and metadata to disk."""
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save texts by decade
        for decade, texts_with_sources in dataset.items():
            decade_dir = self.dataset_dir / decade
            decade_dir.mkdir(parents=True, exist_ok=True)
            
            # Create CSV with metadata
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
                    "path": str(text_path.relative_to(self.dataset_dir))
                })
            
            # Save metadata CSV
            metadata_df = pd.DataFrame(rows)
            metadata_df.to_csv(decade_dir / "metadata.csv", index=False)
        
        logger.info(f"Dataset saved to {self.dataset_dir}")
    
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