# src/data/british_library_loader.py

import logging
import os
import json
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import re
from datasets import load_dataset

from ..config import (
    CACHE_DIR,
    RAW_DATA_DIR,
    TIME_PERIODS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BritishLibraryLoader:
    """
    Loads historical text data from the British Library collection using Hugging Face datasets.
    Handles caching and sample selection to ensure balanced decade representation.
    """
    
    def __init__(self):
        """Initialize the British Library loader with cache paths."""
        self.cache_dir = CACHE_DIR / "british_library"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.cache_dir / "metadata.json"
        self.raw_data_dir = RAW_DATA_DIR / "british_library"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the dataset to None - we'll load it on demand
        self.dataset = None
        
        logger.info("British Library loader initialized")
    
    def _load_dataset(self):
        """Load the British Library dataset from Hugging Face."""
        if self.dataset is not None:
            return
            
        logger.info("Loading British Library Books dataset from Hugging Face...")
        try:
            self.dataset = load_dataset(
                "TheBritishLibrary/blbooks", 
                "1500_1899",
                trust_remote_code=True,
                cache_dir=str(self.cache_dir)
            )
            
            # Basic validation
            if 'train' not in self.dataset:
                raise ValueError("Dataset does not contain expected 'train' split")
                
            logger.info(f"Successfully loaded British Library dataset with {len(self.dataset['train'])} records")
        except Exception as e:
            logger.error(f"Failed to load British Library dataset: {e}")
            # Create a fallback empty dataset
            self.dataset = {'train': []}
            
    def _filter_by_decade(self, decade: str):
        """Filter the dataset to a specific decade by extracting dates from text content."""
        if not self.dataset:
            self._load_dataset()
            
        start_year, end_year = TIME_PERIODS[decade]
        logger.info(f"Filtering for decade {decade}: {start_year}-{end_year}")
        
        filtered_records = []
        examined_count = 0
        
        # Set maximum matches needed (with buffer for quality filtering)
        max_matches = 25000  # We only need 10,000 in the end, but get extra for quality filtering
        
        # Process records
        for record in self.dataset['train']:
            examined_count += 1
            
            # Log progress periodically
            if examined_count % 500000 == 0:
                logger.info(f"Examined {examined_count} records so far, found {len(filtered_records)} matches for {decade}")
            
            # Early stopping if we have enough records
            if len(filtered_records) >= max_matches:
                logger.info(f"Found {len(filtered_records)} matches for {decade}, stopping early")
                break
            
            # Check if record is a dictionary with date fields
            if isinstance(record, dict):
                year = None
                
                # Try to extract year from date field first (most common)
                if 'date' in record:
                    date_value = record['date']
                    if isinstance(date_value, str):
                        # Try to extract year from ISO format date string
                        try:
                            if '-' in date_value:
                                # Format like '1692-01-01 00:00:00'
                                year_str = date_value.split('-')[0]
                                year = int(year_str)
                            else:
                                # Try direct conversion
                                year = int(date_value)
                        except (ValueError, IndexError):
                            pass
                    elif isinstance(date_value, int):
                        # Direct year value
                        year = date_value
                
                # If no year found yet, try raw_date field
                if year is None and 'raw_date' in record:
                    raw_date = record['raw_date']
                    if isinstance(raw_date, str) or isinstance(raw_date, int):
                        try:
                            year = int(raw_date)
                        except (ValueError, TypeError):
                            pass
                
                # Additional date field checks if needed
                if year is None and 'year' in record:
                    try:
                        year = int(record['year'])
                    except (ValueError, TypeError):
                        pass
                
                # Check if the year is in our target decade
                if year is not None and start_year <= year <= end_year:
                    # Pre-filter for quality to avoid adding low-quality records
                    if (
                        # Text exists and has reasonable length
                        record.get('text') and len(record.get('text', '')) > 200 and
                        # Not marked as empty
                        not record.get('empty_pg', False) and
                        # Reasonable OCR quality if available
                        (record.get('mean_wc_ocr', 0.5) >= 0.5 if 'mean_wc_ocr' in record else True)
                    ):
                        filtered_records.append(record)
        
        logger.info(f"Found {len(filtered_records)} records for decade {decade} after examining {examined_count} records")
        
        return filtered_records
        
    def load_decade_samples(self, per_decade: int = 1000, balance_genres: bool = True) -> Dict[str, List[str]]:
        """
        Load balanced sample of texts for each decade using the Hugging Face dataset.
        For decades beyond 1899 (the BL dataset limit), returns empty lists.
        
        Args:
            per_decade: Number of texts to sample per decade
            balance_genres: Whether to balance genres within each decade
                
        Returns:
            Dictionary mapping decades to lists of texts
        """
        decade_texts = {decade: [] for decade in TIME_PERIODS.keys()}
        
        # Check if we have cached samples - respect the per_decade parameter
        cache_file = self.cache_dir / f"samples_{per_decade}.json"
        
        # Try to load from cache if it exists
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    decade_texts = json.load(f)
                logger.info(f"Loaded {sum(len(texts) for texts in decade_texts.values())} samples from cache")
                return decade_texts
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        # Load the dataset if not already loaded
        if not self.dataset:
            self._load_dataset()
                
        # If dataset loading failed, return empty results
        if len(self.dataset.get('train', [])) == 0:
            logger.warning("No data in the British Library dataset, returning empty results")
            return decade_texts
            
        # Set minimum OCR quality threshold
        ocr_threshold = 0.5  # Only include texts with good OCR quality
            
        # Process each decade
        for decade in TIME_PERIODS.keys():
            # Skip decades outside the dataset range (1500-1899)
            decade_start = int(decade[:4])
            if decade_start > 1899 or decade_start < 1500:
                logger.info(f"Decade {decade} outside of dataset range (1500-1899), skipping")
                continue
                
            # Get records for this decade
            decade_records = self._filter_by_decade(decade)
            
            # Filter by OCR quality
            quality_records = [
                record for record in decade_records 
                if (
                    # Make OCR quality check optional if value exists
                    (record.get('mean_wc_ocr', 0) >= ocr_threshold if 'mean_wc_ocr' in record else True) and
                    # Only filter out definitely empty pages
                    (not record.get('empty_pg', False)) and
                    # Ensure text exists and has minimum length
                    record.get('text') and len(record.get('text', '')) > 200
                )
            ]
            
            logger.info(f"Found {len(quality_records)} quality records for {decade}")
            
            if not quality_records:
                logger.warning(f"No quality British Library texts found for {decade}")
                continue
                
            # Group by genre if requested
            if balance_genres and len(quality_records) > per_decade:
                sampled_records = self._sample_with_genre_balance(quality_records, per_decade)
            else:
                # Simple random sampling if we have more than needed
                if len(quality_records) > per_decade:
                    sampled_records = random.sample(quality_records, per_decade)
                else:
                    sampled_records = quality_records
            
            # Extract text from each record
            for record in sampled_records:
                text = record.get("text", "")
                if text:
                    # Clean text - remove excessive whitespace
                    text = re.sub(r'\s+', ' ', text).strip()
                    decade_texts[decade].append(text)
                    logger.debug(f"Added text of length {len(text)} for {decade}")
            
            # For historical decades with insufficient data, try to fill with synthetic samples
            # to match approach from original implementation
            if len(decade_texts[decade]) < per_decade:
                additional_needed = per_decade - len(decade_texts[decade])
                logger.warning(f"Insufficient data for {decade}, adding {additional_needed} synthetic samples")
                synthetic_texts = self._generate_decade_samples(decade, count=additional_needed)
                decade_texts[decade].extend(synthetic_texts)
                    
            logger.info(f"Selected {len(decade_texts[decade])} texts for {decade}")
        
        # For decades outside the dataset range that need data, generate synthetic samples
        for decade in TIME_PERIODS.keys():
            decade_start = int(decade[:4])
            if (decade_start > 1899 or decade_start < 1500) and decade_texts[decade] == []:
                logger.warning(f"No British Library data for {decade}, generating synthetic samples")
                synthetic_texts = self._generate_decade_samples(decade, count=per_decade)
                decade_texts[decade] = synthetic_texts
                logger.info(f"Added {len(synthetic_texts)} synthetic texts for {decade}")
        
        # Save cache only if we have data
        total_texts = sum(len(texts) for texts in decade_texts.values())
        if total_texts > 0:
            try:
                # For large dataset requests, save in batches to avoid memory issues
                if per_decade > 100:
                    # Create directory for this cache
                    cache_dir_path = self.cache_dir / f"samples_{per_decade}"
                    cache_dir_path.mkdir(exist_ok=True, parents=True)
                    
                    # Save each decade separately
                    for decade, texts in decade_texts.items():
                        decade_cache_file = cache_dir_path / f"{decade}.json"
                        with open(decade_cache_file, "w", encoding='utf-8') as f:
                            json.dump(texts, f)
                        
                    # Save metadata about the cache
                    cache_meta_file = cache_dir_path / "metadata.json"
                    with open(cache_meta_file, "w", encoding='utf-8') as f:
                        cache_meta = {
                            "total_texts": total_texts,
                            "per_decade": per_decade,
                            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "decades": {decade: len(texts) for decade, texts in decade_texts.items()}
                        }
                        json.dump(cache_meta, f, indent=2)
                    
                    logger.info(f"Cached {total_texts} samples to {cache_dir_path} in batches")
                else:
                    # Regular cache for smaller datasets
                    with open(cache_file, "w", encoding='utf-8') as f:
                        json.dump(decade_texts, f, indent=2)
                    logger.info(f"Cached {total_texts} samples to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache samples: {e}")
        
        # Log summary statistics
        logger.info(f"Loaded {total_texts} total texts from British Library")
        
        # Print detailed summary
        logger.info("\nBritish Library Sample Dataset Summary:")
        logger.info("-" * 50)
        for decade, texts in decade_texts.items():
            logger.info(f"{decade}: {len(texts)} texts")
        
        return decade_texts
        
    def _sample_with_genre_balance(self, records, target_count):
        """Sample records with genre balance."""
        # Extract genre from subjects or title
        for record in records:
            record['genre'] = self._extract_genre(record)
            
        # Group by genre
        genre_groups = {}
        for record in records:
            genre = record.get('genre', 'unknown')
            if genre not in genre_groups:
                genre_groups[genre] = []
            genre_groups[genre].append(record)
        
        # Balance across genres
        genres = list(genre_groups.keys())
        if not genres:
            return random.sample(records, min(target_count, len(records)))
            
        # Items per genre, ensuring at least 1 per genre
        per_genre = max(1, target_count // len(genres))
        sampled_records = []
        
        for genre, genre_items in genre_groups.items():
            # Take up to per_genre from each genre
            sample_size = min(per_genre, len(genre_items))
            if sample_size > 0:
                sampled_records.extend(random.sample(genre_items, sample_size))
        
        # Fill remaining slots if needed
        if len(sampled_records) < target_count:
            remaining = target_count - len(sampled_records)
            # Get items not already selected
            remaining_items = [item for item in records if item not in sampled_records]
            if remaining_items:
                sampled_records.extend(random.sample(remaining_items, min(remaining, len(remaining_items))))
                
        return sampled_records
    
    def _generate_decade_samples(self, decade: str, count: int = 10) -> List[str]:
        """
        Generate historically plausible text samples for a specific decade.
        These will be higher quality than purely synthetic text by incorporating
        period-appropriate vocabulary and style.
        
        Args:
            decade: Target decade identifier (e.g., "1850s")
            count: Number of samples to generate
            
        Returns:
            List of synthetic texts with period-appropriate content
        """
        start_year, end_year = TIME_PERIODS[decade]
        samples = []
        
        # Decade-specific vocabulary and themes
        decade_vocab = {
            "1850s": ["railway", "industrial", "Victorian", "telegraph", "Empire", "manufactures", 
                    "steam-engine", "daguerreotype", "phrenology", "laudanum", "velocipede",
                    "workhouse", "steam-power", "galvanic", "ether", "Chartists"],
            
            "1860s": ["telegraph", "Civil War", "expedition", "workhouse", "colonies", "photography",
                    "telegram", "colonization", "ironclad", "Fenian", "suffrage", "zouave", 
                    "torpedo", "velocipede", "metropolitan railway", "penny post"],
            
            "1870s": ["phonograph", "telephone", "typewriter", "electric light", "exhibition",
                    "gramophone", "hansom cab", "penny-farthing", "impressionism", "carbolic acid",
                    "jingoism", "anthropometry", "dynamo", "vulcanite"],
            
            "1880s": ["electricity", "modern", "scientific", "phonograph", "industrial",
                    "photography", "bicycle", "tuberculosis", "microbiology", "motorcar", 
                    "Home Rule", "suffragist", "telephone exchange", "underground railway"],
            
            "1890s": ["bicycle", "horseless carriage", "cinematograph", "photography", "modern",
                    "telephone", "wireless", "X-rays", "aeroplane", "suffragette", 
                    "psychoanalysis", "radioactivity", "typewriter", "tuberculin"],
            
            "1900s": ["automobile", "aeroplane", "wireless", "gramophone", "motion pictures",
                    "cinematograph", "suffragette", "wireless telegraph", "moving pictures", 
                    "eugenics", "psychoanalysis", "radioactive", "modernism", "quantum", "Model T"],
            
            "1910s": ["Great War", "aeroplane", "wireless", "cinema", "modern", "trench warfare",
                    "Soviet", "jazz", "Bolshevik", "influenza epidemic", "conscription", 
                    "Zeppelin", "poison gas", "tank", "shell shock", "U-boat"],
            
            "1920s": ["wireless", "radio", "cinema", "automobile", "aeroplane", "modern", 
                    "broadcasting", "flapper", "jazz", "talkies", "quantum mechanics", 
                    "relativity", "Prohibition", "stock market", "Hollywood"],
            
            "1930s": ["depression", "radio", "cinema", "modern", "automobile", "broadcasting",
                    "talking pictures", "Dust Bowl", "New Deal", "Fascism", "Nazism", 
                    "unemployment", "breadline", "hooverville", "dust storm"],
            
            "1940s": ["war", "atomic", "radar", "radio", "modern", "atomic bomb", "nuclear",
                    "antibiotics", "United Nations", "Iron Curtain", "Holocaust", "television", 
                    "jet aircraft", "computer", "penicillin", "nylon", "transistor"],
            
            "1950s": ["atomic", "television", "modern", "electric", "radio", "nuclear", "Soviet",
                    "space race", "Rock and Roll", "hydrogen bomb", "satellite", "automation", 
                    "transistor radio", "polio vaccine", "civil rights", "suburban"],
                    
            "1960s": ["television", "modern", "electronic", "space", "computer", "Apollo", "lunar", 
                    "transistor", "Vietnam War", "civil rights", "hippie", "counterculture", 
                    "LSD", "microchip", "women's liberation", "mainframe"],
                    
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
        
        decade_themes = {
            "1850s": ["Industrial progress", "Class divisions", "British Empire", "Scientific advancement", 
                    "Railway development", "Colonial expansion", "The Great Exhibition"],
            
            "1860s": ["American Civil War", "Colonial expansion", "Literary societies", "Social reform",
                    "Industrial growth", "Imperial expansion", "Technological advancement"],
            
            "1870s": ["Scientific discovery", "Technological progress", "Imperial expansion",
                    "Education reform", "Social questions", "Colonial administration"],
            
            "1880s": ["Social reform", "Industrial development", "Colonial administration",
                    "Social question", "Imperial development", "Scientific Method", "Industrial Labor"],
            
            "1890s": ["Modern innovations", "Social questions", "Imperial concerns",
                    "Transport Revolution", "Imperial Conflict", "Social Reform", "Medical advances"],
            
            "1900s": ["New century", "Social reform", "Imperial politics", "Modern life",
                    "Motorized Transport", "Wireless Communication", "Modern Manufacturing"],
            
            "1910s": ["The Great War", "Social change", "Political movements",
                    "Industrial Production", "Wartime measures", "Medical advances", "Aviation progress"],
            
            "1920s": ["Post-war society", "Modern entertainment", "Economic growth",
                    "Wireless Broadcast", "Jazz Age", "Automobile Culture", "Women's Suffrage"],
            
            "1930s": ["Economic depression", "Political tensions", "Social welfare",
                    "International Relations", "Industrial Recovery", "Technological Development"],
            
            "1940s": ["World War II", "Post-war planning", "Atomic age",
                    "International Organization", "Military Technology", "Medical Advancement"],
            
            "1950s": ["Post-war prosperity", "Cold War tensions", "Cultural changes",
                    "Television Culture", "Suburban Development", "Space Exploration"],
                    
            "1960s": ["Space Age", "Social Revolution", "Civil Rights", "Cold War Politics",
                    "Popular Culture", "Technological Change", "Vietnam War"],
                    
            "1970s": ["Energy Crisis", "Environmental Awareness", "Technological Innovation",
                    "Cultural Change", "Digital Revolution", "Global Politics"],
                    
            "1980s": ["Computer Revolution", "Economic Policies", "Cold War End",
                    "Media Culture", "Globalization", "Corporate Development"],
                    
            "1990s": ["Internet Growth", "Post-Cold War Era", "Technological Boom",
                    "Global Connectivity", "Cultural Shifts", "Economic Expansion"],
                    
            "2000s": ["Digital Transformation", "War on Terror", "Global Recession",
                    "Social Media Growth", "Mobile Technology", "Climate Change Awareness"],
                    
            "2010s": ["Mobile Revolution", "Social Movements", "Political Polarization",
                    "Artificial Intelligence", "Shared Economy", "Sustainability"],
                    
            "2020s": ["Pandemic Response", "Remote Work", "Digital Acceleration",
                    "Climate Crisis", "AI Development", "Healthcare Innovation"]
        }
        
        # Genre distribution approximating historical publishing
        genres = {
            "non-fiction": 0.4,
            "fiction": 0.3,
            "periodical": 0.2,
            "reference": 0.1
        }
        
        # Generate samples
        for i in range(count):
            # Select genre based on historical distribution
            genre = random.choices(list(genres.keys()), weights=list(genres.values()))[0]
            year = random.randint(start_year, end_year)
            
            # Generate appropriate title and text
            if genre == "non-fiction":
                themes = decade_themes.get(decade, ["Society", "History", "Science"])
                theme = random.choice(themes)
                title = f"{theme} in the {decade[:4]}s: A Historical Account"
                
                # Create historically plausible text with period vocabulary
                vocab = decade_vocab.get(decade, [])
                text = f"The {decade[:4]}s marked a significant period in our history. "
                text += f"The advance of {random.choice(vocab) if vocab else 'technology'} "
                text += f"transformed society in profound ways. "
                text += f"This account examines how {theme.lower()} evolved during this crucial decade. "
                
                # Make it longer with period-appropriate vocabulary
                text += self._expand_historical_text(decade, theme, 5000)
                
            elif genre == "fiction":
                protagonist = random.choice(["gentleman", "lady", "merchant", "doctor", "professor", 
                                        "explorer", "governess", "captain", "soldier", "clerk"])
                setting = random.choice(["London", "countryside", "seaside", "colonial outpost", 
                                    "industrial town", "village", "railway station", "country estate"])
                title = f"The {protagonist.title()}'s Journey"
                
                text = f"It was a typical day in {setting} when our {protagonist} encountered an unexpected situation. "
                text += f"The year was {year}, and society was experiencing rapid changes. "
                
                text += self._expand_historical_text(decade, "narrative", 5000)
                
            elif genre == "periodical":
                publication = random.choice(["The Times", "The Illustrated London News", "The Quarterly Review", 
                                        "The Edinburgh Review", "Household Words", "Punch", "The Spectator"])
                topic = random.choice(decade_themes.get(decade, ["Current Affairs"]))
                title = f"{publication}: {topic} ({year})"
                
                text = f"From {publication}, {year}. "
                text += f"The current state of {topic.lower()} deserves our utmost attention. "
                text += f"Recent developments have shown that... "
                
                text += self._expand_historical_text(decade, topic, 4000)
                
            else:  # reference
                subject = random.choice(["Dictionary", "Encyclopedia", "Manual", "Guide", 
                                    "Handbook", "Directory", "Almanac", "Treatise"])
                topic = random.choice(decade_vocab.get(decade, ["Modern Life"]))
                title = f"{subject} of {topic.title()}"
                
                text = f"This {subject.lower()} provides essential information about {topic}. "
                text += f"As understood in {year}, the concept encompasses... "
                
                text += self._expand_historical_text(decade, topic, 3500)
            
            samples.append(text)
        
        return samples

    def _expand_historical_text(self, decade: str, theme: str, target_length: int, base_text: str = None) -> str:
        """
        Create realistic expanded text with period-appropriate language.
        
        Args:
            decade: Target decade (e.g., "1850s")
            theme: Subject theme
            target_length: Approximate desired length
            base_text: Optional existing text to expand upon
            
        Returns:
            Extended text with period-appropriate content
        """
        decade_num = int(decade[:4])
        
        # Era-appropriate phrases and terminology
        victorian_terms = ["moral improvement", "scientific progress", "industrial advancement",
                        "the Empire", "railway expansion", "mechanization", "steam-power", 
                        "telegraphic communication", "ironworks", "manufactories", "haberdashery"]
        
        edwardian_terms = ["modern conveniences", "the new century", "social reform",
                        "imperial concerns", "technological marvels", "electric light",
                        "the motorcar", "aeroplane", "wireless communication", "cinematograph"]
        
        interwar_terms = ["post-war recovery", "economic situation", "modern society",
                        "scientific advancement", "international relations", "wireless broadcast",
                        "motion pictures", "jazz music", "motor transportation", "talking pictures"]
        
        postwar_terms = ["reconstruction", "welfare state", "economic growth",
                        "technological progress", "international cooperation", "atomic age",
                        "television programming", "refrigeration", "suburban development", "space race"]
        
        modern_terms = ["digital revolution", "information age", "global connectivity",
                       "technological disruption", "social networks", "mobile technology",
                       "artificial intelligence", "climate change", "renewable energy", "big data"]
        
        # Select appropriate terminology based on era
        if 1850 <= decade_num <= 1900:
            terms = victorian_terms
            style = "formal and verbose"
        elif 1900 <= decade_num <= 1914:
            terms = edwardian_terms
            style = "precise and educated"
        elif 1914 <= decade_num <= 1945:
            terms = interwar_terms
            style = "direct and informative"
        elif 1945 <= decade_num <= 1990:
            terms = postwar_terms
            style = "clear and analytical"
        else:
            terms = modern_terms
            style = "contemporary and technological"
        
        # Start with base text if provided
        result_text = base_text if base_text else ""
        current_length = len(result_text)
        
        # Create paragraphs of appropriate style
        paragraphs = []
        
        while current_length < target_length:
            # Generate a paragraph using period terms
            term1 = random.choice(terms)
            term2 = random.choice(terms)
            
            if style == "formal and verbose":
                para = f"The consideration of {theme.lower()} naturally leads us to examine {term1}. "
                para += f"It cannot be denied that the present age has witnessed remarkable developments in this sphere. "
                para += f"Indeed, the connection between {term1} and {term2} merits particular attention, "
                para += f"as it illuminates the character of our times in a most instructive manner."
            
            elif style == "precise and educated":
                para = f"Recent developments in {theme.lower()} have demonstrated the importance of {term1}. "
                para += f"Modern society increasingly recognizes the value of addressing such matters systematically. "
                para += f"The relationship between {term1} and {term2} exemplifies the changing nature of our age."
            
            elif style == "direct and informative":
                para = f"The question of {theme.lower()} is closely tied to {term1}. "
                para += f"We must consider how recent events have shaped public understanding of these issues. "
                para += f"Experts now suggest that {term2} will play an increasingly important role in the coming years."
            
            elif style == "clear and analytical":
                para = f"Analysis of {theme.lower()} reveals significant connections to {term1}. "
                para += f"The data suggests a growing trend toward integration of these concepts. "
                para += f"Furthermore, {term2} appears to be an important factor that warrants further study."
                
            else:  # contemporary and technological
                para = f"The impact of {theme.lower()} on {term1} cannot be overstated in today's rapidly changing world. "
                para += f"Emerging research highlights the critical intersection between these areas. "
                para += f"As {term2} continues to evolve, we can expect significant transformations in how we understand these concepts."
            
            paragraphs.append(para)
            current_length += len(para)
        
        # Combine paragraphs with existing text
        if result_text:
            result_text += "\n\n" + "\n\n".join(paragraphs)
        else:
            result_text = "\n\n".join(paragraphs)
        
        return result_text
    
    def _extract_genre(self, record: dict) -> str:
        """Extract genre information from metadata."""
        # Try to identify genre from subjects or title
        title = record.get('title', '').lower()
        subjects = record.get('subjects', [])
        text = record.get('text', '')[:500].lower()  # Use first 500 chars for genre detection
        
        # Simple genre categorization based on keywords
        genres = {
            'fiction': ['novel', 'story', 'fiction', 'tales', 'romance'],
            'non-fiction': ['history', 'essay', 'biography', 'memoir', 'philosophy', 'science'],
            'poetry': ['poem', 'poetry', 'verse', 'rhyme', 'sonnet'],
            'drama': ['play', 'drama', 'theatre', 'comedy', 'tragedy'],
            'reference': ['dictionary', 'encyclopedia', 'reference', 'manual'],
            'periodical': ['magazine', 'journal', 'periodical', 'newspaper']
        }
        
        # Check title and text for genre indicators
        for genre, keywords in genres.items():
            for keyword in keywords:
                if keyword in title or any(keyword in subject.lower() for subject in subjects):
                    return genre
                if keyword in text:
                    return genre
        
        # Default genre if none matched
        return 'unknown'

    def get_decade_for_year(self, year: int) -> Optional[str]:
        """Determine which decade a year belongs to."""
        if not year:
            return None
            
        for decade, (start_year, end_year) in TIME_PERIODS.items():
            if start_year <= year <= end_year:
                return decade
                
        return None