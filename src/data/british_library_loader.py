# src/data/british_library_loader.py

import logging
import os
import json
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import re
from datasets import load_dataset, DownloadConfig
import shutil
from tqdm import tqdm

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
        """Load the British Library dataset from Hugging Face with better error handling and timeouts."""
        if self.dataset is not None:
            return
                
        logger.info("Loading British Library Books dataset from Hugging Face...")
        try:
            # Load each configuration separately with explicit timeouts and retries
            datasets = {}
            # Use the CORRECT configurations that actually exist in the dataset
            for config in ['1510_1699', '1700_1799', '1800_1899', '1500_1899']:
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        logger.info(f"Loading BL configuration {config} (attempt {retry+1}/{max_retries})")
                        
                        # Set explicit timeout and download chunk size
                        ds = load_dataset(
                            "TheBritishLibrary/blbooks", 
                            config,
                            trust_remote_code=True,
                            cache_dir=str(self.cache_dir),
                            download_config=DownloadConfig(
                                max_retries=5,
                                force_download=False,
                                cache_dir=str(self.cache_dir),
                            )
                        )
                        
                        if 'train' in ds:
                            # Take a larger subset to get more data
                            subset_size = min(10000, len(ds['train']))
                            if subset_size > 0:
                                # Sample from the dataset
                                indices = list(range(len(ds['train'])))
                                random.shuffle(indices)
                                subset_indices = indices[:subset_size]
                                datasets[config] = ds['train'].select(subset_indices)
                                logger.info(f"Successfully loaded {subset_size} samples from {config}")
                                break  # Success, exit retry loop
                        else:
                            logger.warning(f"No 'train' split in {config}")
                            
                    except Exception as e:
                        logger.warning(f"Attempt {retry+1}/{max_retries} for {config} failed: {e}")
                        time.sleep(10)  # Wait before retrying
            
            # If we got any successful loads, combine them
            if datasets:
                try:
                    from datasets import concatenate_datasets
                    combined = concatenate_datasets(list(datasets.values()))
                    self.dataset = {'train': combined}
                    logger.info(f"Combined dataset has {len(combined)} records")
                except Exception as e:
                    logger.error(f"Failed to combine datasets: {e}")
                    # Use whatever we got successfully
                    if datasets:
                        first_key = list(datasets.keys())[0]
                        self.dataset = {'train': datasets[first_key]}
                        logger.info(f"Using only dataset {first_key} with {len(datasets[first_key])} records")
                    else:
                        logger.warning("No datasets available after attempted loads")
                        self.dataset = {'train': []}
            else:
                logger.warning("No BL configurations could be loaded, falling back to empty dataset")
                self.dataset = {'train': []}
                
        except Exception as e:
            logger.error(f"Failed to load British Library dataset: {e}")
            # Create a fallback empty dataset
            self.dataset = {'train': []}

    def diagnose_dataset_structure(self):
        """One-off function to diagnose dataset structure for debugging purposes"""
        if not self.dataset or 'train' not in self.dataset:
            self._load_dataset()
            if not self.dataset or 'train' not in self.dataset:
                logger.error("Could not load dataset for diagnosis")
                return
        
        # Sample some records
        sample_size = min(20, len(self.dataset['train']))
        samples = [self.dataset['train'][i] for i in range(sample_size)]
        
        # Collect all field names across samples
        all_fields = set()
        for sample in samples:
            if isinstance(sample, dict):
                all_fields.update(sample.keys())
        
        logger.info(f"Found {len(all_fields)} fields across {sample_size} sample records: {sorted(all_fields)}")
        
        # Check for date information in different fields
        date_patterns = [
            (r'\b(1[5-9]\d\d|20[0-2]\d)\b', "Exact year (1500-2029)"),
            (r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', "Date with separators (DD-MM-YYYY)"),
            (r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', "Date with separators (YYYY-MM-DD)"),
            (r'(1[5-9]|20[0-2])\d{2}', "Four-digit year"),
            (r'(1[5-9]|20[0-2])\d{2}s', "Decade reference (1800s)"),
            (r'(\d{1,2})(?:st|nd|rd|th) [Cc]entury', "Century reference")
        ]
        
        potential_date_fields = {}
        for field in sorted(all_fields):
            date_matches = 0
            for sample in samples:
                if isinstance(sample, dict) and field in sample and sample[field]:
                    value = str(sample[field])
                    for pattern, desc in date_patterns:
                        if re.search(pattern, value):
                            date_matches += 1
                            break
            
            if date_matches > 0:
                match_percent = (date_matches / sample_size) * 100
                potential_date_fields[field] = f"{match_percent:.1f}% of values match date patterns"
        
        logger.info(f"Potential date fields: {potential_date_fields}")
        
        # Sample values from promising fields
        for field, stats in potential_date_fields.items():
            logger.info(f"Sample values for '{field}' ({stats}):")
            for i, sample in enumerate(samples[:5]):
                if isinstance(sample, dict) and field in sample:
                    logger.info(f"  Sample {i+1}: {sample[field]}")

    def load_british_library_historical_data(self, per_decade=1000):
        """
        Load historical texts directly from year-specific subsets of the BL dataset.
        This provides a more direct approach to accessing historical content.
        """       
        try:
            logger.info(f"Cache directory: {self.cache_dir}")
            logger.info(f"Available disk space: {shutil.disk_usage(self.cache_dir).free / (1024**3):.2f} GB")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        # Initialize decade texts container
        decade_texts = {decade: [] for decade in TIME_PERIODS.keys()}

        if self.dataset and 'train' in self.dataset:
            self.diagnose_dataset_structure()  # Call the diagnostic method
        
        # Load dataset if not already loaded
        if not self.dataset:
            self._load_dataset()
        
        if not self.dataset or 'train' not in self.dataset or len(self.dataset['train']) == 0:
            logger.warning("No British Library dataset available")
            return decade_texts
        
        total_records = len(self.dataset['train'])
        logger.info(f"Processing {total_records} British Library records")
        
        # Print sample record for debugging
        if total_records > 0:
            sample_record = self.dataset['train'][0]
            logger.info(f"Sample record structure: {list(sample_record.keys())}")
            for field in ['date', 'title', 'text']:
                if field in sample_record:
                    sample_value = sample_record[field]
                    if isinstance(sample_value, str) and len(sample_value) > 100:
                        sample_value = sample_value[:100] + "..."
                    logger.info(f"Sample {field}: {sample_value}")
        
        # Counter for stats
        processed = 0
        year_found = 0
        assigned = 0
        
        # Process records with progress bar
        for record in tqdm(self.dataset['train'], desc="Processing BL records", total=total_records):
            processed += 1
            
            # Skip if not a dictionary
            if not isinstance(record, dict):
                continue
            
            # Initialize with no year found
            year = None
            
            # COMPREHENSIVE YEAR EXTRACTION STRATEGY
            
            # 1. Check for known date fields with various formats
            date_fields = ['date', 'pubDate', 'publicationDate', 'year', 'created']
            for field in date_fields:
                if field in record and record[field]:
                    date_str = str(record[field])
                    
                    # Try exact year pattern (1500-2029)
                    year_match = re.search(r'\b(1[5-9]\d\d|20[0-2]\d)\b', date_str)
                    if year_match:
                        try:
                            year = int(year_match.group(1))
                            break
                        except ValueError:
                            pass
            
            # 2. If no year yet, search in publication info or title
            if not year:
                search_fields = ['title', 'publisher', 'description', 'creator']
                for field in search_fields:
                    if field in record and record[field]:
                        content = str(record[field])
                        
                        # Look for years in text
                        year_match = re.search(r'\b(1[5-9]\d\d|20[0-2]\d)\b', content)
                        if year_match:
                            try:
                                year = int(year_match.group(1))
                                break
                            except ValueError:
                                pass
            
            # 3. Check for century indicators
            if not year:
                century_patterns = {
                    '16th century': 1550, '17th century': 1650,
                    '18th century': 1750, '19th century': 1850,
                    '20th century': 1950, 'nineteenth century': 1850,
                    'eighteenth century': 1750, 'seventeenth century': 1650
                }
                
                for field in ['title', 'date', 'description']:
                    if field in record and record[field]:
                        content = str(record[field]).lower()
                        for century_text, century_year in century_patterns.items():
                            if century_text in content:
                                year = century_year
                                break
                        if year:
                            break
            
            # 4. If still no year, check if config name provides a clue
            if not year and hasattr(record, '_index') and isinstance(record._index, str):
                config = record._index
                if config == '1510_1699':
                    year = 1650  # Midpoint
                elif config == '1700_1799':
                    year = 1750
                elif config == '1800_1899':
                    year = 1850
            
            # Count if we found a year
            if year:
                year_found += 1
                
                # Determine decade and extract text
                for decade, (start_year, end_year) in TIME_PERIODS.items():
                    if start_year <= year <= end_year:
                        if 'text' in record and record['text'] and len(record['text']) > 500:
                            decade_texts[decade].append(record['text'])
                            assigned += 1
                            
                            # If we have enough for this decade, move on
                            if len(decade_texts[decade]) >= per_decade:
                                break
                        break
            
            # Log progress periodically
            if processed % 5000 == 0:
                logger.info(f"Processed {processed}/{total_records} records, found year for {year_found}, assigned {assigned} texts")
        
        # Final stats
        logger.info(f"BL dataset processing complete: processed {processed} records, found year for {year_found}, assigned {assigned} texts")
        
        for decade, texts in decade_texts.items():
            logger.info(f"Found {len(texts)} British Library texts for {decade}")
        
        return decade_texts

    def create_decade_indexed_dataset(self):
        """Preprocess the entire British Library dataset and organize by decade."""
        import re
        from ..config import TIME_PERIODS
        
        if self.dataset is None:
            self._load_dataset()
                
        decade_indices = {decade: [] for decade in TIME_PERIODS.keys()}
        
        logger.info("Creating decade-indexed dataset (this will take time but save future runs)...")
        
        # Process in batches to avoid memory issues
        batch_size = 100000
        total_records = len(self.dataset['train'])
        
        for i in range(0, total_records, batch_size):
            batch = self.dataset['train'][i:min(i+batch_size, total_records)]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_records+batch_size-1)//batch_size}")
            
            for idx, record in enumerate(batch):
                record_idx = i + idx
                year = None
                
                # Skip if record is not a dictionary
                if not isinstance(record, dict):
                    continue
                    
                # Extract year from date field - IMPROVED PATTERN MATCHING
                if 'date' in record:
                    date_value = record['date']
                    if isinstance(date_value, str):
                        # Try multiple date formats
                        # 1. Simple year (1850)
                        year_match = re.search(r'\b(1[5-9]\d\d|20[0-2]\d)\b', date_value)
                        if year_match:
                            try:
                                year = int(year_match.group(1))
                            except ValueError:
                                pass
                        
                        # 2. Date ranges (1850-1899)
                        if not year:
                            range_match = re.search(r'\b(1[5-9]\d\d|20[0-2]\d)\s*[-–—]\s*(1[5-9]\d\d|20[0-2]\d)', date_value)
                            if range_match:
                                try:
                                    # Use start year of range
                                    year = int(range_match.group(1))
                                except ValueError:
                                    pass
                
                # Check title for year information as fallback
                if not year and 'title' in record and isinstance(record['title'], str):
                    title = record['title']
                    # Look for years in titles (common in historical records)
                    year_patterns = [
                        r'\((\d{4})\)',            # Year in parentheses: "Title (1850)"
                        r', (\d{4})',              # Year after comma: "Title, 1850"
                        r'(\d{4})-(\d{4})',        # Year range: "1850-1900"
                        r'published in (\d{4})',   # Explicit publication year
                        r'written in (\d{4})',     # Year written
                        r'\[(\d{4})\]',            # Year in brackets
                        r'first published (\d{4})' # First publication reference
                    ]
                    
                    for pattern in year_patterns:
                        match = re.search(pattern, title)
                        if match:
                            try:
                                potential_year = int(match.group(1))
                                if 1500 <= potential_year <= 2023:  # Reasonable range
                                    year = potential_year
                                    break
                            except ValueError:
                                pass
                
                # Check dataset-specific metadata fields
                if not year and hasattr(record, 'metadata'):
                    metadata = record.metadata
                    if hasattr(metadata, 'year'):
                        try:
                            year = int(metadata.year)
                        except (ValueError, TypeError):
                            pass
                
                # Find which decade this belongs to
                if year:
                    for decade, (start_year, end_year) in TIME_PERIODS.items():
                        if start_year <= year <= end_year:
                            decade_indices[decade].append(record_idx)
                            break
            
            # Periodically log progress and counts
            if (i // batch_size) % 10 == 0:
                logger.info("Current decade counts:")
                for decade, indices in decade_indices.items():
                    logger.info(f"  {decade}: {len(indices)} records so far")
        
        # Save indices to cache
        indices_path = self.cache_dir / "decade_indices.json"
        with open(indices_path, 'w') as f:
            json.dump(decade_indices, f)
        
        logger.info(f"Created decade index with records per decade:")
        for decade, indices in decade_indices.items():
            logger.info(f"  {decade}: {len(indices)} records")
        
        return decade_indices

    def load_decade_samples(self, per_decade: int = 1000, balance_genres: bool = True, force_fresh: bool = False) -> Dict[str, List[str]]:
        """
        Load balanced sample of texts for each decade using the Hugging Face dataset.
        Enhanced with better error handling and retry logic.
        """
        # Check if we have cached samples - respect the per_decade parameter
        cache_file = self.cache_dir / f"samples_{per_decade}.json"
        
        # Try to load from cache if it exists and we're not forcing fresh processing
        if cache_file.exists() and not force_fresh:
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
        if not self.dataset or len(self.dataset.get('train', [])) == 0:
            logger.warning("No data in the British Library dataset, returning empty results")
            return {decade: [] for decade in TIME_PERIODS.keys()}
        
        # Check if we have a cached index
        indices_path = self.cache_dir / "decade_indices.json"
        decade_indices = {}
        
        # Try to load decade indices from cache first
        if indices_path.exists():
            try:
                logger.info("Loading decade indices from cache")
                with open(indices_path, 'r') as f:
                    decade_indices = json.load(f)
                    # Convert index strings back to integers
                    decade_indices = {k: [int(idx) for idx in v] for k, v in decade_indices.items()}
            except Exception as e:
                logger.warning(f"Failed to load decade indices, creating new ones: {e}")
                decade_indices = {}
        
        # If indices not loaded, create them
        if not decade_indices:
            decade_indices = self.create_decade_indexed_dataset()
        
        # Create result dictionary
        decade_texts = {}
        
        # Process each decade with error handling and retries
        for decade, indices in decade_indices.items():
            decade_texts[decade] = []
            
            if not indices:
                continue
                
            # Sample indices with retry mechanism
            sample_size = min(per_decade, len(indices))
            sampled_indices = random.sample(indices, sample_size)
            
            # Load texts in batches to avoid memory issues
            batch_size = 50
            total_loaded = 0
            
            for start_idx in range(0, len(sampled_indices), batch_size):
                batch_indices = sampled_indices[start_idx:start_idx + batch_size]
                
                # Use retry mechanism for batch loading
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        batch_texts = []
                        for idx in batch_indices:
                            try:
                                record = self.dataset['train'][int(idx)]
                                if 'text' in record and isinstance(record['text'], str) and len(record['text']) > 200:
                                    batch_texts.append(record['text'])
                            except Exception as e:
                                logger.debug(f"Error loading text {idx}: {e}")
                        
                        decade_texts[decade].extend(batch_texts)
                        total_loaded += len(batch_texts)
                        break  # Exit retry loop if successful
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Error in batch {start_idx//batch_size}, attempt {attempt+1}: {e}")
                            time.sleep(1)  # Wait before retry
                        else:
                            logger.error(f"Failed to load batch after {max_retries} attempts: {e}")
            
            logger.info(f"Loaded {total_loaded} texts for {decade}")
        
        # Save to cache for future use
        try:
            with open(cache_file, "w", encoding='utf-8') as f:
                json.dump(decade_texts, f)
            logger.info(f"Cached {sum(len(texts) for texts in decade_texts.values())} samples to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache samples: {e}")
        
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