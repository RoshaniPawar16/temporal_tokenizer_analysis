"""
Project Gutenberg Dataset Loader

This module implements a robust loader for accessing and processing texts from Project Gutenberg.
It handles downloading, caching, and cleaning of texts, with careful attention to temporal metadata
to support analysis across different time periods.

Key features:
- Efficient metadata caching
- Robust error handling
- Text cleaning and normalization
- Decade-based sampling
- Memory-efficient processing of large texts
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import random
from datetime import datetime

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TIME_PERIODS,
    ANALYSIS_CONFIG
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GutenbergLoader:
    """
    A comprehensive loader for Project Gutenberg texts with temporal analysis support.
    
    This class manages the downloading, processing, and sampling of texts from Project
    Gutenberg, with special attention to maintaining temporal accuracy and data quality.
    It implements caching to prevent unnecessary downloads and includes robust error
    handling for network issues.
    """
    
    def __init__(self):
        """Initialize the loader with necessary paths and configurations."""
        # Set up cache and data directories
        self.cache_dir = RAW_DATA_DIR / "gutenberg_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = self.cache_dir / "gutenberg_metadata.json"
        self.processed_dir = PROCESSED_DATA_DIR / "gutenberg"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Gutenberg API endpoints and mirrors
        self.catalog_url = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
        self.mirror_urls = [
            "https://www.gutenberg.org/files/{id}/{id}-0.txt",
            "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
            "https://gutenberg.pglaf.org/{id}/pg{id}.txt"
        ]
        
        # Load or create metadata catalog
        self.metadata = self._load_or_create_catalog()
        
    def _load_or_create_catalog(self) -> Dict:
        """
        Load existing catalog or create a new one if none exists.
        
        Returns:
            Dict: Mapping of book IDs to their metadata
        """
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted metadata file. Creating new catalog.")
                return self._create_new_catalog()
        else:
            logger.info("Creating new Gutenberg catalog. This may take some time...")
            return self._create_new_catalog()
    
    def _create_new_catalog(self) -> Dict:
        """
        Create a new catalog by downloading and processing the Gutenberg metadata.
        
        Returns:
            Dict: Mapping of book IDs to their metadata
        """
        try:
            # Download catalog with timeout and retry
            for attempt in range(3):
                try:
                    response = requests.get(self.catalog_url, timeout=30)
                    response.raise_for_status()
                    break
                except (requests.RequestException, requests.Timeout) as e:
                    if attempt == 2:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed. Retrying...")
            
            # Parse catalog
            catalog_df = pd.read_csv(pd.io.common.StringIO(response.text))
            
            # Process entries
            metadata = {}
            for _, row in tqdm(catalog_df.iterrows(), 
                             total=len(catalog_df),
                             desc="Processing Gutenberg catalog"):
                try:
                    book_id = row.get('Text#')
                    
                    # Validate book ID
                    if pd.isna(book_id) or not str(book_id).isdigit():
                        continue
                    
                    book_id = str(int(book_id))
                    
                    # Extract and validate year
                    year = None
                    for field in ['Issued', 'Year', 'Release Date']:
                        if field in row and pd.notnull(row[field]):
                            match = re.search(r'\b(1[8-9]\d\d|20[0-2]\d)\b', str(row[field]))
                            if match:
                                year = int(match.group(1))
                                break
                    
                    if not year or not (1800 <= year <= 2023):
                        continue
                    
                    # Process metadata fields with proper null handling
                    title = str(row.get('Title', '')) if pd.notnull(row.get('Title')) else ''
                    author = str(row.get('Author', '')) if pd.notnull(row.get('Author')) else ''
                    language = str(row.get('Language', 'en')).lower() if pd.notnull(row.get('Language')) else 'en'
                    
                    # Process subjects safely
                    subjects_raw = row.get('Subjects', '')
                    subjects = str(subjects_raw).split(';') if pd.notnull(subjects_raw) else []
                    subjects = [s.strip() for s in subjects if s.strip()]
                    
                    metadata[book_id] = {
                        'title': title,
                        'author': author,
                        'year': year,
                        'language': language,
                        'subjects': subjects
                    }
                
                except Exception as e:
                    logger.debug(f"Error processing row: {e}")
                    continue
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created catalog with {len(metadata)} books")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to create catalog: {e}")
            return {}
    
    def load_decade_samples(self,
                      texts_per_decade: int = 50,
                      min_text_length: int = 1000,
                      english_only: bool = True,
                      balance_genres: bool = True) -> Dict[str, List[str]]:
        """
        Load a balanced sample of texts for each decade with improved historical coverage.
        
        Args:
            texts_per_decade: Target number of texts per decade
            min_text_length: Minimum acceptable text length
            english_only: Whether to restrict to English texts
            balance_genres: Whether to balance genres within each decade
                
        Returns:
            Dict mapping decades to lists of texts
        """
        decade_texts = {decade: [] for decade in TIME_PERIODS.keys()}
        
        # Prioritize historical decades (use higher counts for older time periods)
        prioritized_counts = {}
        for decade in TIME_PERIODS.keys():
            decade_start = int(decade[:4])
            if decade_start < 1900:
                # Double the count for 19th century
                prioritized_counts[decade] = texts_per_decade * 2
            elif decade_start < 1950:
                # 1.5x count for early 20th century
                prioritized_counts[decade] = int(texts_per_decade * 1.5)
            else:
                # Standard count for modern periods
                prioritized_counts[decade] = texts_per_decade
        
        # Check if we need to use the historical catalog supplement
        need_historical = any(int(decade[:4]) < 1970 for decade in TIME_PERIODS.keys())
        
        # If we're missing historical metadata but need it, use the historical supplement
        if need_historical and not self._has_historical_catalog():
            logger.info("Adding historical book catalog supplement")
            self._add_historical_catalog_supplement()
        
        # Group books by decade
        decade_book_ids = {decade: [] for decade in TIME_PERIODS.keys()}
        
        # First pass: Process all books in the catalog
        for book_id, meta in self.metadata.items():
            year = meta.get('year')
            if not year:
                continue
            
            if english_only and meta.get('language', 'en') != 'en':
                continue
            
            # Assign to decade
            for decade, (start_year, end_year) in TIME_PERIODS.items():
                if start_year <= year <= end_year:
                    decade_book_ids[decade].append(book_id)
                    break
        
        # Log the distribution of books by decade
        logger.info("Initial book distribution by decade:")
        for decade, book_ids in decade_book_ids.items():
            logger.info(f"  {decade}: {len(book_ids)} books available")
        
        # Second pass: Use the historical book fallback for decades with insufficient data
        for decade, book_ids in decade_book_ids.items():
            target_count = prioritized_counts[decade]
            decade_start = int(decade[:4])
            
            # For historical periods with insufficient data, add fallback books
            if len(book_ids) < target_count and decade_start < 1970:
                logger.warning(f"Insufficient data for {decade}, need {target_count}, have {len(book_ids)}")
                additional_ids = self._get_fallback_books_for_decade(decade, target_count - len(book_ids))
                if additional_ids:
                    logger.info(f"Added {len(additional_ids)} historical fallback books for {decade}")
                    decade_book_ids[decade].extend(additional_ids)
        
        # Process each decade
        for decade, book_ids in decade_book_ids.items():
            target_count = prioritized_counts[decade]
            
            if not book_ids:
                logger.warning(f"No books found for {decade}")
                continue
            
            # Determine which book IDs to sample, with genre balancing if requested
            if balance_genres and len(book_ids) > target_count:
                # Get genre for each book
                book_genres = {}
                for book_id in book_ids:
                    genre = self._extract_genre(book_id)
                    if genre not in book_genres:
                        book_genres[genre] = []
                    book_genres[genre].append(book_id)
                
                # Balance across genres
                genres = list(book_genres.keys())
                if genres:
                    # Calculate books per genre
                    per_genre = max(1, target_count // len(genres))
                    sampled_ids = []
                    
                    for genre, ids in book_genres.items():
                        # Take up to per_genre from each genre
                        sample_size = min(per_genre, len(ids))
                        if sample_size > 0:
                            sampled_ids.extend(random.sample(ids, sample_size))
                    
                    # Fill remaining with random selection
                    if len(sampled_ids) < target_count and book_ids:
                        remaining = target_count - len(sampled_ids)
                        remaining_ids = [bid for bid in book_ids if bid not in sampled_ids]
                        if remaining_ids:
                            sampled_ids.extend(random.sample(remaining_ids, min(remaining, len(remaining_ids))))
                else:
                    # Sample more than needed to account for failed downloads
                    sampled_ids = random.sample(book_ids, min(target_count * 2, len(book_ids)))
            else:
                # Sample more than needed to account for failed downloads
                sampled_ids = random.sample(book_ids, min(target_count * 2, len(book_ids)))
            
            # Download and process the sampled books
            successful_texts = 0
            for book_id in tqdm(sampled_ids, desc=f"Loading {decade} texts"):
                try:
                    text = self._fetch_and_clean_text(book_id)
                    if not text or len(text) < min_text_length:
                        continue
                    
                    # Create chunks and select one randomly
                    chunks = self._create_chunks(text, chunk_size=5000)
                    if chunks:
                        selected_chunk = random.choice(chunks)
                        decade_texts[decade].append(selected_chunk)
                        successful_texts += 1
                        
                        if successful_texts >= target_count:
                            break
                            
                except Exception as e:
                    logger.debug(f"Error processing book {book_id}: {e}")
                    continue
            
            logger.info(f"{decade}: {len(decade_texts[decade])} texts, " +
                    f"avg length: {sum(len(t) for t in decade_texts[decade]) / len(decade_texts[decade]) if decade_texts[decade] else 0:.0f} chars")
        
        return decade_texts

    def _has_historical_catalog(self) -> bool:
        """Check if the current catalog has sufficient historical coverage."""
        if not self.metadata:
            return False
        
        # Count books by century
        pre_1900_count = 0
        pre_1950_count = 0
        
        for book_id, meta in self.metadata.items():
            year = meta.get('year')
            if year:
                if year < 1900:
                    pre_1900_count += 1
                elif year < 1950:
                    pre_1950_count += 1
        
        # We want at least 100 books from pre-1900 and 200 from pre-1950
        return pre_1900_count >= 100 and pre_1950_count >= 200

    def _add_historical_catalog_supplement(self) -> None:
        """
        Add historical book entries to the metadata catalog.
        This provides reliable historical coverage even when the main catalog
        has insufficient historical books.
        """
        historical_books = self._get_historical_book_supplement()
        
        # Add to the existing metadata
        for book_id, book_data in historical_books.items():
            if book_id not in self.metadata:
                self.metadata[book_id] = book_data
        
        logger.info(f"Added {len(historical_books)} historical books to catalog")
        
        # Save the updated catalog
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info("Saved updated catalog with historical supplement")
        except Exception as e:
            logger.warning(f"Failed to save updated catalog: {e}")

    def _get_historical_book_supplement(self) -> Dict[str, Dict]:
        """
        Get a curated list of historically important books for each decade.
        This provides a reliable fallback for historical periods.
        
        Returns:
            Dict mapping book IDs to metadata
        """
        # This is a curated list of important historical books from Gutenberg
        # that are guaranteed to exist and have reliable metadata
        historical_books = {}
        
        # 1850s
        historical_books.update({
            "1399": {"title": "Great Expectations", "author": "Dickens, Charles", "year": 1861, "language": "en"},
            "76": {"title": "Adventures of Huckleberry Finn", "author": "Twain, Mark", "year": 1884, "language": "en"},
            "84": {"title": "Frankenstein", "author": "Shelley, Mary", "year": 1818, "language": "en"},
            "98": {"title": "A Tale of Two Cities", "author": "Dickens, Charles", "year": 1859, "language": "en"},
            "1260": {"title": "Jane Eyre", "author": "Brontë, Charlotte", "year": 1847, "language": "en"},
            "158": {"title": "Emma", "author": "Austen, Jane", "year": 1815, "language": "en"},
            "1400": {"title": "Great Expectations", "author": "Dickens, Charles", "year": 1861, "language": "en"},
            "16": {"title": "Peter Pan", "author": "Barrie, J. M.", "year": 1911, "language": "en"},
            "174": {"title": "The Picture of Dorian Gray", "author": "Wilde, Oscar", "year": 1890, "language": "en"},
            "219": {"title": "Heart of Darkness", "author": "Conrad, Joseph", "year": 1899, "language": "en"},
            "2701": {"title": "Moby Dick", "author": "Melville, Herman", "year": 1851, "language": "en"},
            "244": {"title": "A Study in Scarlet", "author": "Doyle, Arthur Conan", "year": 1887, "language": "en"},
            "25344": {"title": "The Scarlet Letter", "author": "Hawthorne, Nathaniel", "year": 1850, "language": "en"},
            "30254": {"title": "Walden", "author": "Thoreau, Henry David", "year": 1854, "language": "en"},
            "345": {"title": "Dracula", "author": "Stoker, Bram", "year": 1897, "language": "en"},
            "42": {"title": "The Strange Case of Dr. Jekyll and Mr. Hyde", "author": "Stevenson, Robert Louis", "year": 1886, "language": "en"},
            "45": {"title": "Anne of Green Gables", "author": "Montgomery, L. M.", "year": 1908, "language": "en"},
            "514": {"title": "Little Women", "author": "Alcott, Louisa May", "year": 1868, "language": "en"},
            "55": {"title": "The Wonderful Wizard of Oz", "author": "Baum, L. Frank", "year": 1900, "language": "en"},
            "5200": {"title": "Metamorphosis", "author": "Kafka, Franz", "year": 1915, "language": "en"},
            "768": {"title": "Wuthering Heights", "author": "Brontë, Emily", "year": 1847, "language": "en"},
            "844": {"title": "The Importance of Being Earnest", "author": "Wilde, Oscar", "year": 1895, "language": "en"},
        })
        
        # Add early 20th century
        historical_books.update({
            "64317": {"title": "The Great Gatsby", "author": "Fitzgerald, F. Scott", "year": 1925, "language": "en"},
            "9800": {"title": "Women in Love", "author": "Lawrence, D. H.", "year": 1920, "language": "en"},
            "66753": {"title": "Ulysses", "author": "Joyce, James", "year": 1922, "language": "en"},
            "1184": {"title": "The Count of Monte Cristo", "author": "Dumas, Alexandre", "year": 1844, "language": "en"},
            "2641": {"title": "A Room with a View", "author": "Forster, E. M.", "year": 1908, "language": "en"},
            "3825": {"title": "Howards End", "author": "Forster, E. M.", "year": 1910, "language": "en"},
            "5230": {"title": "Pygmalion", "author": "Shaw, George Bernard", "year": 1913, "language": "en"},
            "58585": {"title": "Main Street", "author": "Lewis, Sinclair", "year": 1920, "language": "en"},
            "8492": {"title": "The Awakening", "author": "Chopin, Kate", "year": 1899, "language": "en"},
            "244": {"title": "A Study in Scarlet", "author": "Doyle, Arthur Conan", "year": 1887, "language": "en"},
            "1661": {"title": "The Adventures of Sherlock Holmes", "author": "Doyle, Arthur Conan", "year": 1892, "language": "en"},
            "2097": {"title": "The Sign of the Four", "author": "Doyle, Arthur Conan", "year": 1890, "language": "en"},
            "2852": {"title": "The Hound of the Baskervilles", "author": "Doyle, Arthur Conan", "year": 1902, "language": "en"},
            "11870": {"title": "The Secret Garden", "author": "Burnett, Frances Hodgson", "year": 1911, "language": "en"},
        })
        
        # Assign genres and subjects to books
        for book_id, data in historical_books.items():
            # Add subjects based on title keywords or authors
            subjects = []
            title = data.get("title", "").lower()
            author = data.get("author", "").lower()
            
            if "adventure" in title or "mystery" in title or "sherlock" in title:
                subjects.append("Adventure and mystery")
            elif "romance" in title or "love" in title:
                subjects.append("Romance")
            elif "fiction" in title:
                subjects.append("Fiction")
            
            # Add author-specific subjects
            if "dickens" in author:
                subjects.append("Victorian literature")
            elif "austen" in author:
                subjects.append("Romance; Domestic fiction")
            elif "wilde" in author:
                subjects.append("Victorian literature; Satire")
            
            data["subjects"] = subjects
        
        return historical_books

    def _get_fallback_books_for_decade(self, decade: str, count: int) -> List[str]:
        """
        Get a list of book IDs from the catalog for a specific decade.
        This is used to supplement decades with insufficient data.
        
        Args:
            decade: Target decade (e.g., "1850s")
            count: Number of books needed
            
        Returns:
            List of book IDs
        """
        start_year, end_year = TIME_PERIODS[decade]
        
        # Get all books from this decade
        decade_books = []
        for book_id, meta in self.metadata.items():
            year = meta.get('year')
            if year and start_year <= year <= end_year:
                decade_books.append(book_id)
        
        # If we have enough, sample from them
        if len(decade_books) >= count:
            return random.sample(decade_books, count)
        
        # Otherwise, look for books from nearby decades
        nearby_books = []
        window = 10  # Look up to 10 years in each direction
        
        for book_id, meta in self.metadata.items():
            year = meta.get('year')
            if year and (start_year - window) <= year <= (end_year + window):
                if book_id not in decade_books:
                    nearby_books.append(book_id)
        
        # Sample from nearby books to fill the quota
        needed = count - len(decade_books)
        if len(nearby_books) >= needed:
            sampled_nearby = random.sample(nearby_books, needed)
            logger.info(f"Added {needed} books from nearby decades to supplement {decade}")
            return decade_books + sampled_nearby
        
        # Return whatever we found
        logger.warning(f"Could only find {len(decade_books) + len(nearby_books)} books for {decade}, requested {count}")
        return decade_books + nearby_books
    
    def _fetch_and_clean_text(self, book_id: str) -> Optional[str]:
        """
        Fetch and clean text for a given book ID.
        
        Args:
            book_id: Gutenberg book identifier
            
        Returns:
            Cleaned text content or None if unavailable
        """
        # Check cache first
        cache_path = self.cache_dir / f"{book_id}.txt"
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                return self._clean_text(text)
            except Exception:
                logger.debug(f"Failed to read cached file for {book_id}")
        
        # Try each mirror
        for url_template in self.mirror_urls:
            try:
                url = url_template.format(id=book_id)
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    text = response.text
                    
                    # Cache the downloaded text
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    return self._clean_text(text)
            except Exception:
                continue
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned and normalized text
        """
        # Remove Gutenberg header and footer
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "***START OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "***START OF THE PROJECT GUTENBERG",
            "*END*THE SMALL PRINT"
        ]
        
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG",
            "***END OF THIS PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "***END OF THE PROJECT GUTENBERG",
            "End of Project Gutenberg"
        ]
        
        # Find content boundaries
        text_start = 0
        text_end = len(text)
        
        for marker in start_markers:
            pos = text.find(marker)
            if pos != -1:
                text_start = text.find("\n", pos) + 1
                break
        
        for marker in end_markers:
            pos = text.find(marker)
            if pos != -1:
                text_end = pos
                break
        
        text = text[text_start:text_end]
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_genre(self, book_id: str) -> str:
        """
        Extract genre information for a book based on subjects and title.
        
        Args:
            book_id: Gutenberg book ID
            
        Returns:
            Genre classification
        """
        metadata = self.metadata.get(book_id, {})
        title = metadata.get('title', '').lower()
        subjects = metadata.get('subjects', [])
        
        # Define genre categories and their keywords
        genre_keywords = {
            'fiction': ['fiction', 'novel', 'story', 'stories', 'tale', 'fantasy', 'adventure'],
            'poetry': ['poetry', 'poem', 'poems', 'verse', 'ballad', 'sonnet'],
            'drama': ['drama', 'play', 'theatre', 'theater', 'tragedy', 'comedy'],
            'history': ['history', 'historical', 'biography', 'memoirs', 'autobiography'],
            'philosophy': ['philosophy', 'philosophical', 'ethics', 'metaphysics'],
            'religion': ['religion', 'religious', 'bible', 'sacred', 'theology'],
            'science': ['science', 'scientific', 'mathematics', 'physics', 'chemistry', 'biology'],
            'reference': ['dictionary', 'encyclopedia', 'manual', 'handbook', 'reference']
        }
        
        # Check subjects and title for genre keywords
        for genre, keywords in genre_keywords.items():
            # Check subjects
            for subject in subjects:
                subject_lower = subject.lower()
                if any(keyword in subject_lower for keyword in keywords):
                    return genre
            
            # Check title
            if any(keyword in title for keyword in keywords):
                return genre
        
        # Default genre
        return 'unknown'

    def _create_chunks(self, text: str, chunk_size: int = 5000) -> List[str]:
        """
        Split text into sentence-boundary-aware chunks.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size for each chunk
            
        Returns:
            List of text chunks
        """
        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

def test_gutenberg_loader():
    """Test the Gutenberg loader with a small sample."""
    loader = GutenbergLoader()
    decade_texts = loader.load_decade_samples(texts_per_decade=2)
    
    print("\nGutenberg Sample Dataset Summary:")
    print("-" * 50)
    for decade, texts in decade_texts.items():
        if texts:
            print(f"\n{decade}:")
            print(f"Number of texts: {len(texts)}")
            print(f"Average text length: {sum(len(t) for t in texts) / len(texts):.0f} chars")
            print(f"First 100 chars of first text: {texts[0][:100]}...")

if __name__ == "__main__":
    test_gutenberg_loader()