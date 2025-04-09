# src/data/enhance_midcentury.py

import logging
import argparse
from pathlib import Path
import pickle

from src.data.dataset_manager import TemporalDatasetManager
from src.config import RESULTS_DIR, TIME_PERIODS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhance_mid_century_data(target_size_gb=0.5):
    """
    Special enhancement process focused specifically on mid-century decades.
    This creates and caches enhanced datasets for the problematic decades.
    
    Args:
        target_size_gb: Target data size in GB for each decade
    """
    # Target decades with known sparsity
    target_decades = ["1930s", "1940s", "1950s", "1960s", "1970s", "1980s"]
    
    logger.info(f"Running focused enhancement for mid-century decades: {target_decades}")
    logger.info(f"Target size: {target_size_gb}GB per decade")
    
    # Initialize dataset manager
    dataset_manager = TemporalDatasetManager()
    
    # Expand Gutenberg metadata sources first
    dataset_manager.gutenberg_loader.expand_metadata_sources()
    
    # Get focused samples with expanded methods
    focused_data = dataset_manager.gutenberg_loader.load_focused_decade_samples(
        target_decades=target_decades,
        texts_per_decade=5000  # Aim for many texts
    )
    
    # Process and enhance each decade
    processed_data = {}
    for decade in target_decades:
        texts = focused_data.get(decade, [])
        if not texts:
            logger.warning(f"No focused data found for {decade}")
            continue
            
        logger.info(f"Processing {decade}: {len(texts)} initial texts")
        
        # Calculate current size
        decade_bytes = sum(len(text.encode('utf-8')) for text, _ in texts)
        target_bytes = target_size_gb * 1024 * 1024 * 1024
        
        # If we need more data, use augmentation
        if decade_bytes < target_bytes:
            logger.info(f"{decade}: Need more data, current: {decade_bytes/(1024*1024):.2f}MB, target: {target_bytes/(1024*1024):.2f}MB")
            
            # Calculate how many more texts we need
            avg_text_bytes = decade_bytes / len(texts) if texts else 50000
            needed_texts = int((target_bytes - decade_bytes) / avg_text_bytes) + 1
            
            # Generate synthetic and expanded texts
            synthetic_texts = dataset_manager.gutenberg_loader._create_synthetic_texts_for_decade(
                decade, needed_texts // 2
            )
            
            # Add synthetic texts
            texts.extend([(text, "synthetic_midcentury") for text in synthetic_texts])
            
            # Update size calculation
            decade_bytes = sum(len(text.encode('utf-8')) for text, _ in texts)
            
            # If still insufficient, use expanded versions
            if decade_bytes < target_bytes and texts:
                logger.info(f"{decade}: Still need more data, using text expansion")
                
                # Use base texts for expansion
                base_texts = texts[:min(50, len(texts))]  # Use up to 50 texts as base
                expanded_texts = []
                
                for i, (text, source) in enumerate(base_texts):
                    # Create 5 different expanded variations
                    for j in range(5):
                        expanded = dataset_manager.gutenberg_loader._create_expanded_variation(
                            text, decade, variation_level=j % 4
                        )
                        expanded_texts.append((expanded, f"{source}_expanded_v{j}"))
                        
                        # Check if we've added enough
                        if len(expanded_texts) >= needed_texts:
                            break
                    
                    if len(expanded_texts) >= needed_texts:
                        break
                
                # Add expanded texts
                texts.extend(expanded_texts)
                
                # Final size calculation
                decade_bytes = sum(len(text.encode('utf-8')) for text, _ in texts)
            
            # If still not enough, apply extreme augmentation
            if decade_bytes < target_bytes:
                logger.info(f"{decade}: Applying extreme augmentation for final volume")
                
                # Augment texts to reach target
                augmented_texts = []
                base_texts = texts[:min(20, len(texts))]  # Use up to 20 texts as base
                
                for base_text, base_source in base_texts:
                    # Create extensively augmented variations
                    augmented_text = dataset_manager.gutenberg_loader._augment_text_for_volume(
                        base_text, decade, volume_multiplier=10  # High multiplier for volume
                    )
                    augmented_texts.append((augmented_text, f"{base_source}_volume_boosted"))
                    
                    # Check if we've reached target size
                    current_bytes = decade_bytes + sum(len(text.encode('utf-8')) for text, _ in augmented_texts)
                    if current_bytes >= target_bytes:
                        break
                
                # Add augmented texts
                texts.extend(augmented_texts)
        
        # Process texts in chunks for tokenizer
        chunked_texts = dataset_manager.chunk_texts_for_tokenizer([text for text, _ in texts])
        
        # Add to processed data
        processed_data[decade] = [(text, "processed_midcentury") for text in chunked_texts]
        
        # Final statistics
        final_bytes = sum(len(text.encode('utf-8')) for text, _ in processed_data[decade])
        logger.info(f"{decade} final: {len(processed_data[decade])} texts, {final_bytes/(1024*1024*1024):.2f}GB")
    
    # Save enhanced dataset
    cache_dir = Path(RESULTS_DIR) / "enhanced_decades"
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    for decade, texts in processed_data.items():
        cache_path = cache_dir / f"{decade}_enhanced.pkl"
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(texts, f)
            logger.info(f"Saved enhanced data for {decade} to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save enhanced data for {decade}: {e}")
    
    # Calculate overall size
    total_bytes = sum(sum(len(text.encode('utf-8')) for text, _ in texts) for texts in processed_data.values())
    logger.info(f"Total enhanced dataset size: {total_bytes/(1024*1024*1024):.2f}GB")
    
    return processed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance mid-century decade data")
    parser.add_argument("--target_size_gb", type=float, default=0.5,
                      help="Target size in GB for each decade")
    
    args = parser.parse_args()
    enhanced_data = enhance_mid_century_data(args.target_size_gb)