import logging
from pathlib import Path
import matplotlib.pyplot as plt
import random
import numpy as np

from src.data.dataset_manager import TemporalDatasetManager
from src.merge_rules_analyzer import EnhancedMergeRulesAnalyzer
from src.data.fill_missing_decades import fill_missing_decades

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_simplified_analysis():
    """Run a basic analysis to test the pipeline."""
    # Create simple dataset
    manager = TemporalDatasetManager()
    
    # Build small test dataset
    logger.info("Building test dataset...")
    print("Before building dataset")
    dataset = manager.build_temporal_dataset(texts_per_decade=5, save_dataset=True)
    print(f"Dataset built with {sum(len(texts) for texts in dataset.values())} texts")
    
    # Format for analysis (remove source info)
    analysis_dataset = {decade: [text for text, _ in texts] 
                       for decade, texts in dataset.items() 
                       if texts}  # Only include decades with data
    
    # Fill missing decades
    analysis_dataset = fill_missing_decades(analysis_dataset, min_texts_per_decade=3)
    
    # Run analysis with GPT-2
    logger.info("Running analysis...")
    analyzer = EnhancedMergeRulesAnalyzer("gpt2")
    results = analyzer.run_full_analysis(analysis_dataset)
    
    # Display results
    logger.info("Displaying results:")
    distribution = results.get("temporal_distribution", {})
    
    for decade, proportion in sorted(distribution.items()):
        logger.info(f"{decade}: {proportion:.2%}")
    
    # Create simple visualization
    plt.figure(figsize=(10, 6))
    decades = sorted(distribution.keys())
    proportions = [distribution[d] for d in decades]
    
    plt.bar(decades, proportions)
    plt.title("Temporal Distribution in GPT-2 Training Data")
    plt.xlabel("Decade")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("simple_temporal_analysis.png")
    logger.info("Analysis complete. Results saved to simple_temporal_analysis.png")

if __name__ == "__main__":
    run_simplified_analysis()