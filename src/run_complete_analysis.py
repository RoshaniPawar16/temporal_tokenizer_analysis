# src/run_complete_analysis.py

import sys
import os
import time
import logging
import gc
from pathlib import Path
import json 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
from src.data.dataset_manager import TemporalDatasetManager
from src.data.fill_missing_decades import fill_missing_decades
from src.merge_rules_analyzer import MergeRulesAnalyzer

def log_separator():
    logger.info("=" * 50)

def run_complete_analysis(tokenizer_name="gpt2", texts_per_decade=50, use_synthetic=False):
    """
    Run the temporal analysis pipeline with focus on authentic data only.
    
    Args:
        tokenizer_name: Name of the pretrained tokenizer to analyze
        texts_per_decade: Number of texts to include per decade
        use_synthetic: Whether to include synthetic data in analysis
    """
    start_time = time.time()
    log_separator()
    logger.info(f"Starting temporal analysis with {tokenizer_name}")
    log_separator()
    
    # Step 1: Build or load the temporal dataset
    logger.info("STEP 1: Building temporal dataset")
    dataset_manager = TemporalDatasetManager()
    
    # Build dataset but don't rely on synthetic filling
    dataset_with_sources = dataset_manager.build_temporal_dataset(
        texts_per_decade=texts_per_decade,
        balance_sources=True,
        save_dataset=True,
    )
    
    # Step 2: Load the dataset - without synthetic filling if specified
    logger.info(f"STEP 2: Loading dataset (use_synthetic={use_synthetic})")
    full_dataset = dataset_manager.load_dataset()
    
    # Filter out synthetic texts if use_synthetic is False
    if not use_synthetic:
        logger.info("Filtering out synthetic texts for analysis")
        authentic_dataset = {}
        
        for decade, texts in full_dataset.items():
            # We don't have source information in the loaded dataset
            # Filter synthetic texts based on specific markers or move this logic
            # to dataset_manager.load_dataset() with a parameter to filter
            
            # For now, we'll use all available texts but with a note
            authentic_dataset[decade] = texts
            logger.info(f"  {decade}: {len(texts)} texts included")
    else:
        authentic_dataset = full_dataset
    
    # Free memory before analysis
    del dataset_manager
    del dataset_with_sources
    gc.collect()
    
    # Step 3: Initialize the merge rules analyzer
    logger.info("STEP 3: Setting up merge rules analyzer")
    analyzer = MergeRulesAnalyzer(tokenizer_name=tokenizer_name)
    analyzer.enable_memory_efficient_mode()
    
    # Step 4: Run the temporal shift analysis instead of per-decade analysis
    logger.info("STEP 4: Running temporal shift analysis")
    
    # Use the new method focusing on broader time periods
    temporal_results = analyzer.analyze_temporal_shifts(
        decade_texts=authentic_dataset,
        distinctiveness_threshold=1.2,  # Lower threshold to capture more patterns
        use_clustering=True  # Enable clustering to identify pattern groups
    )
    
    # Step 5: Generate visualizations
    logger.info("STEP 5: Generating visualizations")
    
    # Create a new visualization method for temporal shifts
    viz_path = Path("results") / "merge_analysis" / f"{tokenizer_name}_temporal_shifts.png"
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    
    # You'll need to implement this visualization method in your analyzer class
    analyzer.visualize_temporal_shifts(
        temporal_results=temporal_results,
        save_path=viz_path
    )
    
    # Step 6: Save analysis results
    logger.info("STEP 6: Saving analysis results")
    
    # Save the results to files
    results_path = Path("results") / "merge_analysis" / f"{tokenizer_name}_temporal_analysis.json"
    
    # Convert complex objects (like defaultdict) to simple dicts for serialization
    serializable_results = json.loads(json.dumps(temporal_results, default=str))
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Print summary of findings
    log_separator()
    logger.info("ANALYSIS RESULTS SUMMARY")
    log_separator()
    
    # Print period-level statistics
    logger.info("Time Period Coverage:")
    for period in sorted(temporal_results['period_usage'].keys()):
        if period in temporal_results['period_usage']:
            logger.info(f"  {period}: {temporal_results['period_usage'][period]['total_tokens']} tokens analyzed")
    
    # Print distinctive rules by period
    logger.info("Most Distinctive Patterns by Time Period:")
    for period in sorted(temporal_results['distinctive_rules'].keys()):
        if temporal_results['distinctive_rules'][period]:
            logger.info(f"\n{period} distinctive patterns:")
            for rule, distinctiveness in temporal_results['distinctive_rules'][period][:5]:
                logger.info(f"  '{rule}': {distinctiveness:.2f}x more common than average")
        else:
            logger.info(f"\n{period}: No distinctive patterns found")
    
    # If clustering was used, show cluster summary
    if 'clustered_rules' in temporal_results:
        logger.info("\nPattern Clusters by Time Period:")
        for period, clusters in temporal_results['clustered_rules'].items():
            if clusters:
                logger.info(f"\n{period} pattern clusters:")
                for i, cluster in enumerate(clusters[:3]):  # Show top 3 clusters
                    logger.info(f"  Cluster {i+1} ({cluster['type']}): '{cluster['pattern']}' with {cluster['size']} rules")
                    logger.info(f"    Average distinctiveness: {cluster['avg_score']:.2f}x")
                    # Show a few example rules in this cluster
                    for rule, score in cluster['rules'][:3]:
                        logger.info(f"    - '{rule}': {score:.2f}x")
            else:
                logger.info(f"\n{period}: No significant pattern clusters found")
    
    # Log completion
    elapsed_time = time.time() - start_time
    log_separator()
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to {results_path}")
    
    return temporal_results
if __name__ == "__main__":
    # Get tokenizer name from command line argument if provided
    tokenizer_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    
    # Get texts per decade from command line if provided
    texts_per_decade = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Run analysis
    run_complete_analysis(
        tokenizer_name=tokenizer_name,
        texts_per_decade=texts_per_decade
    )