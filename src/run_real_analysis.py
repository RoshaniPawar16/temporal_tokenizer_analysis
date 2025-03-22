# src/run_real_analysis.py

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
from src.merge_rules_analyzer import MergeRulesAnalyzer

def log_separator():
    logger.info("=" * 50)

def run_real_data_analysis(tokenizer_name="gpt2", texts_per_decade=5):
    """
    Run temporal analysis pipeline using only real data with Mac M2 optimizations.
    
    Args:
        tokenizer_name: Name of the pretrained tokenizer to analyze
        texts_per_decade: Target number of texts per decade (may not be reached for all decades)
    """
    start_time = time.time()
    log_separator()
    logger.info(f"Starting temporal analysis using only real data with {tokenizer_name}")
    log_separator()
    
    # Step 1: Build a fresh dataset
    logger.info("STEP 1: Building temporal dataset from scratch")
    dataset_manager = TemporalDatasetManager()
    
    # Build dataset without the use_fallback parameter
    dataset_with_sources = dataset_manager.build_temporal_dataset(
        texts_per_decade=texts_per_decade,
        balance_sources=True,
        save_dataset=True
    )
    
    # Convert format for analyzer (remove source information)
    dataset = {decade: [text for text, _ in texts] for decade, texts in dataset_with_sources.items()}
    
    # Log dataset coverage
    total_texts = sum(len(texts) for texts in dataset.values())
    logger.info(f"Dataset contains {total_texts} texts across {len([d for d, t in dataset.items() if t])} decades")
    
    for decade, texts in dataset.items():
        if texts:
            logger.info(f"  {decade}: {len(texts)} texts")
    
    # Free memory
    del dataset_with_sources
    del dataset_manager
    gc.collect()
    
    # Step 2: Initialize the merge rules analyzer
    logger.info("STEP 2: Setting up merge rules analyzer")
    analyzer = MergeRulesAnalyzer(tokenizer_name=tokenizer_name)
    analyzer.enable_memory_efficient_mode()
    
    # Step 3: Run the analysis with available data
    logger.info("STEP 3: Running merge rules analysis")
    
    # Filter out decades with no texts
    non_empty_dataset = {decade: texts for decade, texts in dataset.items() if texts}
    logger.info(f"Analyzing {len(non_empty_dataset)} decades with data")
    
    # Use batched analysis for memory efficiency
    usage_results = analyzer.analyze_merge_rule_usage_batched(non_empty_dataset)
    
    # Free memory
    gc.collect()
    
    # Step 4: Find distinctive rules
    logger.info("STEP 4: Finding distinctive rules")
    distinctive_rules = analyzer.find_decade_distinctive_rules(usage_results)
    
    # Step 5: Generate visualizations
    logger.info("STEP 5: Generating visualizations")
    viz_path = Path("results") / "merge_analysis" / f"{tokenizer_name}_real_data_distinctive_rules.png"
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    
    analyzer.visualize_rule_usage(
        usage_results=usage_results,
        distinctive_rules=distinctive_rules,
        save_path=viz_path
    )
    
    # Step 6: Save analysis results
    logger.info("STEP 6: Saving analysis results")
    # Create result directory if it doesn't exist
    result_dir = Path("results") / "merge_analysis"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Custom save for clearer file naming
    usage_path = result_dir / f"{tokenizer_name}_real_data_merge_usage.json"
    distinctive_path = result_dir / f"{tokenizer_name}_real_data_distinctive_rules.json"
    
    # Save usage results
    serializable_results = {}
    for decade, data in usage_results.items():
        serializable_results[decade] = {
            'total_tokens': data['total_tokens'],
            'unique_rules_applied': data['unique_rules_applied'],
            'top_rules': data['top_rules'],
            'normalized_usage': dict(data['normalized_usage'])
        }
    
    with open(usage_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
        
    # Save distinctive rules
    serializable_distinctive = {}
    for decade, rules in distinctive_rules.items():
        serializable_distinctive[decade] = rules
    
    with open(distinctive_path, 'w') as f:
        json.dump(serializable_distinctive, f, indent=2)
    
    # Print summary of findings
    log_separator()
    logger.info("ANALYSIS RESULTS SUMMARY")
    log_separator()
    
    # Print decade-level statistics
    logger.info("Decade Coverage:")
    for decade in sorted(usage_results.keys()):
        if decade in usage_results:
            logger.info(f"  {decade}: {usage_results[decade]['total_tokens']} tokens analyzed")
    
    # Print distinctive rules by decade
    logger.info("Most Distinctive Merge Rules by Decade:")
    for decade in sorted(distinctive_rules.keys()):
        if distinctive_rules[decade]:
            logger.info(f"\n{decade} distinctive patterns:")
            for rule, distinctiveness in distinctive_rules[decade][:5]:
                logger.info(f"  '{rule}': {distinctiveness:.2f}x more common than average")
        else:
            logger.info(f"\n{decade}: No distinctive patterns found")
    
    # Document limitations due to data coverage
    log_separator()
    logger.info("ANALYSIS LIMITATIONS")
    missing_decades = [decade for decade in dataset.keys() if decade not in non_empty_dataset]
    if missing_decades:
        logger.info(f"No data available for these decades: {', '.join(missing_decades)}")
    
    limited_decades = [decade for decade, texts in non_empty_dataset.items() if len(texts) < 3]
    if limited_decades:
        logger.info(f"Limited data (<3 texts) for these decades: {', '.join(limited_decades)}")
    
    # Log completion
    elapsed_time = time.time() - start_time
    log_separator()
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to:")
    logger.info(f"  - {viz_path}")
    logger.info(f"  - {usage_path}")
    logger.info(f"  - {distinctive_path}")
    
    return usage_results, distinctive_rules

if __name__ == "__main__":
    # Get tokenizer name from command line argument if provided
    tokenizer_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    
    # Get texts per decade from command line if provided
    texts_per_decade = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Run analysis
    run_real_data_analysis(
        tokenizer_name=tokenizer_name,
        texts_per_decade=texts_per_decade
    )