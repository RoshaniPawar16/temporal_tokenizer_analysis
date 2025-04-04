# src/run_analysis_with_visualizations.py

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
from src.validation.temporal_inference import TemporalDistributionInference
from src.validation.evaluation_metrics import TemporalEvaluationMetrics
from src.visualization.token_visualizer import TokenVisualizer
from src.visualization.temporal_distribution_visualizer import TemporalDistributionVisualizer
from src.visualization.comparative_visualizer import ComparativeVisualizer
from src.visualization.report_generator import ReportGenerator

def log_separator():
    logger.info("=" * 50)

def run_analysis_with_visualizations(
    tokenizer_name="gpt2", 
    distribution_type="uniform", 
    texts_per_decade=50, 
    visualize=True, 
    generate_report=True
):
    """
    Run the temporal analysis pipeline with comprehensive visualizations.
    
    Args:
        tokenizer_name: Name of the pretrained tokenizer to analyze
        distribution_type: Type of distribution to test against ('uniform', 'recency', 'historical', 'bimodal')
        texts_per_decade: Number of texts to include per decade
        visualize: Whether to generate visualizations
        generate_report: Whether to generate a detailed report
    """
    start_time = time.time()
    log_separator()
    logger.info(f"Starting temporal analysis with {tokenizer_name}")
    logger.info(f"Testing against {distribution_type} distribution")
    log_separator()
    
    results_dir = Path("results") / tokenizer_name / distribution_type
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Set up the dataset manager
    logger.info("STEP 1: Setting up dataset manager")
    dataset_manager = TemporalDatasetManager()
    
    # Step 2: Create controlled dataset based on distribution type
    logger.info(f"STEP 2: Creating {distribution_type} dataset")
    
    # Define distribution based on type
    if distribution_type == "uniform":
        # Equal distribution across all decades
        distribution = {decade: 1.0/len(time_periods.keys()) for decade in time_periods.keys()}
    elif distribution_type == "recency":
        # Higher representation of recent decades
        decades = sorted(time_periods.keys())
        distribution = {}
        for i, decade in enumerate(decades):
            # Linear increase from 3% to 12%
            distribution[decade] = 0.03 + (i / len(decades)) * 0.09
        # Normalize to ensure sum to 1
        total = sum(distribution.values())
        distribution = {d: v/total for d, v in distribution.items()}
    elif distribution_type == "historical":
        # Higher representation of historical decades
        decades = sorted(time_periods.keys())
        distribution = {}
        for i, decade in enumerate(decades):
            # Linear decrease from 12% to 3%
            distribution[decade] = 0.12 - (i / len(decades)) * 0.09
        # Normalize to ensure sum to 1
        total = sum(distribution.values())
        distribution = {d: v/total for d, v in distribution.items()}
    elif distribution_type == "bimodal":
        # High at both ends, low in middle
        decades = sorted(time_periods.keys())
        distribution = {}
        mid_point = len(decades) // 2
        for i, decade in enumerate(decades):
            # Distance from middle (normalized to 0-1)
            distance = abs(i - mid_point) / mid_point
            # More weight to points far from middle
            distribution[decade] = 0.03 + distance * 0.09
        # Normalize to ensure sum to 1
        total = sum(distribution.values())
        distribution = {d: v/total for d, v in distribution.items()}
    else:
        # Default to uniform if invalid type
        logger.warning(f"Invalid distribution type: {distribution_type}, using uniform")
        distribution = {decade: 1.0/len(time_periods.keys()) for decade in time_periods.keys()}
    
    # Create controlled dataset
    controlled_dataset = dataset_manager.create_controlled_dataset(
        distribution=distribution,
        total_texts=texts_per_decade * len(time_periods.keys())
    )
    
    # Step 3: Run the inference
    logger.info("STEP 3: Running temporal distribution inference")
    inference = TemporalDistributionInference(tokenizer_name=tokenizer_name)
    
    # Run analysis
    decade_patterns = inference.analyze_decade_patterns(controlled_dataset)
    distinctive_patterns = inference.find_distinctive_patterns(decade_patterns)
    inferred_distribution = inference.infer_temporal_distribution(
        decade_patterns=decade_patterns,
        num_merge_rules=3000,
        weight_early_merges=True
    )
    
    # Step 4: Evaluate results
    logger.info("STEP 4: Evaluating inference results")
    evaluator = TemporalEvaluationMetrics()
    evaluation_results = evaluator.evaluate_distribution(
        inferred=inferred_distribution,
        ground_truth=distribution,
        model_name=tokenizer_name
    )
    
    # Step 5: Generate visualizations
    if visualize:
        logger.info("STEP 5: Generating visualizations")
        
        # Set up visualizers
        token_viz = TokenVisualizer(results_dir=results_dir / "visualizations")
        dist_viz = TemporalDistributionVisualizer(results_dir=results_dir / "visualizations")
        
        # Create token visualizations
        token_viz.visualize_token_distinctiveness(
            distinctive_tokens=distinctive_patterns,
            save_path=results_dir / "visualizations" / "distinctive_patterns.png"
        )
        
        # Create distribution visualizations
        dist_viz.visualize_distribution(
            distribution=inferred_distribution,
            title=f"Inferred Temporal Distribution ({tokenizer_name})",
            save_path=results_dir / "visualizations" / "inferred_distribution.png"
        )
        
        dist_viz.compare_distributions(
            inferred=inferred_distribution,
            ground_truth=distribution,
            title=f"Inferred vs Ground Truth Distribution ({tokenizer_name})",
            save_path=results_dir / "visualizations" / "distribution_comparison.png"
        )
        
        dist_viz.visualize_error_analysis(
            inferred=inferred_distribution,
            ground_truth=distribution,
            metrics={
                "Log10(MSE)": evaluation_results["distribution_metrics"]["log10_mse"],
                "MAE": evaluation_results["distribution_metrics"]["mae"],
                "Rank Correlation": evaluation_results["decade_metrics"]["rank_correlation"]
            },
            save_path=results_dir / "visualizations" / "error_analysis.png"
        )
    
    # Step 6: Generate report
    if generate_report:
        logger.info("STEP 6: Generating analysis report")
        
        report_gen = ReportGenerator(results_dir=results_dir / "reports")
        
        # Prepare visualization paths for the report
        viz_paths = {
            "distribution": "visualizations/inferred_distribution.png",
            "comparison": "visualizations/distribution_comparison.png",
            "distinctive_patterns": "visualizations/distinctive_patterns.png",
            "evaluation": "visualizations/error_analysis.png"
        }
        
        # Create analysis results object
        analysis_results = {
            "tokenizer_info": {
                "name": tokenizer_name,
                "vocab_size": len(inference.merge_rules) if hasattr(inference, "merge_rules") else "Unknown"
            },
            "distribution": inferred_distribution,
            "distinctive_patterns": distinctive_patterns,
            "metrics": {
                "log10_mse": evaluation_results["distribution_metrics"]["log10_mse"],
                "mae": evaluation_results["distribution_metrics"]["mae"],
                "js_distance": evaluation_results["distribution_metrics"]["js_distance"],
                "rank_correlation": evaluation_results["decade_metrics"]["rank_correlation"]
            }
        }
        
        # Generate report
        report_gen.generate_model_report(
            model_name=tokenizer_name,
            analysis_results=analysis_results,
            evaluation_metrics=analysis_results["metrics"],
            visualizations=viz_paths,
            save_path=results_dir / "reports" / f"{tokenizer_name}_{distribution_type}_report.md"
        )
    
    # Step 7: Save results
    logger.info("STEP 7: Saving analysis results")
    
    # Save inference results
    results_path = results_dir / "inference_results.json"
    
    # Convert to serializable format
    results = {
        "tokenizer": tokenizer_name,
        "distribution_type": distribution_type,
        "ground_truth": distribution,
        "inferred_distribution": inferred_distribution,
        "evaluation": {
            "log10_mse": evaluation_results["distribution_metrics"]["log10_mse"],
            "mae": evaluation_results["distribution_metrics"]["mae"],
            "js_distance": evaluation_results["distribution_metrics"]["js_distance"],
            "rank_correlation": evaluation_results["decade_metrics"]["rank_correlation"]
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log completion
    elapsed_time = time.time() - start_time
    log_separator()
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to {results_path}")
    
    return results

def run_comparative_analysis(
    tokenizer_names=["gpt2", "gpt2-medium", "bert-base-uncased"],
    distribution_type="recency", 
    texts_per_decade=50
):
    """
    Run comparative analysis across multiple tokenizers.
    
    Args:
        tokenizer_names: List of tokenizer names to analyze
        distribution_type: Type of distribution to test against
        texts_per_decade: Number of texts per decade
    """
    start_time = time.time()
    log_separator()
    logger.info(f"Starting comparative analysis of {len(tokenizer_names)} tokenizers")
    logger.info(f"Testing against {distribution_type} distribution")
    log_separator()
    
    results_dir = Path("results") / "comparative" / distribution_type
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis for each tokenizer
    model_results = {}
    ground_truth = None
    
    for tokenizer_name in tokenizer_names:
        logger.info(f"Analyzing {tokenizer_name}")
        
        # Run analysis for this tokenizer
        results = run_analysis_with_visualizations(
            tokenizer_name=tokenizer_name,
            distribution_type=distribution_type,
            texts_per_decade=texts_per_decade,
            visualize=True,
            generate_report=True
        )
        
        # Store results
        model_results[tokenizer_name] = {
            "distribution": results["inferred_distribution"],
            "metrics": results["evaluation"]
        }
        
        # Store ground truth (same for all models)
        if ground_truth is None:
            ground_truth = results["ground_truth"]
    
    # Generate comparative visualizations
    logger.info("Generating comparative visualizations")
    
    # Set up visualizers
    comp_viz = ComparativeVisualizer(results_dir=results_dir / "visualizations")
    
    # Create model comparison visualization
    model_distributions = {model: results["distribution"] 
                          for model, results in model_results.items()}
    
    comp_viz.compare_distributions(
        model_distributions=model_distributions,
        ground_truth=ground_truth,
        save_path=results_dir / "visualizations" / "model_comparison.png"
    )
    
    # Create metrics comparison visualization
    model_metrics = {model: results["metrics"] 
                    for model, results in model_results.items()}
    
    comp_viz.compare_model_metrics(
        model_metrics=model_metrics,
        save_path=results_dir / "visualizations" / "metrics_comparison.png"
    )
    
    # Generate comparative report
    logger.info("Generating comparative report")
    
    report_gen = ReportGenerator(results_dir=results_dir / "reports")
    
    # Prepare visualization paths for the report
    viz_paths = {
        "comparison": "visualizations/model_comparison.png",
        "metrics_comparison": "visualizations/metrics_comparison.png"
    }
    
    # Generate report
    report_gen.generate_comparative_report(
        model_results=model_results,
        ground_truth=ground_truth,
        visualizations=viz_paths,
        save_path=results_dir / "reports" / f"comparative_{distribution_type}_report.md"
    )
    
    # Save comparative results
    results_path = results_dir / "comparative_results.json"
    
    # Create serializable results
    comparative_results = {
        "tokenizers": tokenizer_names,
        "distribution_type": distribution_type,
        "ground_truth": ground_truth,
        "model_results": {
            model: {
                "distribution": results["distribution"],
                "metrics": results["metrics"]
            }
            for model, results in model_results.items()
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(results_path, 'w') as f:
        json.dump(comparative_results, f, indent=2)
    
    # Log completion
    elapsed_time = time.time() - start_time
    log_separator()
    logger.info(f"Comparative analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to {results_path}")
    
    return comparative_results

if __name__ == "__main__":
    # Get arguments
    if len(sys.argv) > 1 and sys.argv[1] == "comparative":
        # Run comparative analysis
        tokenizers = sys.argv[2].split(",") if len(sys.argv) > 2 else ["gpt2", "gpt2-medium", "bert-base-uncased"]
        dist_type = sys.argv[3] if len(sys.argv) > 3 else "recency"
        texts_count = int(sys.argv[4]) if len(sys.argv) > 4 else 50
        
        run_comparative_analysis(
            tokenizer_names=tokenizers,
            distribution_type=dist_type,
            texts_per_decade=texts_count
        )
    else:
        # Run single model analysis
        tokenizer = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
        dist_type = sys.argv[2] if len(sys.argv) > 2 else "uniform"
        texts_count = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        
        run_analysis_with_visualizations(
            tokenizer_name=tokenizer,
            distribution_type=dist_type,
            texts_per_decade=texts_count
        )