# src/visualization/report_generator.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generate comprehensive reports with visualizations and analysis explanations.
    """
    
    def __init__(self, results_dir=None):
        """
        Initialize the report generator with output directory.
        
        Args:
            results_dir: Directory to save reports (default: results/reports)
        """
        if results_dir is None:
            results_dir = Path("results/reports")
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_model_report(self,
                          model_name,
                          analysis_results,
                          evaluation_metrics=None,
                          visualizations=None,
                          save_path=None):
        """
        Generate a comprehensive report for a model's temporal analysis.
        
        Args:
            model_name: Name of the model
            analysis_results: Results from temporal analysis
            evaluation_metrics: Optional evaluation metrics
            visualizations: Optional dict of visualization paths
            save_path: Path to save the report
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start building the report
        report = []
        report.append(f"# Temporal Analysis Report: {model_name}\n")
        report.append(f"Generated: {timestamp}\n\n")
        
        # Add model details
        report.append("## Model Information\n")
        report.append(f"- Model: {model_name}\n")
        if hasattr(analysis_results, 'get') and analysis_results.get('tokenizer_info'):
            report.append(f"- Tokenizer: {analysis_results['tokenizer_info'].get('name', 'Unknown')}\n")
            report.append(f"- Vocabulary Size: {analysis_results['tokenizer_info'].get('vocab_size', 'Unknown')}\n")
        report.append("\n")
        
        # Add analysis summary
        report.append("## Analysis Summary\n")
        
        # Add distribution summary if available
        if hasattr(analysis_results, 'get') and analysis_results.get('distribution'):
            report.append("### Temporal Distribution\n")
            report.append("Estimated proportion of training data by decade:\n\n")
            report.append("| Decade | Proportion |\n")
            report.append("|--------|------------|\n")
            
            for decade, proportion in sorted(analysis_results['distribution'].items()):
                report.append(f"| {decade} | {proportion:.2%} |\n")
            
            report.append("\n")
            
            # Add visualization reference if available
            if visualizations and 'distribution' in visualizations:
                report.append(f"![Temporal Distribution]({visualizations['distribution']})\n\n")
        
        # Add distinctive patterns if available
        if hasattr(analysis_results, 'get') and analysis_results.get('distinctive_patterns'):
            report.append("### Distinctive Patterns\n")
            report.append("Patterns that are distinctively common in specific decades:\n\n")
            
            for decade, patterns in sorted(analysis_results['distinctive_patterns'].items()):
                if patterns:
                    report.append(f"#### {decade}\n")
                    report.append("| Pattern | Distinctiveness |\n")
                    report.append("|---------|----------------|\n")
                    
                    for pattern, score in patterns[:10]:  # Show top 10
                        report.append(f"| {pattern} | {score:.2f}× |\n")
                    
                    report.append("\n")
            
            # Add visualization reference if available
            if visualizations and 'distinctive_patterns' in visualizations:
                report.append(f"![Distinctive Patterns]({visualizations['distinctive_patterns']})\n\n")
        
        # Add evaluation metrics if provided
        if evaluation_metrics:
            report.append("## Evaluation Metrics\n")
            report.append("| Metric | Value |\n")
            report.append("|--------|-------|\n")
            
            for metric, value in evaluation_metrics.items():
                report.append(f"| {metric} | {value} |\n")
            
            report.append("\n")
            
            # Add visualization reference if available
            if visualizations and 'evaluation' in visualizations:
                report.append(f"![Evaluation Metrics]({visualizations['evaluation']})\n\n")
        
        # Add interpretation section
        report.append("## Interpretation\n")
        
        # Add auto-generated interpretation if available
        if hasattr(analysis_results, 'get') and analysis_results.get('interpretation'):
            report.append(analysis_results['interpretation'])
        else:
            # Default interpretation
            report.append("### Key Findings\n")
            
            # Add recency bias analysis if distribution is available
            if hasattr(analysis_results, 'get') and analysis_results.get('distribution'):
                # Check for recency bias (higher proportion in recent decades)
                decades = sorted(analysis_results['distribution'].keys())
                if len(decades) >= 5:
                    older_half = decades[:len(decades)//2]
                    newer_half = decades[len(decades)//2:]
                    
                    older_avg = sum(analysis_results['distribution'][d] for d in older_half) / len(older_half)
                    newer_avg = sum(analysis_results['distribution'][d] for d in newer_half) / len(newer_half)
                    
                    if newer_avg > older_avg * 1.2:  # 20% higher
                        report.append("- **Recency Bias Detected**: The model's training data appears to have a significant ")
                        report.append("  recency bias, with higher representation of recent decades compared to historical periods.\n")
                    elif older_avg > newer_avg * 1.2:  # 20% higher
                        report.append("- **Historical Bias Detected**: Interestingly, the model's training data appears to have ")
                        report.append("  higher representation of historical periods compared to recent decades.\n")
                    else:
                        report.append("- **Balanced Temporal Distribution**: The model's training data appears to have a relatively ")
                        report.append("  balanced distribution across time periods, without strong recency bias.\n")
            
            # Add general interpretation
            report.append("- The distribution of training data across decades affects how the model represents ")
            report.append("  and generates content from different time periods. Under-represented periods ")
            report.append("  may have less accurate representations in the model output.\n")
            report.append("- Distinctive patterns reveal vocabulary and language structures that are characteristic ")
            report.append("  of specific time periods as captured by the tokenizer.\n")
        
        # Add recommendations
        report.append("### Recommendations\n")
        report.append("- For more balanced temporal representation, consider augmenting the training data ")
        report.append("  for under-represented decades.\n")
        report.append("- When using this model for historical text analysis, be aware of potential biases ")
        report.append("  in how content from different time periods is processed and generated.\n")
        
        # Generate the report string
        report_text = "".join(report)
        
        # Save the report
        if save_path is None:
            save_path = self.results_dir / f"{model_name}_report.md"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {save_path}")
        
        return report_text
    
    def generate_comparative_report(self,
                            model_results,
                            ground_truth=None,
                            visualizations=None,
                            save_path=None):
        """
        Generate a comparative report across multiple models.
        
        Args:
            model_results: Dict mapping model names to analysis results
            ground_truth: Optional ground truth distribution
            visualizations: Optional dict of visualization paths
            save_path: Path to save the report
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start building the report
        report = []
        report.append("# Comparative Temporal Analysis Report\n")
        report.append(f"Generated: {timestamp}\n\n")
        
        # Add models overview
        report.append("## Models Analyzed\n")
        for model_name in model_results.keys():
            report.append(f"- {model_name}\n")
        report.append("\n")
        
        # Add comparison of distributions
        report.append("## Distribution Comparison\n")
        
        # Add visualization reference if available
        if visualizations and 'comparison' in visualizations:
            report.append(f"![Distribution Comparison]({visualizations['comparison']})\n\n")
        
        # Add distribution table
        report.append("### Distribution by Decade\n")
        report.append("| Decade |")
        for model_name in model_results.keys():
            report.append(f" {model_name} |")
        if ground_truth:
            report.append(" Ground Truth |")
        report.append("\n")
        
        report.append("|--------|")
        for _ in model_results.keys():
            report.append("----------|")
        if ground_truth:
            report.append("----------|")
        report.append("\n")
        
        # Get all decades across all models
        all_decades = set()
        for results in model_results.values():
            if hasattr(results, 'get') and results.get('distribution'):
                all_decades.update(results['distribution'].keys())
        if ground_truth:
            all_decades.update(ground_truth.keys())
        
        # Add rows for each decade
        for decade in sorted(all_decades):
            report.append(f"| {decade} |")
            for model_name, results in model_results.items():
                if hasattr(results, 'get') and results.get('distribution'):
                    prop = results['distribution'].get(decade, 0)
                    report.append(f" {prop:.2%} |")
                else:
                    report.append(" - |")
            if ground_truth:
                prop = ground_truth.get(decade, 0)
                report.append(f" {prop:.2%} |")
            report.append("\n")
        
        report.append("\n")
        
        # Add metrics comparison if available
        has_metrics = any(hasattr(results, 'get') and results.get('metrics') 
                        for results in model_results.values())
        
        if has_metrics:
            report.append("## Evaluation Metrics Comparison\n")
            
            # Add visualization reference if available
            if visualizations and 'metrics_comparison' in visualizations:
                report.append(f"![Metrics Comparison]({visualizations['metrics_comparison']})\n\n")
            
            # Add metrics table
            all_metrics = set()
            for results in model_results.values():
                if hasattr(results, 'get') and results.get('metrics'):
                    all_metrics.update(results['metrics'].keys())
            
            report.append("| Metric |")
            for model_name in model_results.keys():
                report.append(f" {model_name} |")
            report.append("\n")
            
            report.append("|--------|")
            for _ in model_results.keys():
                report.append("----------|")
            report.append("\n")
            
            for metric in sorted(all_metrics):
                report.append(f"| {metric} |")
                for model_name, results in model_results.items():
                    if (hasattr(results, 'get') and results.get('metrics') 
                        and metric in results['metrics']):
                        value = results['metrics'][metric]
                        if isinstance(value, float):
                            report.append(f" {value:.4f} |")
                        else:
                            report.append(f" {value} |")
                    else:
                        report.append(" - |")
                report.append("\n")
            
            report.append("\n")
        
        # Add distinctive patterns comparison
        report.append("## Distinctive Patterns Comparison\n")
        report.append("This section compares the most distinctive patterns identified across different models.\n\n")
        
        # Collect distinctive patterns from all models
        period_distinctive_patterns = {}
        for model_name, results in model_results.items():
            if hasattr(results, 'get') and results.get('distinctive_patterns'):
                for period, patterns in results['distinctive_patterns'].items():
                    if period not in period_distinctive_patterns:
                        period_distinctive_patterns[period] = {}
                    period_distinctive_patterns[period][model_name] = patterns
        
        # Add pattern comparison by period
        for period, model_patterns in sorted(period_distinctive_patterns.items()):
            report.append(f"### {period}\n")
            report.append("| Model | Top Distinctive Patterns |\n")
            report.append("|-------|-------------------------|\n")
            
            for model_name, patterns in model_patterns.items():
                if patterns:
                    # Format top patterns
                    top_patterns = []
                    for pattern, score in patterns[:5]:  # Show top 5
                        top_patterns.append(f"{pattern} ({score:.2f}×)")
                    
                    patterns_text = ", ".join(top_patterns)
                    report.append(f"| {model_name} | {patterns_text} |\n")
                else:
                    report.append(f"| {model_name} | No distinctive patterns found |\n")
            
            report.append("\n")
        
        # Add interpretation section
        report.append("## Comparative Analysis\n")
        
        # Add distribution similarity analysis
        report.append("### Distribution Similarity\n")
        
        # Calculate distribution similarity between models
        if len(model_results) > 1:
            report.append("Similarity between model distributions:\n\n")
            report.append("| Model A | Model B | Similarity |\n")
            report.append("|--------|---------|------------|\n")
            
            model_names = list(model_results.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model_a = model_names[i]
                    model_b = model_names[j]
                    
                    # Get distributions
                    dist_a = model_results[model_a].get('distribution', {})
                    dist_b = model_results[model_b].get('distribution', {})
                    
                    # Calculate simple similarity (could be enhanced with proper distance metrics)
                    all_decades = sorted(set(dist_a.keys()) | set(dist_b.keys()))
                    diff_sum = sum(abs(dist_a.get(decade, 0) - dist_b.get(decade, 0)) for decade in all_decades)
                    similarity = 1 - (diff_sum / 2)  # Scale to 0-1
                    
                    report.append(f"| {model_a} | {model_b} | {similarity:.2%} |\n")
            
            report.append("\n")
        
        # Add ground truth comparison if available
        if ground_truth:
            report.append("### Comparison to Ground Truth\n")
            report.append("This section evaluates how closely each model's inferred distribution matches the ground truth.\n\n")
            
            report.append("| Model | MSE | MAE | Correlation |\n")
            report.append("|-------|-----|-----|-------------|\n")
            
            for model_name, results in model_results.items():
                dist = results.get('distribution', {})
                
                if dist:
                    # Calculate mean squared error
                    all_decades = sorted(set(dist.keys()) | set(ground_truth.keys()))
                    squared_errors = [(dist.get(decade, 0) - ground_truth.get(decade, 0))**2 for decade in all_decades]
                    mse = sum(squared_errors) / len(squared_errors)
                    
                    # Calculate mean absolute error
                    abs_errors = [abs(dist.get(decade, 0) - ground_truth.get(decade, 0)) for decade in all_decades]
                    mae = sum(abs_errors) / len(abs_errors)
                    
                    # Calculate correlation (simplified)
                    x = [dist.get(decade, 0) for decade in all_decades]
                    y = [ground_truth.get(decade, 0) for decade in all_decades]
                    
                    # Simple correlation calculation
                    mean_x = sum(x) / len(x)
                    mean_y = sum(y) / len(y)
                    
                    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
                    denominator_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
                    denominator_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
                    
                    correlation = 0
                    if denominator_x > 0 and denominator_y > 0:
                        correlation = numerator / (denominator_x * denominator_y)
                    
                    report.append(f"| {model_name} | {mse:.6f} | {mae:.6f} | {correlation:.4f} |\n")
                else:
                    report.append(f"| {model_name} | - | - | - |\n")
            
            report.append("\n")
        
        # Add conclusion and recommendations
        report.append("## Conclusion and Recommendations\n")
        
        # Add general conclusion
        report.append("Based on the comparative analysis of temporal distributions across models:\n\n")
        
        # Add model specific patterns
        for model_name, results in model_results.items():
            if hasattr(results, 'get') and results.get('distribution'):
                report.append(f"### {model_name}\n")
                
                # Analyze for recency bias
                decades = sorted(results['distribution'].keys())
                if len(decades) >= 4:
                    older_half = decades[:len(decades)//2]
                    newer_half = decades[len(decades)//2:]
                    
                    older_avg = sum(results['distribution'][d] for d in older_half) / len(older_half)
                    newer_avg = sum(results['distribution'][d] for d in newer_half) / len(newer_half)
                    
                    if newer_avg > older_avg * 1.2:  # 20% higher
                        report.append("- Shows significant recency bias with higher representation of recent decades\n")
                    elif older_avg > newer_avg * 1.2:  # 20% higher
                        report.append("- Shows unusual historical bias with higher representation of older decades\n")
                    else:
                        report.append("- Shows relatively balanced temporal distribution\n")
                
                # Find most represented decade
                if results['distribution']:
                    max_decade = max(results['distribution'].items(), key=lambda x: x[1])
                    report.append(f"- Highest representation: {max_decade[0]} ({max_decade[1]:.1%})\n")
                
                # Add more model-specific insights if available
                if hasattr(results, 'get') and results.get('interpretation'):
                    report.append(f"- {results['interpretation']}\n")
                
                report.append("\n")
        
        # Generate the report string
        report_text = "".join(report)
        
        # Save the report
        if save_path is None:
            save_path = self.results_dir / "comparative_report.md"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Comparative report saved to {save_path}")
        
        return report_text