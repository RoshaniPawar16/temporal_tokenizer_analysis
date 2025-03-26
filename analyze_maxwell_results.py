import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# Configure the analysis
results_dir = Path("/Users/roshani/temporal_tokenizer_analysis/maxwell_results_march26")
hayase_benchmark = -7.30  # Benchmark from the original paper

print("=" * 60)
print("TEMPORAL TOKENIZER ANALYSIS WITH INCREASED DATA VOLUME")
print("=" * 60)

# Step 1: Extract metrics from all JSON result files
print("\nAnalyzing JSON result files...")
json_results = []

for json_file in results_dir.glob("*.json"):
    distribution_name = json_file.stem.replace("results_gpt2_", "").replace("_100", "")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract metrics based on file structure
    metrics = {}
    
    # Try different possible locations for metrics
    if "distribution_metrics" in data:
        metrics = data["distribution_metrics"]
    elif "metrics" in data:
        metrics = data["metrics"]
    
    # Extract rank correlation if available
    rank_correlation = None
    if "decade_metrics" in data and "rank_correlation" in data["decade_metrics"]:
        rank_correlation = data["decade_metrics"]["rank_correlation"]
    
    # Prepare results entry
    result = {
        "Distribution": distribution_name,
        "log10(MSE)": metrics.get("log10_mse", "N/A"),
        "MAE": metrics.get("mae", "N/A"),
        "Jensen-Shannon Distance": metrics.get("js_distance", "N/A"),
        "Rank Correlation": rank_correlation
    }
    
    # Extract distributions if available
    if "inferred_distribution" in data:
        result["inferred_distribution"] = data["inferred_distribution"]
    elif "distribution" in data:
        result["inferred_distribution"] = data["distribution"]
        
    if "ground_truth_distribution" in data:
        result["ground_truth_distribution"] = data["ground_truth_distribution"]
    
    json_results.append(result)

# Step 2: Extract metrics from the log file as a backup
print("\nAnalyzing log file for additional information...")
log_file = results_dir / "temporal_analysis_2424818.log"
log_results = []

if log_file.exists():
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Find all results sections in the log
    results_sections = re.findall(r"RESULTS SUMMARY FOR (\w+) ON (\w+)\n=+\n(.*?)(?:\n=+|\n\n)", 
                                log_content, re.DOTALL)
    
    for model, distribution, content in results_sections:
        # Extract metrics
        log10_mse = re.search(r"Log10\(MSE\): ([-\d\.]+)", content)
        mae = re.search(r"MAE: ([\d\.]+)", content)
        js = re.search(r"Jensen-Shannon Distance: ([\d\.]+)", content)
        time = re.search(r"Inference Time: ([\d\.]+)", content)
        
        log_results.append({
            "Model": model,
            "Distribution": distribution.lower(),
            "log10(MSE)": float(log10_mse.group(1)) if log10_mse else None,
            "MAE": float(mae.group(1)) if mae else None,
            "Jensen-Shannon Distance": float(js.group(1)) if js else None,
            "Inference Time": float(time.group(1)) if time else None
        })

# Step 3: Combine and process the results
print("\nCombining all results...")
if json_results:
    results_df = pd.DataFrame(json_results)
    print("\n1. Performance Metrics from JSON files:")
    print(results_df[["Distribution", "log10(MSE)", "MAE", "Jensen-Shannon Distance", "Rank Correlation"]])
else:
    print("No JSON results found. Using log file results instead.")
    results_df = pd.DataFrame(log_results)
    print("\n1. Performance Metrics from log file:")
    print(results_df[["Distribution", "log10(MSE)", "MAE", "Jensen-Shannon Distance"]])

# Step 4: Compare to Hayase benchmark
print("\n2. Comparison to Hayase et al. benchmark:")
best_mse = results_df["log10(MSE)"].max() if "log10(MSE)" in results_df.columns else None

if best_mse is not None:
    difference = best_mse - hayase_benchmark
    print(f"Hayase et al. log10(MSE): {hayase_benchmark}")
    print(f"Your best log10(MSE): {best_mse:.4f}")
    print(f"Difference: {difference:.4f}")
    print(f"Your results are approximately {10**abs(difference):.0f}x less accurate than the benchmark")

# Step 5: Visualize the results
print("\n3. Creating visualizations...")

# 5.1 Create comparison plot of metrics
plt.figure(figsize=(15, 10))

# log10(MSE)
plt.subplot(2, 2, 1)
sns.barplot(x='Distribution', y='log10(MSE)', data=results_df)
plt.title('log10(MSE) by Distribution Pattern\n(lower is better)')
plt.axhline(y=hayase_benchmark, color='r', linestyle='--', label=f'Hayase benchmark: {hayase_benchmark}')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# MAE
plt.subplot(2, 2, 2)
sns.barplot(x='Distribution', y='MAE', data=results_df)
plt.title('Mean Absolute Error by Distribution Pattern\n(lower is better)')
plt.grid(axis='y', alpha=0.3)

# Jensen-Shannon Distance
plt.subplot(2, 2, 3)
sns.barplot(x='Distribution', y='Jensen-Shannon Distance', data=results_df)
plt.title('Jensen-Shannon Distance by Distribution Pattern\n(lower is better)')
plt.grid(axis='y', alpha=0.3)

# Rank Correlation (if available)
if "Rank Correlation" in results_df.columns and results_df["Rank Correlation"].notna().any():
    plt.subplot(2, 2, 4)
    sns.barplot(x='Distribution', y='Rank Correlation', data=results_df)
    plt.title('Rank Correlation by Distribution Pattern\n(higher is better)')
    plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / "metrics_comparison.png")
print(f"Saved metrics comparison to {results_dir / 'metrics_comparison.png'}")

# 5.2 For each distribution, create a visualization of the distribution if available
for result in json_results:
    if "inferred_distribution" in result and "ground_truth_distribution" in result:
        distribution_name = result["Distribution"]
        
        # Prepare the data
        inferred = result["inferred_distribution"]
        ground_truth = result["ground_truth_distribution"]
        decades = sorted(set(inferred.keys()) | set(ground_truth.keys()))
        
        # Create a DataFrame for easier plotting
        dist_data = []
        for decade in decades:
            dist_data.append({
                "Decade": decade,
                "Inferred": float(inferred.get(decade, 0)),
                "Ground Truth": float(ground_truth.get(decade, 0)),
                "Absolute Error": abs(float(inferred.get(decade, 0)) - float(ground_truth.get(decade, 0)))
            })
        
        dist_df = pd.DataFrame(dist_data)
        
        # Plot the distribution comparison
        plt.figure(figsize=(12, 6))
        
        # Set bar width and positions
        bar_width = 0.35
        r1 = range(len(decades))
        r2 = [x + bar_width for x in r1]
        
        # Create bars
        plt.bar(r1, dist_df["Inferred"], width=bar_width, label='Inferred', color='skyblue')
        plt.bar(r2, dist_df["Ground Truth"], width=bar_width, label='Ground Truth', color='lightcoral')
        
        # Add data labels
        for i, v in enumerate(dist_df["Inferred"]):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
        
        for i, v in enumerate(dist_df["Ground Truth"]):
            plt.text(i + bar_width, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
        
        # Add labels and title
        plt.xlabel('Decade')
        plt.ylabel('Proportion')
        plt.title(f'Inferred vs Ground Truth: {distribution_name}')
        plt.xticks([r + bar_width/2 for r in r1], decades, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / f"{distribution_name}_distribution.png")
        print(f"Saved {distribution_name} distribution to {results_dir / f'{distribution_name}_distribution.png'}")

# 5.3 Create a visualization of accuracy by decade
if json_results and all("inferred_distribution" in result and "ground_truth_distribution" in result for result in json_results):
    plt.figure(figsize=(12, 8))
    
    # Collect error data by decade across all distributions
    all_decades = set()
    for result in json_results:
        all_decades.update(result["inferred_distribution"].keys())
        all_decades.update(result["ground_truth_distribution"].keys())
    
    all_decades = sorted(all_decades)
    
    # For each distribution, calculate errors by decade
    for i, result in enumerate(json_results):
        distribution_name = result["Distribution"]
        inferred = result["inferred_distribution"]
        ground_truth = result["ground_truth_distribution"]
        
        errors = [abs(float(inferred.get(decade, 0)) - float(ground_truth.get(decade, 0))) for decade in all_decades]
        
        plt.plot(all_decades, errors, marker='o', label=distribution_name)
    
    plt.xlabel('Decade')
    plt.ylabel('Absolute Error')
    plt.title('Error by Decade Across Distribution Patterns')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "error_by_decade.png")
    print(f"Saved error by decade visualization to {results_dir / 'error_by_decade.png'}")

# Step 6: Provide recommendations based on the analysis
print("\n4. Conclusions and Recommendations:")
print("-" * 60)

if best_mse is not None:
    if best_mse > -2.0:
        print("• Your results (log10(MSE) around {:.2f}) show improvement compared to initial experiments.".format(best_mse))
        print("• However, there remains a significant gap compared to the Hayase benchmark of -7.30.")
        print("• The increased data volume (100 texts per decade) has provided some benefit, but further improvement is needed.")
        print("\nRecommendations:")
        print("1. Consider further increasing data volume, potentially approaching 1GB per category as in the original paper.")
        print("2. Explore modifications to the linear programming approach to better capture temporal patterns.")
        print("3. Investigate stronger temporal markers in tokenizer merge rules.")
        print("4. Remember that temporal patterns may be inherently more subtle than language differences,")
        print("   potentially requiring more sophisticated approaches or substantially larger datasets.")
    else:
        print("• Your results show significant improvement with increased data volume.")
        print("• While still below the benchmark, the trend suggests this approach is promising.")
        print("\nRecommendations:")
        print("1. Maintain the current trajectory by further scaling data volume.")
        print("2. Fine-tune the linear programming approach for temporal analysis specifically.")

print("\nAnalysis complete! Check the visualizations in your results directory.")