import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_existing_data(results_dir):
    """Analyze distribution files regardless of their format."""
    results_dir = Path(results_dir)
    dist_dir = results_dir / "distributions"
    summary_dir = results_dir / "summary_custom"
    summary_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all distribution files
    dist_files = list(dist_dir.glob("*_distribution.json"))
    
    # Create summary dataframe
    results = []
    
    for file_path in dist_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            tokenizer = data.get("tokenizer", "unknown")
            filename = file_path.stem
            
            # Extract distribution type from filename
            parts = filename.split("_")
            if len(parts) > 1:
                dist_type = parts[1]  # Assuming format like gpt2_uniform_...
            else:
                dist_type = "unknown"
                
            # Get evaluation metrics
            metrics = data.get("evaluation", {})
            
            # Get distribution data (in whatever format it's in)
            if "distribution" in data:
                distribution = data["distribution"]
            elif "inferred_distribution" in data:
                distribution = data["inferred_distribution"]
            else:
                distribution = {}
                
            # Add to results
            results.append({
                "tokenizer": tokenizer,
                "distribution_type": dist_type,
                "filename": filename,
                "log10_mse": metrics.get("log10_mse", 0),
                "mae": metrics.get("mae", 0),
                "js_distance": metrics.get("js_distance", 0),
                "rank_correlation": metrics.get("rank_correlation", 0),
                "distribution": distribution
            })
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create a summary dataframe
    if not results:
        print("No valid results found")
        return
        
    df = pd.DataFrame(results)
    
    # Save summary to CSV
    df[['tokenizer', 'distribution_type', 'log10_mse', 'mae', 'js_distance', 'rank_correlation']].to_csv(
        summary_dir / "metrics_summary.csv", index=False)
    
    # Create visualizations
    
    # 1. Performance by distribution type and tokenizer
    plt.figure(figsize=(12, 6))
    sns.barplot(x="distribution_type", y="log10_mse", hue="tokenizer", data=df)
    plt.axhline(y=-7.3, color='red', linestyle='--', label="Hayase Benchmark")
    plt.title("log₁₀(MSE) by Distribution Type", fontsize=14)
    plt.xlabel("Distribution Pattern", fontsize=12)
    plt.ylabel("log₁₀(MSE)", fontsize=12)
    plt.legend(title="Tokenizer")
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(summary_dir / "mse_by_distribution.png", dpi=300)
    plt.close()
    
    # 2. Distribution visualization for each run
    for _, row in df.iterrows():
        tokenizer = row['tokenizer']
        dist_type = row['distribution_type']
        distribution = row['distribution']
        
        if not distribution:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Sort decades chronologically
        decades = sorted(distribution.keys())
        values = [distribution[d] for d in decades]
        
        plt.bar(decades, values, color='skyblue')
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
            
        plt.title(f"{tokenizer} on {dist_type} Distribution", fontsize=14)
        plt.xlabel("Decade", fontsize=12)
        plt.ylabel("Proportion", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(summary_dir / f"{tokenizer}_{dist_type}_distribution.png", dpi=300)
        plt.close()
    
    # 3. Create a summary report
    with open(summary_dir / "summary_report.md", 'w') as f:
        f.write(f"# Analysis of {results_dir} Results\n\n")
        f.write("## Performance Summary\n\n")
        f.write("| Tokenizer | Distribution | log₁₀(MSE) | MAE | JS Distance | Rank Correlation |\n")
        f.write("|-----------|--------------|------------|-----|-------------|------------------|\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['tokenizer']} | {row['distribution_type']} | {row['log10_mse']:.2f} | {row['mae']:.4f} | {row['js_distance']:.4f} | {row['rank_correlation']:.2f} |\n")
        
        f.write("\n## Comparison to Hayase Benchmark\n\n")
        f.write("The original Hayase et al. paper reported a log₁₀(MSE) of -7.30±1.31. ")
        
        best_idx = df['log10_mse'].idxmax() if not df['log10_mse'].empty else None
        if best_idx is not None:
            f.write(f"Our best result is {df.loc[best_idx, 'log10_mse']:.2f}, achieved by {df.loc[best_idx, 'tokenizer']} on {df.loc[best_idx, 'distribution_type']} distribution.\n\n")
        else:
            f.write("No valid results found for comparison.\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("1. There remains a gap between our current results and the Hayase benchmark of -7.30 log₁₀(MSE)\n")
        f.write("2. Different tokenizers show varying strengths across different distribution patterns\n")
        f.write("3. Further refinement of the methodology is needed to improve accuracy\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. Increase the data volume for training and inference\n")
        f.write("2. Refine the linear programming approach for temporal distribution inference\n")
        f.write("3. Improve statistical validation through better bootstrap analysis\n")
    
    print(f"Analysis completed. Results saved to {summary_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_existing_data(sys.argv[1])
    else:
        print("Please provide the path to the results directory")
        print("Example: python analyze_results.py maxwell_results_march27")