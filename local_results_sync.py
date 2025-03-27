"""
Script to retrieve and analyze results from Maxwell HPC.

This script helps transfer results from Maxwell to your local machine
and visualize the findings. It automatically creates uniquely named
folders with timestamps to prevent overwriting previous results.
"""

import os
import argparse
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

def create_unique_directory(base_name):
    """Create a unique directory based on date with auto-incrementing suffix if needed."""
    # Get current date
    today = datetime.now().strftime("%b%d").lower()  # e.g., "mar26"
    
    # Create base directory name
    base_dir = f"{base_name}_{today}"
    
    # Check if the base directory exists
    if not Path(base_dir).exists():
        return base_dir
    
    # If it exists, try adding numeric suffixes
    counter = 1
    while True:
        new_dir = f"{base_dir}_{counter}"
        if not Path(new_dir).exists():
            return new_dir
        counter += 1

def download_results(username, remote_dir, local_base_dir="maxwell_results"):
    """Download results from Maxwell to local machine in a uniquely named folder."""
    # Create a unique local directory
    local_dir = create_unique_directory(local_base_dir)
    local_path = Path(local_dir)
    local_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Creating new results directory: {local_dir}")
    
    # Create subdirectories
    subdirs = ["distributions", "figures", "metrics", "bootstrap", "logs"]
    for subdir in subdirs:
        (local_path / subdir).mkdir(exist_ok=True)
    
    # Configure SSH connection settings
    ssh_host = "127.0.0.1"
    ssh_port = "1024"
    
    # Download image files
    cmd_images = [
        "scp", 
        "-P", ssh_port,
        f"{username}@{ssh_host}:{remote_dir}/figures/*.png", 
        f"{local_dir}/figures/"
    ]
    print(f"Executing: {' '.join(cmd_images)}")
    subprocess.run(cmd_images)
    
    # Download distribution JSON files
    cmd_distributions = [
        "scp", 
        "-P", ssh_port,
        f"{username}@{ssh_host}:{remote_dir}/distributions/*.json", 
        f"{local_dir}/distributions/"
    ]
    print(f"Executing: {' '.join(cmd_distributions)}")
    subprocess.run(cmd_distributions)
    
    # Download bootstrap results
    cmd_bootstrap = [
        "scp", 
        "-P", ssh_port,
        f"{username}@{ssh_host}:{remote_dir}/bootstrap/*.json", 
        f"{local_dir}/bootstrap/"
    ]
    print(f"Executing: {' '.join(cmd_bootstrap)}")
    subprocess.run(cmd_bootstrap)
    
    # Download metrics
    cmd_metrics = [
        "scp", 
        "-P", ssh_port,
        f"{username}@{ssh_host}:{remote_dir}/metrics/*.json", 
        f"{local_dir}/metrics/"
    ]
    print(f"Executing: {' '.join(cmd_metrics)}")
    subprocess.run(cmd_metrics)
    
    # Download log files
    cmd_logs = [
        "scp", 
        "-P", ssh_port,
        f"{username}@{ssh_host}:{remote_dir}/../*.log", 
        f"{local_dir}/logs/"
    ]
    print(f"Executing: {' '.join(cmd_logs)}")
    subprocess.run(cmd_logs)
    
    print(f"Downloaded results to {local_dir}")
    return local_dir

def analyze_results(local_dir):
    """Analyze downloaded results and create summary visualizations."""
    # Create summary directory
    summary_dir = Path(local_dir) / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    # Load all distribution results
    dist_dir = Path(local_dir) / "distributions"
    if not dist_dir.exists():
        print(f"No distribution results found in {dist_dir}")
        return
    
    # Collect distribution files
    dist_files = list(dist_dir.glob("*_distribution.json"))
    if not dist_files:
        print(f"No distribution files found in {dist_dir}")
        return
    
    results = []
    
    # Process each file
    for file_path in dist_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract key information
                file_parts = file_path.stem.split("_")
                distribution_type = file_parts[1] if len(file_parts) > 1 else "unknown"
                
                results.append({
                    "tokenizer": data.get("tokenizer", "unknown"),
                    "distribution": distribution_type,
                    "log10_mse": data.get("evaluation", {}).get("log10_mse", 0),
                    "mae": data.get("evaluation", {}).get("mae", 0),
                    "js_distance": data.get("evaluation", {}).get("js_distance", 0),
                    "rank_correlation": data.get("evaluation", {}).get("rank_correlation", 0),
                    "filename": file_path.name
                })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create DataFrame
    if not results:
        print("No valid results found")
        return
        
    df = pd.DataFrame(results)
    
    # Save summary as CSV
    df.to_csv(summary_dir / "results_summary.csv", index=False)
    
    # Create summary visualizations
    create_summary_visualizations(df, summary_dir)
    
    # Generate detailed summary report
    generate_summary_report(df, local_dir, summary_dir)
    
    print(f"Analysis complete. Summary saved to {summary_dir}")
    return summary_dir

def create_summary_visualizations(df, output_dir):
    """Create summary visualizations from results DataFrame."""
    # Set visualization style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("muted")
    
    # 1. MSE by distribution and tokenizer
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x="distribution", y="log10_mse", hue="tokenizer")
    plt.title("log10(MSE) by Distribution and Tokenizer", fontsize=14)
    plt.xlabel("Distribution Pattern", fontsize=12)
    plt.ylabel("log10(MSE)", fontsize=12)
    plt.axhline(y=-7.3, color='red', linestyle='--', label="Hayase Benchmark")
    plt.legend(title="Tokenizer", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "mse_comparison.png", dpi=300)
    plt.close()
    
    # 2. MAE by distribution and tokenizer
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x="distribution", y="mae", hue="tokenizer")
    plt.title("Mean Absolute Error by Distribution and Tokenizer", fontsize=14)
    plt.xlabel("Distribution Pattern", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Tokenizer", fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "mae_comparison.png", dpi=300)
    plt.close()
    
    # 3. Rank correlation by distribution and tokenizer
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x="distribution", y="rank_correlation", hue="tokenizer")
    plt.title("Rank Correlation by Distribution and Tokenizer", fontsize=14)
    plt.xlabel("Distribution Pattern", fontsize=12)
    plt.ylabel("Rank Correlation", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Tokenizer", fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_comparison.png", dpi=300)
    plt.close()
    
    # 4. Performance overview
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # MSE
    sns.barplot(data=df, x="tokenizer", y="log10_mse", hue="distribution", ax=axs[0, 0])
    axs[0, 0].set_title("log10(MSE) by Tokenizer", fontsize=14)
    axs[0, 0].set_xlabel("Tokenizer", fontsize=12)
    axs[0, 0].set_ylabel("log10(MSE)", fontsize=12)
    axs[0, 0].axhline(y=-7.3, color='red', linestyle='--', label="Hayase")
    axs[0, 0].legend(title="Distribution")
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    # MAE
    sns.barplot(data=df, x="tokenizer", y="mae", hue="distribution", ax=axs[0, 1])
    axs[0, 1].set_title("MAE by Tokenizer", fontsize=14)
    axs[0, 1].set_xlabel("Tokenizer", fontsize=12)
    axs[0, 1].set_ylabel("MAE", fontsize=12)
    axs[0, 1].legend(title="Distribution")
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    axs[0, 1].tick_params(axis='x', rotation=45)
    
    # Jensen-Shannon Distance
    sns.barplot(data=df, x="tokenizer", y="js_distance", hue="distribution", ax=axs[1, 0])
    axs[1, 0].set_title("Jensen-Shannon Distance by Tokenizer", fontsize=14)
    axs[1, 0].set_xlabel("Tokenizer", fontsize=12)
    axs[1, 0].set_ylabel("Jensen-Shannon Distance", fontsize=12)
    axs[1, 0].legend(title="Distribution")
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    axs[1, 0].tick_params(axis='x', rotation=45)
    
    # Rank Correlation
    sns.barplot(data=df, x="tokenizer", y="rank_correlation", hue="distribution", ax=axs[1, 1])
    axs[1, 1].set_title("Rank Correlation by Tokenizer", fontsize=14)
    axs[1, 1].set_xlabel("Tokenizer", fontsize=12)
    axs[1, 1].set_ylabel("Rank Correlation", fontsize=12)
    axs[1, 1].legend(title="Distribution")
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    axs[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_overview.png", dpi=300)
    plt.close()
    
    # 5. Distribution-specific visualization
    for dist in df['distribution'].unique():
        dist_df = df[df['distribution'] == dist]
        
        plt.figure(figsize=(12, 7))
        
        # Create a grouped bar chart for all metrics per tokenizer
        x = range(len(dist_df))
        width = 0.2
        
        plt.bar([i - 1.5*width for i in x], dist_df['log10_mse'], width, label='log10(MSE)', color='royalblue')
        plt.bar([i - 0.5*width for i in x], dist_df['mae'], width, label='MAE', color='lightcoral')
        plt.bar([i + 0.5*width for i in x], dist_df['js_distance'], width, label='JS Distance', color='forestgreen')
        plt.bar([i + 1.5*width for i in x], dist_df['rank_correlation'], width, label='Rank Correlation', color='gold')
        
        plt.xlabel('Tokenizer', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.title(f'Performance Metrics for {dist} Distribution', fontsize=14)
        plt.xticks([i for i in x], dist_df['tokenizer'], rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(output_dir / f"{dist}_metrics.png", dpi=300)
        plt.close()

def generate_summary_report(df, local_dir, summary_dir):
    """Generate a detailed summary report in markdown format."""
    # Create summary report content
    summary_md = f"""# Temporal Distribution Inference Results Summary
    
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

This report summarizes the results of temporal distribution inference experiments 
with different tokenizers and distribution patterns.

### Results Directory Structure
### ## Performance Summary

| Tokenizer | Distribution | log10(MSE) | MAE | JS Distance | Rank Correlation |
|-----------|--------------|------------|-----|-------------|------------------|
"""

    # Add rows for each result
    for _, row in df.iterrows():
        summary_md += f"| {row['tokenizer']} | {row['distribution']} | {row['log10_mse']:.2f} | {row['mae']:.4f} | {row['js_distance']:.4f} | {row['rank_correlation']:.2f} |\n"
    
    # Add benchmark comparison
    summary_md += f"""
## Comparison to Hayase Benchmark

The original Hayase et al. paper reported a log10(MSE) of -7.30±1.31. Our best result is 
{df['log10_mse'].min():.2f}, achieved by {df.loc[df['log10_mse'].idxmin(), 'tokenizer']} on 
{df.loc[df['log10_mse'].idxmin(), 'distribution']} distribution.

## Distribution Performance Analysis

"""

    # Add distribution-specific analysis
    for dist in df['distribution'].unique():
        dist_df = df[df['distribution'] == dist]
        best_tokenizer = dist_df.loc[dist_df['log10_mse'].idxmin(), 'tokenizer']
        worst_tokenizer = dist_df.loc[dist_df['log10_mse'].idxmax(), 'tokenizer']
        
        summary_md += f"""### {dist.capitalize()} Distribution

- Best performing tokenizer: **{best_tokenizer}** (log10(MSE): {dist_df['log10_mse'].min():.2f})
- Worst performing tokenizer: **{worst_tokenizer}** (log10(MSE): {dist_df['log10_mse'].max():.2f})
- Average MAE: {dist_df['mae'].mean():.4f}
- Average Rank Correlation: {dist_df['rank_correlation'].mean():.2f}

"""

    # Add summary of findings
    summary_md += """## Key Findings

1. Performance varies significantly across distribution patterns, with uniform distributions typically being easier to infer.
2. There remains a substantial gap between our current results and the Hayase benchmark.
3. Different tokenizers show varying strengths across different distribution patterns.

## Next Steps

Based on these results, potential next steps include:

1. Further increasing the data volume for training and inference
2. Refining the linear programming approach for temporal distribution inference
3. Implementing more sophisticated statistical validation techniques
4. Investigating distinctive temporal markers in merge rules
"""

    # Save report
    with open(summary_dir / "summary_report.md", 'w') as f:
        f.write(summary_md)
    
    # Also save as a text file for easier viewing
    with open(summary_dir / "summary_report.txt", 'w') as f:
        f.write(summary_md)

def main():
    parser = argparse.ArgumentParser(description="Retrieve and analyze results from Maxwell HPC")
    parser.add_argument("--username", type=str, default="t06rp23", 
                      help="Your Maxwell username (default: t06rp23)")
    parser.add_argument("--remote_dir", type=str, required=False,
                      help="Path to results directory on Maxwell")
    parser.add_argument("--base_dir", type=str, default="maxwell_results", 
                      help="Base name for local results directory (date will be appended)")
    parser.add_argument("--analyze_only", type=str, 
                      help="Only analyze existing local directory, don't download")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        print(f"Analyzing existing results directory: {args.analyze_only}")
        summary_dir = analyze_results(args.analyze_only)
    else:
        # In this case, remote_dir is required
        if not args.remote_dir:
            parser.error("--remote_dir is required when not using --analyze_only")
            
        # Download results from Maxwell
        local_dir = download_results(args.username, args.remote_dir, args.base_dir)
        
        # Analyze results
        summary_dir = analyze_results(local_dir)

if __name__ == "__main__":
    main() 