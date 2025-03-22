# Temporal Tokenizer Analysis

This project investigates temporal biases in large language model training data by analyzing tokenizer patterns.

## Overview

This research extends Hayase et al.'s work on Data Mixture Inference through tokenizer analysis to examine how language models represent content from different decades. It focuses on three key questions:
1. How are training data distributed across different time periods?
2. What is the correlation between data volume and temporal recency?
3. How do temporal distributions affect model performance across different decades?

## Methodology

The approach adapts linear programming techniques to identify temporal signatures in tokenizer merge rules, allowing us to:
- Identify decade-specific language patterns
- Quantify the temporal distribution of training data
- Measure the impact of these patterns on model performance

## Project Structure

- `notebooks/`: Jupyter notebooks for analysis and visualization
- `src/data/`: Data loading and processing utilities
- `src/validation/`: Evaluation metrics and statistical validation

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, matplotlib, seaborn, transformers, etc.

### Installation

# Clone the repository
git clone https://github.com/RoshaniPawar16/temporal_tokenizer_analysis.git

# Install dependencies
pip install -r requirements.txt