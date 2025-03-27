# Temporal Distribution Inference Results Summary
    
Generated on: 2025-03-27 00:18:35

## Overview

This report summarizes the results of temporal distribution inference experiments 
with different tokenizers and distribution patterns.

### Results Directory Structure
### ## Performance Summary

| Tokenizer | Distribution | log10(MSE) | MAE | JS Distance | Rank Correlation |
|-----------|--------------|------------|-----|-------------|------------------|
| gpt2 | historical | -1.58 | 0.1372 | 0.5081 | -0.15 |
| gpt2 | bimodal | -1.42 | 0.1613 | 0.5426 | 0.36 |
| gpt2-medium | recency | -1.85 | 0.1036 | 0.3834 | 0.49 |
| gpt2 | uniform | -1.91 | 0.0849 | 0.6121 | 0.00 |
| gpt2 | recency | -1.56 | 0.1327 | 0.4851 | 0.15 |
| bert-base-uncased | recency | -1.68 | 0.1171 | 0.4061 | 0.77 |

## Comparison to Hayase Benchmark

The original Hayase et al. paper reported a log10(MSE) of -7.30±1.31. Our best result is 
-1.91, achieved by gpt2 on 
uniform distribution.

## Distribution Performance Analysis

### Historical Distribution

- Best performing tokenizer: **gpt2** (log10(MSE): -1.58)
- Worst performing tokenizer: **gpt2** (log10(MSE): -1.58)
- Average MAE: 0.1372
- Average Rank Correlation: -0.15

### Bimodal Distribution

- Best performing tokenizer: **gpt2** (log10(MSE): -1.42)
- Worst performing tokenizer: **gpt2** (log10(MSE): -1.42)
- Average MAE: 0.1613
- Average Rank Correlation: 0.36

### Recency Distribution

- Best performing tokenizer: **gpt2-medium** (log10(MSE): -1.85)
- Worst performing tokenizer: **gpt2** (log10(MSE): -1.56)
- Average MAE: 0.1178
- Average Rank Correlation: 0.47

### Uniform Distribution

- Best performing tokenizer: **gpt2** (log10(MSE): -1.91)
- Worst performing tokenizer: **gpt2** (log10(MSE): -1.91)
- Average MAE: 0.0849
- Average Rank Correlation: 0.00

## Key Findings

1. Performance varies significantly across distribution patterns, with uniform distributions typically being easier to infer.
2. There remains a substantial gap between our current results and the Hayase benchmark.
3. Different tokenizers show varying strengths across different distribution patterns.

## Next Steps

Based on these results, potential next steps include:

1. Further increasing the data volume for training and inference
2. Refining the linear programming approach for temporal distribution inference
3. Implementing more sophisticated statistical validation techniques
4. Investigating distinctive temporal markers in merge rules
