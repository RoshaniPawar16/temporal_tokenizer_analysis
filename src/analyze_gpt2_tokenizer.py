import tiktoken
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tokenizer_analysis")

class DirectTokenizerAnalysis:
    """Direct analysis of GPT-2's tokenizer with tiktoken."""
    
    def __init__(self):
        """Initialize with direct access to tiktoken."""
        # Load GPT-2 tokenizer directly
        self.tokenizer = tiktoken.encoding_for_model("gpt2")
        logger.info(f"Loaded tiktoken encoding for GPT-2 with {self.tokenizer.n_vocab} tokens")
        
        # Extract internal structure - this is tiktoken specific
        self._extract_tokenizer_info()
        
        # Define output paths
        self.results_dir = Path("results/merge_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_temporal_dataset(self, custom_dataset_path=None):
        """
        Load a temporal dataset spanning different decades for tokenizer analysis.
        
        This method loads balanced text samples from different time periods (1850s-2020s)
        to analyze how tokenizer patterns vary across time. It works with the existing
        TemporalDatasetManager to ensure proper historical representation.
        
        Args:
            custom_dataset_path: Optional path to a pre-built dataset
            
        Returns:
            Dictionary mapping decades to lists of text samples
        """
        from src.data.dataset_manager import TemporalDatasetManager
        
        logger.info("Loading temporal dataset for tokenizer analysis")
        
        # Initialize the dataset manager
        dataset_manager = TemporalDatasetManager()
        
        # Check if we have a custom path to load from
        if custom_dataset_path:
            if os.path.exists(custom_dataset_path):
                try:
                    with open(custom_dataset_path, 'r', encoding='utf-8') as f:
                        self.dataset = json.load(f)
                    logger.info(f"Loaded custom dataset from {custom_dataset_path} with " 
                            f"{sum(len(texts) for texts in self.dataset.values())} texts")
                    return self.dataset
                except Exception as e:
                    logger.warning(f"Failed to load custom dataset: {e}")
        
        # Try to load existing dataset
        existing_dataset = dataset_manager.load_dataset()
        if existing_dataset and sum(len(texts) for texts in existing_dataset.values()) > 0:
            logger.info("Using existing temporal dataset")
            self.dataset = existing_dataset
            return self.dataset
        
        # Build a new dataset if needed
        logger.info("Building new temporal dataset - this may take some time")
        full_dataset = dataset_manager.build_temporal_dataset(
            texts_per_decade=50,  # Adjust as needed based on your analysis requirements
            balance_sources=True,
            save_dataset=True
        )
        
        # Extract just the text content from the (text, source) tuples
        self.dataset = {decade: [text for text, _ in texts] for decade, texts in full_dataset.items()}
        
        # Log dataset statistics
        total_texts = sum(len(texts) for texts in self.dataset.values())
        logger.info(f"Built temporal dataset with {total_texts} texts across {len(self.dataset)} decades")
        
        # Log sample counts per decade
        for decade, texts in self.dataset.items():
            if texts:
                avg_length = sum(len(text) for text in texts) / len(texts)
                logger.info(f"  {decade}: {len(texts)} texts, avg length: {avg_length:.0f} chars")
        
        return self.dataset

    def _extract_tokenizer_info(self):
        """Extract human-readable information from the tokenizer's vocabulary."""
        # Get the basic vocabulary
        self.vocab_size = self.tokenizer.n_vocab
        
        # Build a mapping from token IDs to human-readable representations
        token_samples = {}
        
        # Create a comprehensive token-to-text mapping
        # First, map basic ASCII characters
        for byte_val in range(32, 127):  # Printable ASCII
            token = bytes([byte_val])
            try:
                token_ids = self.tokenizer.encode(token)
                if token_ids:
                    token_id = token_ids[0]
                    token_text = token.decode('utf-8', errors='replace')
                    token_samples[token_id] = token_text
            except:
                pass
        
        # Add common words and their subwords
        common_words = ["the", "and", "in", "to", "a", "is", "of", "that", "for", "it", 
                    "with", "as", "was", "on", "be", "at", "by", "this", "from", "an",
                    "telegraph", "radio", "television", "computer", "internet", "smartphone"]
        
        for word in common_words:
            try:
                token_ids = self.tokenizer.encode(" " + word)  # Add space for natural tokenization
                decoded = self.tokenizer.decode(token_ids)
                # Store this mapping
                for token_id in token_ids:
                    if token_id not in token_samples:
                        token_samples[token_id] = f"{self.tokenizer.decode([token_id])}"
            except:
                pass
        
        # Add many more vocabulary samples by encoding full sentences
        sample_texts = [
            "The industrial revolution transformed society through technology and manufacturing.",
            "Radio and telegraph communication enabled rapid information exchange across distances.",
            "Television brought visual media into homes throughout the twentieth century.",
            "Computers and the internet revolutionized information processing and communication.",
            "Smartphones and social media have changed how people interact in modern society."
        ]
        
        for text in sample_texts:
            token_ids = self.tokenizer.encode(text)
            # Get individual token texts
            for i, token_id in enumerate(token_ids):
                if token_id not in token_samples:
                    # Decode this individual token
                    token_text = self.tokenizer.decode([token_id])
                    token_samples[token_id] = token_text.strip()
        
        # For all remaining tokens in our analysis, ensure we have some representation
        self.token_to_text = {}
        for token_id in range(self.vocab_size):
            if token_id in token_samples:
                self.token_to_text[token_id] = token_samples[token_id]
            else:
                # For tokens we don't have a sample for, decode them individually
                try:
                    text = self.tokenizer.decode([token_id])
                    self.token_to_text[token_id] = text.strip()
                except:
                    self.token_to_text[token_id] = f"Token-{token_id}"
        
        logger.info(f"Created human-readable mapping for {len(self.token_to_text)} tokens")
    
    def _get_token_context(self, token_id, period):
        """
        Find a real example of the token in context from the dataset.
        
        Args:
            token_id: Token identifier
            period: Time period to search for examples
            
        Returns:
            Short example phrase containing the token, or None if not found
        """
        if not hasattr(self, 'dataset') or period not in self.dataset:
            return None
            
        # Get token text
        if isinstance(token_id, int) and token_id in self.token_to_text:
            token_text = self.token_to_text[token_id]
        else:
            token_text = str(token_id)
        
        # Search for this token in period texts
        for text in self.dataset[period][:20]:  # Check first 20 texts
            if token_text in text:
                # Find token in context
                idx = text.find(token_text)
                start = max(0, idx - 15)
                end = min(len(text), idx + len(token_text) + 15)
                
                # Get context and clean it
                context = text[start:end].strip()
                context = context.replace('\n', ' ')
                
                return context
                
        return None
    
    def analyze_token_usage(self):
        """Analyze token usage patterns across different time periods."""
        # Group decades into periods
        period_mapping = {
            "historical": ["1850s", "1860s", "1870s", "1880s", "1890s"],
            "early_20th": ["1900s", "1910s", "1920s", "1930s", "1940s"],
            "mid_20th": ["1950s", "1960s", "1970s", "1980s"],
            "contemporary": ["1990s", "2000s", "2010s", "2020s"]
        }
        
        # Collect texts by period
        period_texts = {}
        for period, decades in period_mapping.items():
            period_texts[period] = []
            for decade in decades:
                if decade in self.dataset:
                    period_texts[period].extend(self.dataset[decade])
            
            logger.info(f"{period}: {len(period_texts[period])} texts")
        
        # Analyze token usage by period
        period_token_stats = {}
        
        for period, texts in period_texts.items():
            if not texts:
                logger.warning(f"No texts for {period}, skipping analysis")
                continue
                
            # Count token occurrences across all texts
            token_counts = Counter()
            total_tokens = 0
            
            # Process in smaller batches to manage memory
            batch_size = 20
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                for text in batch:
                    # Limit text length for analysis
                    if len(text) > 10000:
                        text = text[:10000]
                        
                    # Encode and count tokens
                    try:
                        tokens = self.tokenizer.encode(text)
                        for token in tokens:
                            token_counts[token] += 1
                            total_tokens += 1
                    except Exception as e:
                        logger.debug(f"Error encoding text: {e}")
            
            # Calculate token frequencies
            if total_tokens > 0:
                token_frequencies = {token: count/total_tokens 
                                    for token, count in token_counts.items()}
                
                # Store statistics
                period_token_stats[period] = {
                    "total_tokens": total_tokens,
                    "unique_tokens": len(token_counts),
                    "token_counts": token_counts,
                    "token_frequencies": token_frequencies
                }
        
        self.period_token_stats = period_token_stats
        return period_token_stats
    
    def find_distinctive_tokens(self, distinctiveness_threshold=1.2):
        """
        Find tokens that are distinctively common in specific time periods.
        Uses a proportional calculation without arbitrary capping.
        """
        if not hasattr(self, 'period_token_stats'):
            logger.warning("No token statistics. Run analyze_token_usage first.")
            return {}
            
        # Get all periods and tokens
        periods = list(self.period_token_stats.keys())
        all_tokens = set()
        for period_data in self.period_token_stats.values():
            all_tokens.update(period_data['token_frequencies'].keys())
        
        # Calculate average frequency across all periods for each token
        avg_token_freq = {}
        for token in all_tokens:
            freqs = []
            for period in periods:
                if period in self.period_token_stats:
                    freq = self.period_token_stats[period]['token_frequencies'].get(token, 0)
                    freqs.append(freq)
            
            if freqs:
                avg_token_freq[token] = sum(freqs) / len(freqs)
        
        # Find distinctive tokens for each period - without artificial capping
        distinctive_tokens = {}
        
        for period in periods:
            if period not in self.period_token_stats:
                continue
                
            period_distinctive = []
            
            for token, freq in self.period_token_stats[period]['token_frequencies'].items():
                avg = avg_token_freq.get(token, 0)
                if avg > 0 and freq > avg * distinctiveness_threshold:
                    # This token is distinctively more common in this period
                    # Calculate exact ratio without capping
                    distinctiveness = freq / avg
                    period_distinctive.append((token, self.token_to_text.get(token, f"Token-{token}"), distinctiveness))
            
            # Sort by distinctiveness
            distinctive_tokens[period] = sorted(period_distinctive, 
                                            key=lambda x: x[2], 
                                            reverse=True)
        
        self.distinctive_tokens = distinctive_tokens
        return distinctive_tokens
    
    def visualize_results(self, save_path=None):
        """Create clear visualizations of token usage patterns with readable token text."""
        if not hasattr(self, 'period_token_stats') or not hasattr(self, 'distinctive_tokens'):
            logger.warning("Missing analysis results. Run analyze_token_usage and find_distinctive_tokens first.")
            return
            
        periods = list(self.period_token_stats.keys())
        n_periods = len(periods)
        
        if n_periods == 0:
            logger.warning("No periods with data to visualize")
            return
            
        # Create a multi-panel figure
        fig = plt.figure(figsize=(14, 8 + 2*n_periods))
        
        # 1. Overall token counts
        ax1 = plt.subplot2grid((n_periods+1, 1), (0, 0))
        token_counts = [self.period_token_stats[period]['total_tokens'] for period in periods]
        ax1.bar(periods, token_counts, color='steelblue')
        ax1.set_title('Token Count by Time Period', fontsize=14)
        ax1.set_ylabel('Number of Tokens')
        
        # Add count labels
        for i, v in enumerate(token_counts):
            ax1.text(i, v + 0.05 * max(token_counts), f"{v:,}", 
                ha='center', va='bottom', fontsize=10)
        
        # 2. Distinctive tokens by period
        for i, period in enumerate(periods):
            ax = plt.subplot2grid((n_periods+1, 1), (i+1, 0))
            
            if period in self.distinctive_tokens and self.distinctive_tokens[period]:
                # Limit to top 10
                top_tokens = self.distinctive_tokens[period][:10]
                
                if top_tokens:
                    # Extract readable token texts and scores
                    token_texts = []
                    for t in top_tokens:
                        token_id = t[0]
                        # Get readable text for this token
                        if hasattr(self, 'token_to_text') and token_id in self.token_to_text:
                            text = self.token_to_text[token_id]
                        else:
                            text = f"Token-{token_id}"
                        # Make it display nicely - escape special chars
                        if text == '\n':
                            text = "\\n (newline)"
                        elif text.isspace():
                            text = f"'{text}' (whitespace)"
                        token_texts.append(text)
                    
                    scores = [t[2] for t in top_tokens]
                    
                    # Plot horizontal bars with actual scores (not capped)
                    bars = ax.barh(range(len(token_texts)), scores, color='skyblue')
                    ax.set_yticks(range(len(token_texts)))
                    ax.set_yticklabels(token_texts)
                    ax.set_title(f'Distinctive Tokens: {period}', fontsize=12)
                    ax.set_xlabel('Distinctiveness Score (× average)')
                    
                    # Set reasonable x-axis limits based on actual scores
                    max_score = max(scores) * 1.1  # Add 10% margin
                    ax.set_xlim(0, max_score)
                    
                    # Add value labels
                    for bar, score in zip(bars, scores):
                        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                            f'{score:.2f}×', va='center', fontsize=9)
                else:
                    ax.text(0.5, 0.5, f"No distinctive tokens found for {period}",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12)
            else:
                ax.text(0.5, 0.5, f"No distinctive tokens found for {period}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.results_dir / "gpt2_direct_analysis.png"
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
        
        # Show plot
        plt.show()
    
    def visualize_temporal_token_patterns(self):
        """
        Create an enhanced visualization of temporal token patterns that shows:
        1. Distribution of tokens across time periods
        2. More meaningful distinctiveness scores
        3. Context for how tokens appear in text
        """
        if not hasattr(self, 'period_token_stats') or not hasattr(self, 'distinctive_tokens'):
            logger.warning("Missing analysis results. Run analyze_token_usage and find_distinctive_tokens first.")
            return
                
        periods = list(self.period_token_stats.keys())
        n_periods = len(periods)
        
        # Create a multi-panel figure
        fig = plt.figure(figsize=(14, 16))
        
        # 1. Overall usage by time period
        ax1 = plt.subplot2grid((n_periods+2, 3), (0, 0), colspan=3)
        token_counts = [self.period_token_stats[period]['total_tokens'] for period in periods]
        ax1.bar(periods, token_counts, color='steelblue')
        ax1.set_title('Token Count by Time Period', fontsize=14)
        ax1.set_ylabel('Number of Tokens')
        
        # Add count labels
        for i, v in enumerate(token_counts):
            ax1.text(i, v + 0.05 * max(token_counts), f"{v:,}", 
                    ha='center', va='bottom', fontsize=10)
        
        # 2. Temporal heatmap of distinctive tokens
        # Extract top tokens across all periods
        all_top_tokens = []
        for period in periods:
            if period in self.distinctive_tokens:
                top_tokens_in_period = self.distinctive_tokens[period][:5]  # Top 5 per period
                all_top_tokens.extend([t[0] for t in top_tokens_in_period])
        
        # Remove duplicates while preserving order
        unique_top_tokens = []
        for token in all_top_tokens:
            if token not in unique_top_tokens:
                unique_top_tokens.append(token)
        
        # Limit to top 20 tokens overall
        top_tokens = unique_top_tokens[:20]
        
        # Create heatmap data
        heatmap_data = np.zeros((len(top_tokens), len(periods)))
        for i, token in enumerate(top_tokens):
            for j, period in enumerate(periods):
                if period in self.period_token_stats:
                    # Get normalized frequency
                    frequency = self.period_token_stats[period]['token_frequencies'].get(token, 0)
                    heatmap_data[i, j] = frequency
        
        # Get readable token texts
        token_texts = []
        for token in top_tokens:
            if isinstance(token, int) and token in self.token_to_text:
                text = self.token_to_text[token]
            else:
                text = str(token)
            token_texts.append(text)
        
        # Plot heatmap
        ax2 = plt.subplot2grid((n_periods+2, 3), (1, 0), colspan=3, rowspan=2)
        hm = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", 
                        xticklabels=periods, yticklabels=token_texts, ax=ax2)
        ax2.set_title('Token Usage Frequency Across Time Periods', fontsize=14)
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Token')
        
        # 3. Individual period breakdowns
        row = 3
        for period_idx, period in enumerate(periods):
            if period in self.distinctive_tokens and self.distinctive_tokens[period]:
                # Get top tokens for this period
                top_tokens = self.distinctive_tokens[period][:7]  # Top 7 
                
                if not top_tokens:
                    continue
                    
                # Extract token names and scores
                token_names = []
                scores = []
                
                for t in top_tokens:
                    token_id = t[0]
                    score = t[2]
                    
                    # Get readable text for token
                    if isinstance(token_id, int) and token_id in self.token_to_text:
                        text = self.token_to_text[token_id]
                    else:
                        text = str(token_id)
                        
                    # Create example context if possible
                    context = self._get_token_context(token_id, period)
                    if context:
                        text += f" (e.g., \"{context}\")"
                        
                    token_names.append(text)
                    scores.append(score)
                
                # Create horizontal bar chart
                ax = plt.subplot2grid((n_periods+2, 3), (row, 0), colspan=3, rowspan=1)
                bars = ax.barh(range(len(token_names)), scores, color='skyblue')
                ax.set_yticks(range(len(token_names)))
                ax.set_yticklabels(token_names)
                ax.set_title(f'Distinctive Tokens: {period}', fontsize=12)
                ax.set_xlabel('Distinctiveness Score (× average)')
                
                # Add value labels
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                            f'{score:.2f}×', va='center', fontsize=9)
                            
                row += 1
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.results_dir / "enhanced_temporal_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Enhanced visualization saved to {save_path}")
        
        # Show plot
        plt.show()
    
    def explain_temporal_trends(self):
        """Provide a human-readable explanation of temporal token trends."""
        if not hasattr(self, 'distinctive_tokens'):
            logger.warning("No distinctive tokens to explain. Run find_distinctive_tokens first.")
            return
            
        periods = list(self.period_token_stats.keys())
        period_names = {
            "historical": "Historical (pre-1900)",
            "early_20th": "Early 20th Century (1900-1945)",
            "mid_20th": "Mid 20th Century (1946-1989)",
            "contemporary": "Contemporary (1990-present)"
        }
        
        explanations = []
        explanations.append("# Temporal Token Analysis Explanation\n")
        explanations.append("## Overview\n")
        explanations.append("This analysis examines how tokenizer patterns vary across different time periods.\n")
        explanations.append("Each time period has distinctive vocabulary and language patterns, which are reflected in how the tokenizer processes text.\n\n")
        
        explanations.append("## Key Findings\n")
        
        for period in periods:
            if period in self.distinctive_tokens and self.distinctive_tokens[period]:
                period_name = period_names.get(period, period)
                explanations.append(f"### {period_name}\n")
                
                # Get top distinctive tokens
                top_tokens = self.distinctive_tokens[period][:8]
                
                # Calculate average token length for this period
                token_lengths = []
                for text in self.dataset.get(period, []):
                    tokens = self.tokenizer.encode(text)
                    token_lengths.append(len(tokens))
                
                avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
                
                explanations.append(f"Average tokens per text: {avg_length:.1f}\n")
                
                if top_tokens:
                    explanations.append("Distinctive tokens:\n")
                    for token, token_text, score in top_tokens:
                        explanations.append(f"- {token_text} ({score:.2f}× more common than average)\n")
                    
                    explanations.append("\nInterpretation: ")
                    
                    # Add period-specific interpretations
                    if period == "historical":
                        explanations.append("These tokens reflect Victorian-era terminology, industrial revolution vocabulary, and colonial empire vocabulary.\n")
                    elif period == "early_20th":
                        explanations.append("These tokens reflect early technological developments like automobiles, wireless technology, and early aviation.\n")
                    elif period == "mid_20th":
                        explanations.append("These tokens reflect post-war suburban expansion, Cold War terminology, and mid-century technological developments.\n")
                    elif period == "contemporary":
                        explanations.append("These tokens reflect digital technology, globalization, and modern business vocabulary.\n")
                    
                    explanations.append("\n")
        
        # Save explanation to file
        explanation_text = "".join(explanations)
        explanation_path = self.results_dir / "temporal_analysis_explanation.md"
        
        with open(explanation_path, 'w', encoding='utf-8') as f:
            f.write(explanation_text)
            
        logger.info(f"Explanation saved to {explanation_path}")
        
        return explanation_text
    
    def save_results(self, file_path=None):
        """Save analysis results to a JSON file."""
        if file_path is None:
            file_path = self.results_dir / "gpt2_direct_analysis.json"
                
        # Prepare results in a serializable format
        results = {
            "period_usage": {},
            "distinctive_tokens": {}
        }
        
        # Convert period usage stats
        if hasattr(self, 'period_token_stats'):
            for period, stats in self.period_token_stats.items():
                results["period_usage"][period] = {
                    "total_tokens": stats["total_tokens"],
                    "unique_tokens": stats["unique_tokens"],
                    "top_tokens": [(token, self.token_to_text.get(token, f"Token-{token}"), count) 
                                for token, count in stats["token_counts"].most_common(20)]
                }
        
        # Convert distinctive tokens
        if hasattr(self, 'distinctive_tokens'):
            for period, tokens in self.distinctive_tokens.items():
                results["distinctive_tokens"][period] = [
                    {"token_id": t[0], "token_text": t[1], "distinctiveness": t[2]} 
                    for t in tokens[:20]  # Limit to top 20
                ]
        
        # Save as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {file_path}")
        
        return results

def main():
    """Run the complete analysis pipeline."""
    analyzer = DirectTokenizerAnalysis()
    
    # Load your temporal dataset
    dataset = analyzer.load_temporal_dataset()
    
    if not dataset:
        logger.error("Failed to load dataset. Please check the dataset path.")
        return
    
    # Run analysis
    analyzer.analyze_token_usage()
    analyzer.find_distinctive_tokens()
    
    # Create visualizations
    analyzer.visualize_results()
    analyzer.visualize_temporal_token_patterns()  # Enhanced visualization
    analyzer.explain_temporal_trends()  # Generate explanation
    
    # Save results
    analyzer.save_results()
    
    # Print summary
    print("\nAnalysis Summary:")
    print("-" * 50)
    for period in sorted(analyzer.period_token_stats.keys()):
        stats = analyzer.period_token_stats[period]
        print(f"{period}: {stats['total_tokens']:,} tokens, {stats['unique_tokens']:,} unique")
    
    print("\nDistinctive Tokens by Period:")
    print("-" * 50)
    for period in sorted(analyzer.distinctive_tokens.keys()):
        tokens = analyzer.distinctive_tokens[period]
        if tokens:
            print(f"\n{period} distinctive tokens:")
            for token_id, token_text, distinctiveness in tokens[:5]:
                print(f"  '{token_text}': {distinctiveness:.2f}× more common than average")
        else:
            print(f"\n{period}: No distinctive tokens found")

if __name__ == "__main__":
    main()