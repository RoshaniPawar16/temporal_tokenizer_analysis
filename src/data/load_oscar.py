from datasets import load_dataset
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

def load_oscar_data(language="en", split="train"):
    """
    Load data from the OSCAR corpus.
    
    Args:
        language (str): Language code
        split (str): Dataset split ('train' or 'validation')
    
    Returns:
        dataset: HuggingFace dataset object
    """
    try:
        dataset = load_dataset(
            OSCAR_DATASET_NAME,
            language,
            split=split,
        )
        return dataset
    except Exception as e:
        print(f"Error loading OSCAR dataset: {e}")
        return None

if __name__ == "__main__":
    # Test data loading
    dataset = load_oscar_data()
    if dataset is not None:
        print(f"Successfully loaded {len(dataset)} examples")
        print("\nSample text:")
        print(dataset[0]['text'][:500])