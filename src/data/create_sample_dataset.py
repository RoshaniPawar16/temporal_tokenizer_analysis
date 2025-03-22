import pandas as pd
import numpy.random as np
from pathlib import Path
import re
from datetime import datetime

def create_temporal_sample_dataset(output_path, size_per_decade=1000):
    """
    Create a sample dataset with text from different decades.
    Following Professor Wei's guidance on statistical validation and baseline approach.
    
    Args:
        output_path: Path to save the dataset
        size_per_decade: Number of samples per decade
    """
    # Decades we'll focus on (based on project plan)
    decades = list(range(1950, 2030, 10))
    
    data = []
    
    # Create sample data for each decade
    for decade in decades:
        # Generate synthetic text entries with temporal markers
        for _ in range(size_per_decade):
            # Add decade-specific vocabulary and patterns
            if decade < 1980:
                tech_terms = ["telegram", "radio", "typewriter"]
            elif decade < 2000:
                tech_terms = ["computer", "fax", "cd-rom"]
            else:
                tech_terms = ["smartphone", "website", "email"]
                
            # Create a sample text with temporal markers
            sample_text = f"In the {decade}s, people used {np.choice(tech_terms)}. "
            sample_text += "This is a sample text for temporal analysis."
            
            data.append({
                'text': sample_text,
                'decade': f"{decade}s",
                'year': decade + np.randint(0, 10)
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save dataset
    df.to_csv(output_path, index=False)
    print(f"Created sample dataset with {len(df)} entries")
    return df

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "temporal_sample.csv"
    create_temporal_sample_dataset(output_path)