cat > debug_bl_dataset.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=debug_bl_dataset
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=debug_bl_dataset_%j.log

# Display job information
echo "Running on node: $(hostname)"
echo "Job started at: $(date)"

# Load modules (using the ones that worked in your main script)
echo "Loading Python modules..."
module load anaconda3/2022.10 || echo "Failed to load anaconda module"

# Activate your virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate || echo "Failed to activate venv"
else
    echo "No venv found, using system Python"
fi

# Create the debugging script
echo "Creating debugging script..."
cat > debug_dataset.py << 'PYEOF'
from datasets import load_dataset
import json
import re

# Load the dataset
print("Loading British Library dataset...")
try:
    dataset = load_dataset("TheBritishLibrary/blbooks", "1500_1899", trust_remote_code=True, cache_dir="./hf_cache")
    
    # Analyze sample records
    print(f"Dataset loaded successfully. Contains {len(dataset['train'])} records.")
    
    # Sample the first 10 records
    print("\nSample records:")
    for i, record in enumerate(dataset['train'][:10]):
        print(f"\nRecord {i+1}:")
        # Print all fields to examine structure
        for key, value in record.items():
            if key != 'text':  # Skip printing the full text
                print(f"  {key}: {value}")
    
    # Analyze date formats
    print("\nAnalyzing date formats:")
    date_formats = {}
    year_distribution = {}
    
    for i, record in enumerate(dataset['train'][:1000]):  # Check first 1000 records
        date_value = record.get('date')
        
        # Record the format of the date
        date_type = type(date_value).__name__
        date_formats[date_type] = date_formats.get(date_type, 0) + 1
        
        # Try to extract year
        year = None
        
        if isinstance(date_value, int) and 1000 < date_value < 3000:
            year = date_value
        elif isinstance(date_value, str):
            year_match = re.search(r'\b(1[5-9]\d\d|20\d\d)\b', date_value)
            if year_match:
                year = int(year_match.group(1))
        
        if year:
            decade = f"{(year // 10) * 10}s"
            year_distribution[decade] = year_distribution.get(decade, 0) + 1
    
    print(f"Date formats found: {date_formats}")
    print(f"Year distribution by decade: {dict(sorted(year_distribution.items()))}")
    
    # Look specifically for 1850s and 1860s records
    print("\nLooking for 1850s and 1860s records:")
    decades_to_find = {'1850s': (1850, 1859), '1860s': (1860, 1869)}
    
    for decade, (start_year, end_year) in decades_to_find.items():
        count = 0
        examples = []
        
        for i, record in enumerate(dataset['train'][:10000]):
            date_value = record.get('date')
            year = None
            
            # Try all potential date formats
            if isinstance(date_value, int) and start_year <= date_value <= end_year:
                year = date_value
            elif isinstance(date_value, str):
                year_match = re.search(r'\b(1[5-9]\d\d|20\d\d)\b', date_value)
                if year_match:
                    year = int(year_match.group(1))
                    if start_year <= year <= end_year:
                        count += 1
                        if len(examples) < 3:
                            examples.append(record)
            
            # Also check publication_date if it exists
            pub_date = record.get('publication_date')
            if pub_date:
                if isinstance(pub_date, int) and start_year <= pub_date <= end_year:
                    count += 1
                    if len(examples) < 3:
                        examples.append(record)
                elif isinstance(pub_date, str):
                    year_match = re.search(r'\b(1[5-9]\d\d|20\d\d)\b', pub_date)
                    if year_match:
                        year = int(year_match.group(1))
                        if start_year <= year <= end_year:
                            count += 1
                            if len(examples) < 3:
                                examples.append(record)
            
            if count >= 10 and len(examples) >= 3:
                break
        
        print(f"\n{decade}: Found {count} records in first 10,000")
        if examples:
            print("Example records:")
            for i, ex in enumerate(examples):
                print(f"Example {i+1}:")
                for key, value in ex.items():
                    if key != 'text':  # Skip printing the full text
                        print(f"  {key}: {value}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")

PYEOF

# Check if datasets is installed
echo "Checking Python packages..."
python -c "import sys; print(f'Python version: {sys.version}')"
pip list | grep -E 'datasets|huggingface'

# Install datasets if needed
if ! python -c "import datasets" 2>/dev/null; then
    echo "Installing datasets package..."
    pip install datasets
fi

# Run the debugging script
echo "Running debug script..."
python debug_dataset.py

echo "Debug completed at $(date)"
EOF