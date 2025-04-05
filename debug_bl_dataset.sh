cat > debug_bl_dataset.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=bl_debug_simple
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=bl_debug_simple_%j.log
# Display job information
echo "Running on node: $(hostname)"
echo "Job started at: $(date)"
# Load modules
module load anaconda3/2022.10 || echo "Failed to load anaconda module"
# Activate virtual environment
source venv/bin/activate || echo "Failed to activate venv"
# Create a simpler debugging script
cat > simple_debug.py << 'PYEOF'
from datasets import load_dataset
print("Loading British Library dataset...")
try:
    dataset = load_dataset("TheBritishLibrary/blbooks", "1500_1899", trust_remote_code=True, cache_dir="./hf_cache")
    print(f"Dataset loaded successfully. Contains {len(dataset['train'])} records.")
    # Check the type of the first few records
    print("\nChecking record types:")
    for i in range(min(5, len(dataset['train']))):
        record = dataset['train'][i]
        print(f"Record {i} type: {type(record)}")
        # If it's a string, print a sample
        if isinstance(record, str):
            print(f"Sample content: {record[:100]}...")
        # If it's a dictionary, print the keys
        elif isinstance(record, dict):
            print(f"Keys: {record.keys()}")
    # For dictionaries, check for date fields
    if len(dataset['train']) > 0 and isinstance(dataset['train'][0], dict):
        print("\nSearching for date-related fields:")
        sample_record = dataset['train'][0]
        date_fields = [field for field in sample_record.keys() if 'date' in field.lower() or 'year' in field.lower()]
        print(f"Potential date fields: {date_fields}")
        # Show values for these fields
        for field in date_fields:
            print(f"{field} value: {sample_record[field]}")
    # If records are strings, use regex to find dates in the text
    if len(dataset['train']) > 0 and isinstance(dataset['train'][0], str):
        print("\nSearching for dates in text content:")
        import re
        # Function to extract potential year mentions
        def extract_years(text):
            # Look for years in the 1800s
            year_matches = re.findall(r'\b(18[5-9][0-9])\b', text)
            return year_matches
        # Check a few records for years
        for i in range(min(10, len(dataset['train']))):
            text = dataset['train'][i]
            years = extract_years(text)
            if years:
                print(f"Record {i} contains years: {years}")
                print(f"Context: ...{text[text.find(years[0])-50:text.find(years[0])+50]}...")
except Exception as e:
    import traceback
    print(f"Error loading dataset: {e}")
    print(traceback.format_exc())
PYEOF
# Run the script
echo "Running simple debug script..."
python simple_debug.py
echo "Debug completed at $(date)"
EOF