# src/data/create_paste_data.py
import json
from pathlib import Path
import random

# Create more comprehensive sample data covering multiple decades
british_library_data = [
    # 1850s
    {
        "record_id": "001850001",
        "title": "Victorian Essays on Society",
        "date": "1855",
        "text": "The industrial revolution has brought forth remarkable changes to our society. Steam engines and railways have transformed transportation beyond recognition...",
        "language_1": "English",
        "mean_wc_ocr": 0.92,
        "place": "London"
    },
    # 1880s
    {
        "record_id": "001880001",
        "title": "Scientific Progress and Society",
        "date": "1882",
        "text": "The telegraph has immeasurably altered our understanding of distance. Messages that once took weeks to deliver now arrive instantaneously across continents...",
        "language_1": "English",
        "mean_wc_ocr": 0.89,
        "place": "Edinburgh"
    },
    # 1900s equivalent
    {
        "record_id": "001900001",
        "title": "The Dawn of a New Century",
        "date": "1901-1910",
        "text": "The new century dawns with remarkable progress in science and industry. The motorcar is no longer a curiosity but begins to find practical application in daily life...",
        "language_1": "English",
        "mean_wc_ocr": 0.87,
        "place": "Manchester"
    },
    # 1950s
    {
        "record_id": "001950001",
        "title": "The Atomic Age",
        "date": "1952",
        "text": "Nuclear power promises to revolutionize energy production. The recent developments at Calder Hall demonstrate Britain's leading role in this peaceful application of atomic energy...",
        "language_1": "English",
        "mean_wc_ocr": 0.95,
        "place": "London"
    },
    # 1970s
    {
        "record_id": "001970001", 
        "title": "Computing and Modern Society",
        "date": "1975",
        "text": "The microprocessor represents a quantum leap in computing capability. These silicon chips, despite their minute size, contain thousands of transistors and promise to transform both industry and eventually the home...",
        "language_1": "English",
        "mean_wc_ocr": 0.96,
        "place": "Cambridge"
    },
    # 1990s
    {
        "record_id": "001990001",
        "title": "The Information Superhighway",
        "date": "1994",
        "text": "The Internet, once a tool for academics and military researchers, is rapidly expanding into businesses and homes. Electronic mail and the World Wide Web represent new frontiers in communication technology...",
        "language_1": "English",
        "mean_wc_ocr": 0.98,
        "place": "London"
    },
    # 2010s
    {
        "record_id": "002010001",
        "title": "Social Media and Society",
        "date": "2012",
        "text": "Facebook and Twitter have fundamentally altered human interaction. These platforms enable unprecedented connectivity while simultaneously raising questions about privacy, attention spans, and the nature of friendship...",
        "language_1": "English",
        "mean_wc_ocr": 0.99,
        "place": "London"
    },
    # 2020s
    {
        "record_id": "002020001",
        "title": "Artificial Intelligence in the Third Decade",
        "date": "2022",
        "text": "Large language models demonstrate capabilities that were science fiction just a decade ago. These systems can generate human-like text, translate between languages, and engage in coherent dialogue across countless domains...",
        "language_1": "English",
        "mean_wc_ocr": 0.99,
        "place": "London"
    }
]

# Add more sample texts for each decade to ensure multiple samples per period
decades = ["1850s", "1880s", "1900s", "1950s", "1970s", "1990s", "2010s", "2020s"]
existing_records = [entry["record_id"] for entry in british_library_data]

for decade in decades:
    year_base = int(decade[:4])
    
    # Add 4 more samples per decade (total of 5 per decade including the ones above)
    for i in range(4):
        record_id = f"00{year_base}{1000+i+2}"  # Ensure unique IDs
        while record_id in existing_records:
            record_id = f"00{year_base}{1000+random.randint(2, 999)}"
        
        year = random.randint(year_base, year_base+9)
        
        text_themes = {
            "1850s": ["Victorian society", "industrialization", "railway expansion", "colonial perspectives"],
            "1880s": ["scientific advances", "imperial expansion", "literary societies", "economic theory"],
            "1900s": ["Edwardian society", "early automobiles", "wireless telegraphy", "imperial politics"],
            "1950s": ["post-war reconstruction", "television culture", "cold war perspectives", "suburban life"],
            "1970s": ["economic uncertainty", "changing social values", "technological advances", "political upheaval"],
            "1990s": ["digital revolution", "post-cold war politics", "globalization", "modern arts"],
            "2010s": ["smartphone society", "financial crisis aftermath", "climate awareness", "digital privacy"],
            "2020s": ["pandemic reflections", "remote work", "algorithm ethics", "sustainability challenges"]
        }
        
        theme = text_themes[decade][i % len(text_themes[decade])]
        
        british_library_data.append({
            "record_id": record_id,
            "title": f"{theme.title()} in {year}",
            "date": str(year),
            "text": f"This sample text explores {theme} in {year}. The document contains sufficient length to be useful for tokenization analysis and temporal language pattern studies. The vocabulary and phrasing are representative of writing from the {decade} decade. Additional sentences ensure adequate length for meaningful analysis.",
            "language_1": "English",
            "mean_wc_ocr": random.uniform(0.85, 0.99),
            "place": random.choice(["London", "Edinburgh", "Oxford", "Cambridge", "Manchester", "Birmingham"])
        })

# Save the data
file_path = Path(__file__).parent / "paste_data.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(british_library_data, f, indent=2, ensure_ascii=False)

print(f"Created paste_data.json with {len(british_library_data)} sample texts across {len(decades)} decades at {file_path}")