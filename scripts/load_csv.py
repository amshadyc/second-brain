"""
Load and preprocess notes from CSV file.

This script loads notes from the CSV file, cleans the text data,
and prepares it for chunking. Uses pandas for efficient CSV handling.
"""

import pandas as pd
import re
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CSV_FILE


def normalize_whitespace(text):
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with normalized whitespace
    """
    if pd.isna(text) or text == "":
        return ""
    # Convert to string if not already
    text = str(text)
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def load_notes():
    """
    Load notes from CSV file and clean them.
    
    Returns:
        DataFrame with cleaned notes and metadata
    """
    print(f"Loading notes from {CSV_FILE}...")
    
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
    
    # Load CSV using pandas
    df = pd.read_csv(CSV_FILE)
    
    print(f"Loaded {len(df)} notes from CSV")
    
    # Extract text column and clean
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain 'text' column")
    
    # Normalize whitespace in text
    df['text_clean'] = df['text'].apply(normalize_whitespace)
    
    # Remove empty notes
    df = df[df['text_clean'] != ""]
    
    print(f"After cleaning: {len(df)} notes remaining")
    
    # Preserve metadata columns if present
    metadata_cols = ['created_at', 'modified_at']
    for col in metadata_cols:
        if col not in df.columns:
            df[col] = None
    
    return df


if __name__ == "__main__":
    # Test loading
    df = load_notes()
    print(f"\nSample note:")
    print(df.iloc[0]['text_clean'][:200])
    print(f"\nTotal notes: {len(df)}")

