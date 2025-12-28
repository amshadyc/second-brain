"""
Chunk notes into semantic pieces suitable for embedding.

This script splits notes into chunks of appropriate size for embedding
generation, preserving metadata for each chunk.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKED_NOTES_FILE
from scripts.load_csv import load_notes


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


def chunk_notes():
    """
    Load notes, chunk them, and save to JSON.
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    print("Loading notes...")
    df = load_notes()
    
    print("Chunking notes...")
    chunked_data = []
    chunk_id = 0
    
    for idx, row in df.iterrows():
        text = row['text_clean']
        chunks = chunk_text(text)
        
        for chunk_idx, chunk_content in enumerate(chunks):
            chunk_dict = {
                'chunk_id': chunk_id,
                'original_index': int(idx),
                'chunk_index': chunk_idx,
                'text': chunk_content,
                'created_at': row.get('created_at'),
                'modified_at': row.get('modified_at'),
                'total_chunks': len(chunks)
            }
            chunked_data.append(chunk_dict)
            chunk_id += 1
    
    print(f"Created {len(chunked_data)} chunks from {len(df)} notes")
    
    # Save to JSON
    print(f"Saving chunked notes to {CHUNKED_NOTES_FILE}...")
    with open(CHUNKED_NOTES_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, indent=2, ensure_ascii=False)
    
    print("Chunking complete!")
    return chunked_data


if __name__ == "__main__":
    chunked_data = chunk_notes()
    print(f"\nSample chunk:")
    print(chunked_data[0]['text'][:200])

