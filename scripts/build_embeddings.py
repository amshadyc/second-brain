"""
Generate embeddings for chunked notes and build FAISS index.

This script uses sentence-transformers to generate embeddings locally,
then creates a FAISS index for fast similarity search.
"""

import json
import numpy as np
import faiss
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    CHUNKED_NOTES_FILE,
    FAISS_INDEX_FILE,
    CHUNK_METADATA_FILE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION
)

# Import sentence-transformers for local embedding generation
from sentence_transformers import SentenceTransformer


def generate_embeddings():
    """
    Load chunks, generate embeddings, and build FAISS index.
    
    Returns:
        FAISS index and metadata mapping
    """
    print(f"Loading chunked notes from {CHUNKED_NOTES_FILE}...")
    
    if not CHUNKED_NOTES_FILE.exists():
        raise FileNotFoundError(
            f"Chunked notes file not found: {CHUNKED_NOTES_FILE}\n"
            "Please run chunk_notes.py first."
        )
    
    with open(CHUNKED_NOTES_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Load sentence-transformers model locally
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Extract texts for embedding
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings in batches for efficiency
    print("Generating embeddings...")
    batch_size = 32
    embeddings_list = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings_list.append(batch_embeddings)
        if (i + batch_size) % 100 == 0:
            print(f"Processed {min(i + batch_size, len(texts))} / {len(texts)} chunks")
    
    # Concatenate all embeddings
    embeddings = np.vstack(embeddings_list).astype('float32')
    print(f"Generated embeddings of shape: {embeddings.shape}")
    
    # Create FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    
    # Add embeddings to index
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")
    
    # Save FAISS index
    print(f"Saving FAISS index to {FAISS_INDEX_FILE}...")
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    
    # Save metadata mapping (chunk_id -> metadata)
    metadata = {
        str(i): {
            'chunk_id': chunk['chunk_id'],
            'text': chunk['text'],
            'original_index': chunk['original_index'],
            'created_at': chunk.get('created_at'),
            'modified_at': chunk.get('modified_at')
        }
        for i, chunk in enumerate(chunks)
    }
    
    print(f"Saving metadata to {CHUNK_METADATA_FILE}...")
    with open(CHUNK_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("Embedding generation complete!")
    return index, metadata


if __name__ == "__main__":
    index, metadata = generate_embeddings()
    print(f"\nFAISS index ready with {index.ntotal} vectors")
    print(f"Metadata contains {len(metadata)} entries")

