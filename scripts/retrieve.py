"""
Retrieve relevant chunks for a query using FAISS similarity search.

This script embeds a query, searches the FAISS index, and returns
the most relevant chunks with their metadata.
"""

import json
import numpy as np
import faiss
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    FAISS_INDEX_FILE,
    CHUNK_METADATA_FILE,
    EMBEDDING_MODEL_NAME,
    TOP_K
)

from sentence_transformers import SentenceTransformer


class Retriever:
    """
    Retrieval class for semantic search over notes.
    """
    
    def __init__(self):
        """Initialize retriever with FAISS index and embedding model."""
        print("Loading FAISS index and embedding model...")
        
        if not FAISS_INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {FAISS_INDEX_FILE}\n"
                "Please run build_embeddings.py first."
            )
        
        if not CHUNK_METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {CHUNK_METADATA_FILE}\n"
                "Please run build_embeddings.py first."
            )
        
        # Load FAISS index
        self.index = faiss.read_index(str(FAISS_INDEX_FILE))
        
        # Load metadata
        with open(CHUNK_METADATA_FILE, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load embedding model
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        print(f"Retriever initialized with {self.index.ntotal} vectors")
    
    def retrieve(self, query, top_k=TOP_K):
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: Search query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # Embed query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Retrieve chunks with metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk_meta = self.metadata.get(str(idx))
            if chunk_meta:
                result = {
                    'text': chunk_meta['text'],
                    'chunk_id': chunk_meta['chunk_id'],
                    'original_index': chunk_meta['original_index'],
                    'created_at': chunk_meta.get('created_at'),
                    'modified_at': chunk_meta.get('modified_at'),
                    'distance': float(distance)
                }
                results.append(result)
        
        return results


def retrieve_chunks(query, top_k=TOP_K):
    """
    Convenience function to retrieve chunks for a query.
    
    Args:
        query: Search query string
        top_k: Number of chunks to retrieve
        
    Returns:
        List of retrieved chunks
    """
    retriever = Retriever()
    return retriever.retrieve(query, top_k)


if __name__ == "__main__":
    # Test retrieval
    test_query = "What are the main themes in my notes?"
    print(f"Test query: {test_query}\n")
    
    results = retrieve_chunks(test_query, top_k=5)
    
    for i, result in enumerate(results, 1):
        print(f"Result {i} (distance: {result['distance']:.4f}):")
        print(result['text'][:200])
        print()

