"""
Configuration settings for the RAG system.

This file contains all configuration parameters for the system including
paths, model names, and processing parameters.
"""

from pathlib import Path

# Base directory for the project
BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
PROMPTS_DIR = BASE_DIR / "prompts"
RESPONSES_DIR = BASE_DIR / "responses"

# File paths
CSV_FILE = DATA_DIR / "all_notes.csv"
CHUNKED_NOTES_FILE = PROCESSED_DIR / "chunked_notes.json"
FAISS_INDEX_FILE = EMBEDDINGS_DIR / "faiss_index"
CHUNK_METADATA_FILE = EMBEDDINGS_DIR / "chunk_metadata.json"

# Embedding model configuration
# Using sentence-transformers for local embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and efficient model
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Chunking configuration
CHUNK_SIZE = 512  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

# Retrieval configuration
TOP_K = 10  # Number of chunks to retrieve for each query

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
PROMPTS_DIR.mkdir(exist_ok=True)
RESPONSES_DIR.mkdir(exist_ok=True)

