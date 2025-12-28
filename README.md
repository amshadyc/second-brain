# Personal Knowledge RAG System

A local RAG (Retrieval-Augmented Generation) system for semantic search and AI-powered analysis of personal notes.

## Features

- Load notes from CSV file
- Clean and chunk text for optimal embedding
- Generate embeddings locally using sentence-transformers
- Store embeddings in FAISS for fast similarity search
- Retrieve relevant notes for queries
- Generate deep analysis using Gemini API
- Fully local (except for Gemini API calls)

## Installation

### 1. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Why each library:**

- `pandas`: Efficient CSV loading and data manipulation
- `numpy`: Numerical operations for embeddings and arrays
- `faiss-cpu`: Fast similarity search for vector databases (fully local)
- `sentence-transformers`: State-of-the-art semantic embeddings (runs offline)
- `google-generativeai`: Client library for Gemini API

### 3. Set Gemini API Key

Get your API key from: https://makersuite.google.com/app/apikey

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Or edit `config/api_keys.py` directly (not recommended for production).

## Exporting Notes from Google Keep

If you want to use your Google Keep notes with this system, follow these steps:

### 1. Export Notes from Google Keep

1. Go to [Google Takeout](https://takeout.google.com/)
2. Select **Keep** from the list of services
3. Click "Create export" and download the archive
4. Extract the archive to access your Keep notes (they will be in JSON format)

### 2. Convert Google Keep Notes to CSV

1. Copy all JSON files from your Google Keep export into the `data/keep_json/` folder

2. Run the conversion script:

   ```bash
   python scripts/all_keep_notes_to_csv.py
   ```

   This script will:

   - Read all JSON files from the `data/keep_json/` folder
   - Extract text content and timestamps
   - Create `data/all_notes.csv` with all your notes

**Note:** The script expects JSON files with the standard Google Keep export format. Each JSON file should contain `textContent`, `createdTimestampUsec`, and `userEditedTimestampUsec` fields.

## Usage

### Step 1: Load and Chunk Notes

```bash
python scripts/chunk_notes.py
```

This will:

- Load notes from `data/all_notes.csv`
- Clean the text
- Chunk notes into semantic pieces
- Save to `processed/chunked_notes.json`

### Step 2: Generate Embeddings and Build Index

```bash
python scripts/build_embeddings.py
```

This will:

- Generate embeddings for all chunks using sentence-transformers
- Build FAISS index for fast similarity search
- Save index to `embeddings/faiss_index`
- Save metadata to `embeddings/chunk_metadata.json`

### Step 3: Run CLI Application

```bash
python app/cli.py
```

The CLI supports three modes:

- `analysis`: Deep analysis with themes, patterns, and insights
- `summary`: Condensed narrative summary
- `patterns`: Detect repeated beliefs and thought loops

Change mode by typing: `mode:analysis`, `mode:summary`, or `mode:patterns`

## Project Structure

```
second_brain/
├── data/
│   └── all_notes.csv          # Input CSV file
├── processed/
│   └── chunked_notes.json     # Chunked notes
├── embeddings/
│   ├── faiss_index            # FAISS vector index
│   └── chunk_metadata.json    # Metadata mapping
├── prompts/
│   ├── analysis.txt           # Analysis prompt template
│   ├── summary.txt            # Summary prompt template
│   └── patterns.txt           # Patterns prompt template
├── scripts/
│   ├── all_keep_notes_to_csv.py  # Convert Google Keep JSON to CSV
│   ├── load_csv.py            # CSV loading utility
│   ├── chunk_notes.py         # Note chunking script
│   ├── build_embeddings.py    # Embedding generation script
│   └── retrieve.py            # Retrieval module
├── app/
│   └── cli.py                 # CLI application
└── config/
    ├── settings.py            # Configuration settings
    └── api_keys.py            # API key configuration
```

## CSV Format

The CSV file should have at least a `text` column. Optional columns:

- `created_at`: Creation timestamp
- `modified_at`: Modification timestamp

Example:

```csv
text,created_at,modified_at
"This is a note",1550637857917000,1550637857917000
"Another note",1551283379420000,1551283379420000
```
