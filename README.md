# Local RAG System Demo

A Retrieval-Augmented Generation (RAG) system that runs entirely locally using ChromaDB for vector storage, SentenceTransformers for embeddings, and Ollama for language model inference.

## 🚀 Overview

This project demonstrates a complete RAG pipeline that can process text documents, create semantic embeddings, store them in a local vector database, and answer questions using a local LLM. The system is designed to work with text files and provides intelligent document chunking with semantic overlap for better context preservation.

## 🎯 Features

- **Local Processing**: Everything runs on your machine - no external API calls required
- **Intelligent Text Chunking**: Semantic document splitting with configurable overlap
- **Vector Search**: ChromaDB for efficient similarity search
- **Local LLM Integration**: Uses Ollama for natural language generation
- **Interactive Chat Interface**: Command-line interface for asking questions
- **Metadata Enrichment**: Automatic summarization of chunks for better retrieval
- **Flexible Input**: Supports text files with automatic preprocessing

## 🛠️ Tech Stack

- **Vector Database**: ChromaDB (persistent local storage)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Language Model**: Ollama (configurable model)
- **Text Processing**: Custom semantic chunking with overlap
- **Backend**: Python with async support

## 📋 Prerequisites

1. **Python 3.8+** installed on your system, I am using `uv` as package manager
2. **Ollama** installed and running locally
   ```bash
   
   # Pull a model (e.g., Gemma 3)
   ollama pull gemma3:latest
   ```



## 🚀 Quick Start

1. **Prepare your text file** (or use the included `nfl.txt` sample)

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Enter the path to your text file when prompted:**
   ```
   Enter the path to your TXT file: nfl.txt
   ```

4. **Wait for processing to complete:**
   - The system will load the embedding model
   - Create semantic text chunks
   - Generate embeddings and store in ChromaDB
   - Start the interactive chat interface

5. **Ask questions about your document:**
   ```
   Your question: What was the score of Super Bowl 50?
   ```

## 📁 Project Structure

```
rag_demo/
├── app.py              # Main RAG system implementation
├── requirements.txt           # Python dependencies
├── nfl.txt            # Sample text data (Super Bowl 50 article)
├── chroma_db/         # ChromaDB persistent storage (auto-created)
├── __pycache__/       # Python cache files
└── README.md          # This file
```

## ⚙️ Configuration

### Chunking Parameters
You can modify these parameters in the `create_text_chunks()` method:

```python
max_chunk_size = 1500  # Characters per chunk
min_chunk_size = 200   # Minimum characters to form a chunk
overlap = 100          # Characters of overlap between chunks
```

### LLM Settings
Modify the LLM parameters in the `query_llm()` method:

```python
"options": {
    "temperature": 0.1,    # Creativity level (0.0-1.0)
    "top_p": 0.9,         # Nucleus sampling
    "max_tokens": 2000    # Maximum response length
}
```

### Embedding Model
Change the embedding model in the `__init__()` method:

```python
self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Alternatives: 'all-mpnet-base-v2', 'multi-qa-MiniLM-L6-cos-v1'
```

## 💡 How It Works

### 1. Document Processing
- Loads text files and splits them into semantic chunks
- Uses paragraph boundaries as primary dividers
- Handles long paragraphs by splitting on sentences
- Adds configurable overlap between chunks for context preservation

### 2. Embedding Generation
- Converts text chunks into vector embeddings using SentenceTransformers
- Stores embeddings in ChromaDB with metadata
- Processes in batches for memory efficiency

### 3. Query Processing
- Converts user questions into embeddings
- Performs similarity search in ChromaDB
- Retrieves top-k most relevant chunks

### 4. Response Generation
- Constructs context-aware prompts
- Queries local Ollama LLM
- Returns natural language responses


## 📊 Example Usage

```bash
$ python app.py
Local RAG System
==============================
Enter the path to your TXT file: nfl.txt

Loading embedding model...
Initializing ChromaDB...
Loaded text file: nfl.txt
Text length: 2847 characters
Creating semantically meaningful text chunks from document...
Created 3 semantically meaningful chunks from document
Generating embeddings...
Processed batch 1/1
Embeddings generated and stored!

==================================================
Local Banking RAG System Ready!
Ask questions about your banking data.
Type 'quit' to exit.
==================================================

Your question: What teams played in Super Bowl 50?

Answer:  The Denver Broncos (AFC champion) and the Carolina Panthers (NFC champion) played in Super Bowl 50.
```

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

**Note**: This is a demonstration project. For production use, consider adding error handling, logging, authentication, and scalability improvements.
