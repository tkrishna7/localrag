# RAG System Architecture

## System Overview

This document provides a comprehensive overview of the Local RAG (Retrieval-Augmented Generation) System architecture, including data flow, components, and interactions.

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LOCAL RAG SYSTEM ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────┐
│   Text Input    │────│  Document        │────│   Text Chunks   │────│  Embedding  │
│   (.txt files)  │    │  Preprocessing   │    │   Generation    │    │  Generation │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────┘
                                │                         │                    │
                                ▼                         ▼                    ▼
                       ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐
                       │ Semantic        │    │ Overlap &       │    │ Sentence    │
                       │ Chunking        │    │ Context         │    │ Transformer │
                       │ • Paragraphs    │    │ Preservation    │    │ (MiniLM)    │
                       │ • Sentences     │    │ • 100 chars     │    │             │
                       │ • 1500 chars    │    │ • Summaries     │    │ 384-dim     │
                       └─────────────────┘    └─────────────────┘    └─────────────┘
                                                                             │
                                                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              VECTOR DATABASE                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   ChromaDB      │  │   Embeddings    │  │   Metadata      │              │
│  │                 │  │                 │  │                 │              │
│  │ • Persistent    │  │ • 384-dim       │  │ • chunk_id      │              │
│  │ • Cosine Sim.   │  │ • Batch stored  │  │ • position      │              │
│  │ • HNSW Index    │  │ • Local storage │  │ • filename      │              │
│  │                 │  │                 │  │ • summary       │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            QUERY PROCESSING                                    │
│                                                                                 │
│  User Query ──────┐                                                            │
│                   ▼                                                            │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │ Query Embedding │────│ Similarity      │────│ Top-K Retrieval │            │
│  │ • Same model    │    │ Search          │    │ • 5 chunks      │            │
│  │ • 384-dim       │    │ • Cosine dist.  │    │ • With metadata │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        RESPONSE GENERATION                                     │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │ Context         │────│ Prompt          │────│ LLM Response    │            │
│  │ Construction    │    │ Engineering     │    │ Generation      │            │
│  │ • Retrieved     │    │ • Instructions  │    │                 │            │
│  │   chunks        │    │ • Context       │    │  ┌───────────┐  │            │
│  │ • Metadata      │    │ • User query    │    │  │  Ollama   │  │            │
│  │ • Position info │    │                 │    │  │  (Local)  │  │            │
│  └─────────────────┘    └─────────────────┘    │  │           │  │            │
│                                                 │  │ Gemma 3   │  │            │
│                                                 │  │ Latest    │  │            │
│                                                 │  └───────────┘  │            │
│                                                 └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ User Interface  │
                            │ • CLI Chat      │
                            │ • Interactive   │
                            │ • Exit options  │
                            └─────────────────┘
```

## Component Details

### 1. Input Processing Layer

#### Document Preprocessing
- **Input**: Text files (.txt)
- **Function**: File reading and initial validation
- **Output**: Raw text content

#### Text Chunking Engine
- **Strategy**: Semantic chunking with paragraph boundaries
- **Parameters**:
  - Max chunk size: 1,500 characters
  - Min chunk size: 200 characters
  - Overlap: 100 characters
- **Features**:
  - Intelligent paragraph splitting
  - Long paragraph handling (sentence-level splitting)
  - Context preservation through overlap
  - Automatic summarization of chunks

### 2. Embedding Layer

#### Sentence Transformer Model
- **Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Processing**: Batch encoding for efficiency
- **Local**: No external API calls

### 3. Vector Storage Layer

#### ChromaDB Configuration
- **Type**: Persistent local storage
- **Similarity**: Cosine similarity
- **Indexing**: HNSW (Hierarchical Navigable Small World)
- **Metadata**: Rich metadata storage for enhanced retrieval

### 4. Retrieval Layer

#### Query Processing
```python
query -> embedding -> similarity_search -> top_k_results
```

#### Similarity Search
- **Algorithm**: Cosine similarity
- **Results**: Top 5 most relevant chunks
- **Metadata**: Position, filename, summary included

### 5. Generation Layer

#### Ollama Integration
- **Model**: Gemma 3 (configurable)
- **API**: Local HTTP endpoint (localhost:11434)
- **Parameters**:
  - Temperature: 0.1 (low creativity)
  - Top-p: 0.9 (nucleus sampling)
  - Max tokens: 2000

#### Prompt Engineering
- Context-aware prompts
- Detailed instructions for comprehensive answers
- Source attribution and grounding

## Data Flow Sequence

```
1. Document Loading
   ├── Read text file
   ├── Validate encoding
   └── Store in memory

2. Preprocessing
   ├── Split into paragraphs
   ├── Handle long paragraphs
   ├── Create overlapping chunks
   └── Generate chunk summaries

3. Embedding Generation
   ├── Batch process chunks
   ├── Generate 384-dim vectors
   ├── Store in ChromaDB
   └── Persist metadata

4. Query Processing
   ├── Accept user input
   ├── Generate query embedding
   ├── Search vector database
   └── Retrieve top-k chunks

5. Response Generation
   ├── Construct context
   ├── Engineer prompt
   ├── Query local LLM
   └── Return response

6. Interactive Loop
   ├── Display response
   ├── Accept new query
   └── Repeat until exit
```

## Performance Characteristics

### Memory Usage
- **Embedding Model**: ~500MB
- **ChromaDB**: ~50-100MB per 1000 chunks
- **Text Storage**: Proportional to document size
- **Peak Usage**: ~2-4GB for typical documents

### Processing Speed
- **Initial Setup**: 30-60 seconds (model loading)
- **Chunk Processing**: 1-5 seconds per 100 chunks
- **Query Response**: 2-10 seconds (depending on LLM)
- **Subsequent Queries**: 2-5 seconds (cached embeddings)

### Scalability
- **Document Size**: Efficiently handles documents up to 10MB
- **Chunk Limit**: Thousands of chunks supported
- **Concurrent Queries**: Single-threaded (can be extended)

## Configuration Points

### Chunking Parameters
```python
max_chunk_size = 1500  # Adjustable based on content
min_chunk_size = 200   # Prevents too-small chunks
overlap = 100          # Context preservation
```

### Embedding Model Options
```python
# Current: 'all-MiniLM-L6-v2' (384-dim, fast)
# Alternatives:
# 'all-mpnet-base-v2' (768-dim, better quality)
# 'multi-qa-MiniLM-L6-cos-v1' (384-dim, Q&A optimized)
```

### LLM Configuration
```python
temperature = 0.1      # Deterministic responses
top_p = 0.9           # Nucleus sampling
max_tokens = 2000     # Response length limit
```

## Security & Privacy

### Local Processing
- **No External APIs**: All processing happens locally
- **Data Privacy**: Documents never leave the machine
- **Network**: Only local Ollama communication

### Storage Security
- **ChromaDB**: Local file system storage
- **Embeddings**: Not human-readable
- **Metadata**: Stored locally in SQLite format

## Extension Points

### 1. Multi-Format Support
- Add PDF, DOCX, HTML parsers
- Implement format-specific chunking strategies

### 2. Advanced Retrieval
- Hybrid search (keyword + semantic)
- Re-ranking algorithms
- Query expansion techniques

### 3. Enhanced Generation
- Multiple LLM support
- Response validation
- Citation generation

### 4. Performance Optimization
- GPU acceleration for embeddings
- Async processing
- Caching strategies

### 5. User Interface
- Web interface
- API endpoints
- Batch processing mode

## Dependencies Architecture

```
┌─────────────────┐
│   Application   │
│    (app.py)     │
└─────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌─────────────┐
│ Core    │ │ ML/AI       │
│ Python  │ │ Libraries   │
├─────────┤ ├─────────────┤
│ os      │ │ sentence-   │
│ re      │ │ transformers│
│ json    │ │ chromadb    │
│ pandas  │ │ torch       │
│ numpy   │ │ transformers│
│ requests│ │             │
└─────────┘ └─────────────┘
         │
         ▼
┌─────────────────┐
│ External        │
│ Services        │
├─────────────────┤
│ Ollama          │
│ (localhost:     │
│  11434)         │
└─────────────────┘
```

This architecture provides a solid foundation for local RAG processing while maintaining privacy, performance, and extensibility.
