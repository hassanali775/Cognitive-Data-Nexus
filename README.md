# Cognitive Data Nexus

A production-grade, air-gapped document intelligence and graph-augmented retrieval-augmented generation (GraphRAG) engine.

This repository contains the complete asynchronous processing factory and interactive observability console designed to operate with absolute data privacy—executing local semantic vector embedding, structural text partition chunking, and topological relation extraction with zero external API network dependencies.

---

## Technical Architecture Overview

Cognitive Data Nexus processes raw text assets through a decoupled, multi-dimensional execution pipeline:

Raw Assets --> DocumentProcessor --> Sliding-Window Partitions 
                                         |
                +────────────────────────┴────────────────────────+
                |                                                 |
                v                                                 v
    Geometric Vector Space                            Topological Graph Space
  [Sentence-Transformers Inference]                [Rule-Based Relation Extraction]
                |                                                 |
                v                                                 v
    Persistent ChromaDB Nodes                         Active NetworkX State Cache
                |                                                 |
                +────────────────────────┬────────────────────────+
                                         |
                                         v
                         Reciprocal Rank Fusion (RRF)
                                         |
                                         v
                         Deterministic Local Inference
                             [Air-Gapped Llama3]

---

## Core System Subsystems

* Asynchronous Parsing Engine: Native structural text extraction from multiple input formats including PDF, DOCX, and Markdown/TXT with dynamic character-set encoding safety.
* Sliding-Window Partitioning: High-performance semantic chunking utilizing parameterized token boundaries and overlap margins to preserve context continuity across document boundaries.
* Localized Embedding Vectorization: Local inference loop using `all-MiniLM-L6-v2` models to project textual data fragments into a dense 384-dimensional geometric coordinate space.
* Dual-Store Index Persistence: Simultaneous storage allocation across local vector indexes (ChromaDB) and topological relation caches (NetworkX) bound to a BM25 inverted keyword search layer.
* Hybrid Retrieval & Rank Fusion: Mathematical blending of structural proximity and topological graph path traversals using a custom Reciprocal Rank Fusion implementation to optimize grounded context payload routing.

---

## Quick Start & Local Deployment

### Runtime Prerequisites
* Runtime Environment: Python 3.10 or 3.11
* System Memory: 8GB RAM minimum (16GB recommended for heavy batch tokenizations)
* Local Infrastructure: Active local Ollama instance running the `llama3` model cluster

### Installation Sequence
1. Clone the repository and initialize the isolated runtime environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

2. Synchronize dependency requirements:
   pip install -r requirements.txt

3. Initialize environment variables from the distribution template:
   cp .env.example .env

### System Execution

#### 1. Launch the Core Backend Gateway Server
Start the asynchronous FastAPI server factory to listen for execution payloads on port 8000:
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

#### 2. Launch the Observability & Control Command Console
In an isolated terminal instance, run the frontend panel to map out live transaction logs and force-directed graph structures:
python -m streamlit run app/ui_dashboard.py

---

## System Configuration Specifications

The platform behavior is controlled via explicit variables located in your local `.env` configuration file:

# Network Routing Parameters
API_HOST=127.0.0.1
API_PORT=8000
API_WORKERS=4

# Persistent Storage Allocations
DATA_BASE_PATH=./data
CHROMA_DB_PATH=./data/chroma_db
BM25_INDEX_PATH=./data/bm25_index
GRAPH_STORAGE_PATH=./data/graph

# Token Partition Boundaries
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=128

# Vector Inference Model Architecture
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32

# Production Telemetry Options
LOG_LEVEL=INFO
LOG_FORMAT=json

---

## Programmatic Usage and Implementation Examples

### Standard Ingestion & Dual-Indexing Pipeline Execution

```python
import numpy as np
from app.services import (
    DocumentProcessor,
    ChunkingService,
    EmbeddingService,
    VectorStoreService,
    BM25Service,
    GraphService,
)

# 1. Initialize structural text parsers
processor = DocumentProcessor()
document = processor.process_document(file_path="architecture_spec.txt")

# 2. Segment processed text strings into overlapping semantic chunks
chunker = ChunkingService(chunk_size_tokens=512, chunk_overlap_tokens=128)
chunks = chunker.chunk_document(
    document_id=document.document_id,
    document_name=document.file_name,
    raw_text=document.raw_text,
)

# 3. Generate high-density geometric vector matrices locally
embedder = EmbeddingService()
embeddings = embedder.embed_texts([c.text for c in chunks], batch_size=32)

# 4. Commit embeddings and chunks to persistent ChromaDB collection
vector_store = VectorStoreService()
vector_store.create_collection(document_id=document.document_id)
vector_store.add_vectors(
    document_id=document.document_id,
    chunk_ids=[c.chunk_id for c in chunks],
    embeddings=embeddings.tolist(),
    documents=[c.text for c in chunks],
)

# 5. Extract structural entity-relation nodes into topological memory maps
graph_service = GraphService()
graph, nodes, edges = graph_service.extract_graph(
    document_id=document.document_id,
    document_name=document.file_name,
    text=document.raw_text,
)
graph_service.persist_graph(document.document_id)