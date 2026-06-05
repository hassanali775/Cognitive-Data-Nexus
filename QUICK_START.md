# Cognitive Data Nexus - Quick Start Guide

**Get up and running in 5 minutes.**

---

## ⚡ 30-Second Setup

```bash
# 1. Setup environment
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Verify installation
python verify_installation.py

# 3. Start server
python -m uvicorn app.main:app --reload
# Visit: http://localhost:8000/health
```

---

## 📖 5-Minute Example

```python
from app.services import (
    DocumentProcessor, ChunkingService, EmbeddingService,
    VectorStoreService, BM25Service, GraphService
)
from app.config import DocumentFormat

# Process a PDF
processor = DocumentProcessor()
doc = processor.process_document("document.pdf", DocumentFormat.PDF)
print(f"Extracted {doc.raw_text_length} characters")

# Split into chunks
chunker = ChunkingService()
chunks = chunker.chunk_document(doc.document_id, doc.file_name, raw_text)
print(f"Created {len(chunks)} chunks")

# Compute embeddings
embedder = EmbeddingService()
embeddings = embedder.embed_texts([c.text for c in chunks])
print(f"Embeddings shape: {embeddings.shape}")

# Store in vector DB
vector_store = VectorStoreService()
vector_store.create_collection(doc.document_id, doc.file_name)
vector_store.add_vectors(doc.document_id, [c.chunk_id for c in chunks],
                         embeddings.tolist(), [c.text for c in chunks],
                         [{"i": i} for i in range(len(chunks))])
print("Vectors stored ✓")

# Index with BM25
bm25 = BM25Service()
bm25.create_index(doc.document_id, [c.chunk_id for c in chunks],
                  [c.text for c in chunks])
bm25.persist_index(doc.document_id)
print("BM25 indexed ✓")

# Extract knowledge graph
graph_svc = GraphService()
graph, nodes, edges = graph_svc.extract_graph(doc.document_id,
                                               doc.file_name, raw_text)
graph_svc.persist_graph(doc.document_id)
print(f"Graph: {nodes} nodes, {edges} edges")

# Search
query = "machine learning algorithms"
query_emb = embedder.embed_text(query)
vec_ids, docs, scores, _ = vector_store.search_vectors(
    doc.document_id, query_emb.tolist(), top_k=3)
print(f"\nVector search results: {len(vec_ids)}")
for doc, score in zip(docs[:2], scores[:2]):
    print(f"  {score:.3f}: {doc[:50]}...")

bm25_results = bm25.search(doc.document_id, query, top_k=3)
print(f"\nBM25 results: {len(bm25_results)}")
```

---

## 🧪 Run Tests

```bash
# Integration test (shows pipeline)
pytest tests/integration/ -v -s

# All tests
pytest tests/ -v

# Specific test
pytest tests/integration/test_end_to_end_pipeline.py::TestEndToEndPipeline::test_full_pipeline -v
```

---

## 🛠️ Common Tasks

### Process a Document

```python
from app.services import DocumentProcessor
from app.config import DocumentFormat

processor = DocumentProcessor()
doc = processor.process_document("file.pdf", DocumentFormat.PDF)
# doc.document_id - unique ID
# doc.raw_text_length - characters extracted
```

### Create Chunks

```python
from app.services import ChunkingService

chunker = ChunkingService(
    chunk_size_tokens=512,      # Size of each chunk
    chunk_overlap_tokens=128     # Overlap between chunks
)
chunks = chunker.chunk_document(
    document_id=doc.document_id,
    document_name=doc.file_name,
    raw_text=raw_text
)
# chunks[0].text - chunk content
# chunks[0].metadata.chunk_index - position
```

### Compute Embeddings

```python
from app.services import EmbeddingService

embedder = EmbeddingService()
embeddings = embedder.embed_texts(
    texts=[c.text for c in chunks],
    batch_size=32,
    normalize=True
)
# Shape: (num_chunks, 384)
```

### Store & Search Vectors

```python
from app.services import VectorStoreService

store = VectorStoreService()
store.create_collection(doc_id, doc_name)
store.add_vectors(doc_id, chunk_ids, embeddings, documents, metadatas)

# Search
query_embedding = embedder.embed_text("what is X?")
ids, docs, scores, metadata = store.search_vectors(
    document_id=doc_id,
    query_embedding=query_embedding.tolist(),
    top_k=5,
    threshold=0.3
)
```

### Index & Search Keywords

```python
from app.services import BM25Service

bm25 = BM25Service()
bm25.create_index(doc_id, chunk_ids, chunk_texts)
bm25.persist_index(doc_id)

# Search
results = bm25.search(
    document_id=doc_id,
    query="search term",
    top_k=5,
    min_score=0.0
)
# results: [(chunk_id, text, score), ...]
```

### Extract Knowledge Graph

```python
from app.services import GraphService

graph_svc = GraphService()
graph, node_count, edge_count = graph_svc.extract_graph(
    document_id=doc_id,
    document_name=doc_name,
    text=raw_text
)
graph_svc.persist_graph(doc_id)

stats = graph_svc.get_graph_stats(doc_id)
# stats['entity_type_distribution']
# stats['relation_type_distribution']
```

---

## 📝 Configuration

Edit `.env`:

```bash
# Chunking
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=128

# Embeddings
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=cpu  # cpu, cuda, or mps

# Logging
LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json       # json or text

# Server
API_PORT=8000
API_WORKERS=4
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | First run downloads 400MB. Check internet & disk space. |
| Out of memory | Reduce `EMBEDDING_BATCH_SIZE` to 8 or 4 in `.env` |
| Port already in use | Change `API_PORT` in `.env` or kill existing process |
| ChromaDB locked | Delete `data/chroma_db/.lock` and restart |
| Empty document error | Document must have extractable text |
| BM25 index corrupted | Delete `data/bm25_index/` and re-index |

---

## 📊 Performance Tips

1. **Batch embeddings**: Process 10+ texts at once (not one-by-one)
2. **Adjust batch size**: Smaller batches (8-16) if OOM errors
3. **Reuse embeddings**: Store computed embeddings, don't recompute
4. **Use GPU**: Set `EMBEDDING_DEVICE=cuda` for 3-5x speedup
5. **Index once**: Persist indexes after creation

---

## 📦 Supported Formats

| Format | Parser | Notes |
|--------|--------|-------|
| PDF | pypdf | Extracts text from all pages |
| DOCX | python-docx | Includes tables |
| Markdown | Built-in | UTF-8, Latin-1, CP1252 |
| TXT | Built-in | Auto-detect encoding |

---

## 🔍 Debug Mode

Enable detailed logging:

```python
from app.logging_config import setup_logging
setup_logging(log_level="DEBUG", log_format="json")

# Or set in .env:
LOG_LEVEL=DEBUG
```

Check logs:
```bash
tail -f logs/app.log  # If LOG_FILE is set
```

---

## 🚀 Production Checklist

- [ ] Python 3.11+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Verification passed (`python verify_installation.py`)
- [ ] Tests pass (`pytest tests/ -v`)
- [ ] `.env` configured for your setup
- [ ] `data/` directory has write access
- [ ] Enough disk space (~5GB for embeddings + vectors)

---

## 📚 Full Documentation

- **README.md** - Complete setup & usage guide
- **PHASE_1_COMPLETE.md** - Detailed implementation
- **FINAL_SUMMARY.md** - Architecture overview

---

## 💡 Pro Tips

1. **First load is slow**: Embedding model downloads on first use
2. **Batch processing**: Process multiple documents together
3. **Persist indexes**: Save BM25 and graphs after creation
4. **Memory management**: Load indexes as needed, not all at once
5. **Error handling**: Check exception context for debugging

---

## 🎯 Next Steps

1. Run verification: `python verify_installation.py`
2. Start server: `python -m uvicorn app.main:app --reload`
3. Run integration test: `pytest tests/integration/ -v`
4. Process your first document (see 5-minute example above)
5. Explore Phase 2: API routes, hybrid search, chat

---

**Ready to go!** 🚀

For detailed docs, see `backend/README.md`
