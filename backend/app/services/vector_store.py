import os
import chromadb
from typing import List, Dict, Any
from app.models.chunk_model import DocumentChunk

class VectorStoreService:
    """
    Local Vector Database Controller.
    Manages client connections, schema collections, and persistence for ChromaDB.
    """
    def __init__(self, storage_path: str = "./chroma_db"):
        # Ensure path resolution is absolute and clean
        self.storage_path = os.path.abspath(storage_path)
        self.client = chromadb.PersistentClient(path=self.storage_path)
        # Initialize or fetch our master data collection
        self.collection = self.client.get_or_create_collection(
            name="cognitive_nexus_chunks"
        )

    def upsert_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Persists text chunks and pre-generated embeddings to the local disk."""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        
        # Flatten complex dictionary metadata strings for strict ChromaDB validation compatibility
        metadatas = []
        for c in chunks:
            meta = {
                "document_id": c.document_id,
                "chunk_index": c.index,
                "word_count": c.metadata.get("word_count", 0)
            }
            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query_similar(self, query_embedding: List[float], n_results: int = 3) -> Dict[str, Any]:
        """Queries local vector space using cosine distance defaults."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )