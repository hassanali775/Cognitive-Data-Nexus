from typing import Dict, Any, List, Optional
from datetime import datetime

class DocumentChunk:
    """
    Domain entity representing a parsed, processed, and embedded 
    slice of a corporate document with strict metadata tracking.
    """
    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        content: str,
        index: int,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        parent_id: Optional[str] = None,
        created_at: Optional[datetime] = None
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.index = index
        self.metadata = metadata or {}
        self.embedding = embedding
        self.parent_id = parent_id
        self.created_at = created_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the chunk model for storage in ChromaDB or JSON caches."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "index": self.index,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat()
        }

    def __repr__(self) -> str:
        return f"<DocumentChunk id={self.chunk_id} doc={self.document_id} idx={self.index}>"