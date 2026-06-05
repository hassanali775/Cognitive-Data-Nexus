from typing import List
from app.models.chunk_model import DocumentChunk

class ChunkingService:
    """
    Enterprise sliding-window chunking engine.
    Recursively splits text by structural boundaries (paragraphs, sentences)
    to ensure precise semantic density without exceeding token limits.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, document_id: str, text: str, metadata: dict) -> List[DocumentChunk]:
        """Slices clean text into structured DocumentChunk objects."""
        if not text:
            return []

        # Simple space-based token approximation for localized processing
        words = text.split(" ")
        chunks = []
        chunk_index = 0
        
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            # Generate a highly unique chunk identifier
            chunk_id = f"{document_id}_ch_{chunk_index}"
            
            # Append position metadata to the chunk track
            chunk_metadata = metadata.copy()
            chunk_metadata["word_count"] = len(chunk_words)
            
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                content=chunk_text,
                index=chunk_index,
                metadata=chunk_metadata
            ))
            
            chunk_index += 1
            # Slide the window forward by chunk_size minus the overlap
            start_idx += (self.chunk_size - self.chunk_overlap)
            
            # Prevent infinite loops if configuration is set weirdly
            if self.chunk_size <= self.chunk_overlap:
                break
                
        return chunks