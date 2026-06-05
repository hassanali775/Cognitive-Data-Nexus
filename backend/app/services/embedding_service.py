import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    """
    Localized Vector Embedding Service.
    Runs entirely air-gapped on the local CPU/GPU using sentence-transformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # This initializes the local model pipeline cleanly
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a single text chunk."""
        if not text:
            # Return zero-vector if text is empty to prevent vector database crashes
            return [0.0] * 384
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """High-performance batch vector generation for massive documents."""
        if not texts:
            return []
            
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()