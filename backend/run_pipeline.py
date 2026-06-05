import asyncio
import os
from app.services.document_processor import DocumentProcessorService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

async def main():
    print("⚡ Starting Cognitive-Data-Nexus Phase 1 E2E Verification...")
    
    # Setup test file footprint
    test_file_path = "./sample_insight.txt"
    sample_data = """
    Cognitive Data Nexus Enterprise Architecture.
    The foundational engine utilizes an air-gapped Sentence-Transformer model for multi-format text ingestion.
    Vector persistence handles geometric distances locally within a customized ChromaDB collection matrix.
    Graph augmentations map cross-document entity entities using unified NetworkX layers.
    """
    
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(sample_data.strip())
        
    print(f"✅ Created mock text payload asset at: {test_file_path}")

    # Initialize complete localized architectural stack
    doc_processor = DocumentProcessorService()
    chunker = ChunkingService(chunk_size=15, chunk_overlap=3) # Small bounds explicitly for test tracking
    embedder = EmbeddingService()
    vector_db = VectorStoreService()

    try:
        # 1. Parse Phase
        print("➡️  Executing Document Extractor Engine...")
        clean_text = await doc_processor.extract_text(test_file_path)
        
        # 2. Chunk Phase
        print("➡️  Running Recursive Sliding-Window Chunker...")
        chunks = chunker.create_chunks(document_id="doc_test_001", text=clean_text, metadata={"source": "mock_e2e"})
        print(f"📊 Generated {len(chunks)} structural context chunks.")

        # 3. Embedding Vector Generation
        print("➡️  Encoding Local Text Vectors via Sentence-Transformers (all-MiniLM-L6-v2)...")
        raw_texts = [c.content for c in chunks]
        embeddings = embedder.generate_embeddings_batch(raw_texts)
        
        # 4. Storage Injection
        print("➡️  Injecting context matrices to local ChromaDB collections...")
        vector_db.upsert_chunks(chunks, embeddings)
        print("✅ Storage persistence locked down successfully.")

        # 5. Pipeline RAG Retrieval Query Test
        print("\n🔎 Testing Vector Query Pipeline...")
        test_query = "What database does the architecture use?"
        query_vector = embedder.generate_embedding(test_query)
        search_results = vector_db.query_similar(query_vector, n_results=1)
        
        print("\n🎯 E2E Pipeline Search Return Match:")
        print(f" > Retracted Document Chunk Match: {search_results['documents'][0][0]}")
        print("\n🎉 PHASE 1 INTEGRATION: 100% SUCCESSFUL COMPLETION!")

    finally:
        # Clean up test artifact footprint
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

if __name__ == "__main__":
    asyncio.run(main())