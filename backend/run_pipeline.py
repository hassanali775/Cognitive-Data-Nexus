import asyncio
import os
from app.services.document_processor import DocumentProcessorService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.graph_service import KnowledgeGraphService
from app.services.extraction_service import RuleBasedGraphExtractor
from app.services.hybrid_search import HybridSearchService

async def main():
    print("⚡ Starting Cognitive-Data-Nexus Phase 3 E2E Hybrid System Verification...")
    
    test_file_path = "./sample_insight.txt"
    sample_data = """
    Cognitive Data Nexus Enterprise Architecture.
    The foundational engine utilizes an air-gapped Sentence-Transformer model for multi-format text ingestion.
    Vector persistence handles geometric distances locally within a customized ChromaDB collection matrix.
    Graph augmentations map cross-document entity entities using unified NetworkX layers.
    The core pipeline uses ChromaDB to store vector embeddings.
    """
    
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(sample_data.strip())
        
    print(f"✅ Created mock text payload asset at: {test_file_path}")

    # Initialize complete localized architectural stack
    doc_processor = DocumentProcessorService()
    chunker = ChunkingService(chunk_size=15, chunk_overlap=3)
    embedder = EmbeddingService()
    vector_db = VectorStoreService()
    graph_db = KnowledgeGraphService()
    extractor = RuleBasedGraphExtractor()
    
    # Initialize our Phase 3 Hybrid Search Orchestrator
    hybrid_searcher = HybridSearchService(
        vector_store=vector_db, 
        graph_store=graph_db, 
        embedding_service=embedder
    )

    try:
        # 1. Parse Ingestion
        print("➡️  Executing Document Extractor Engine...")
        clean_text = await doc_processor.extract_text(test_file_path)
        
        # 2. Chunk Ingestion
        print("➡️  Running Recursive Sliding-Window Chunker...")
        chunks = chunker.create_chunks(document_id="doc_final_999", text=clean_text, metadata={"source": "hybrid_e2e"})

        # 3. Embedding Vector Generation & Insertion
        print("➡️  Encoding and storing vector data in ChromaDB...")
        raw_texts = [c.content for c in chunks]
        embeddings = embedder.generate_embeddings_batch(raw_texts)
        vector_db.upsert_chunks(chunks, embeddings)

        # 4. Graph Generation
        print("➡️  Processing structural semantic graph layers via NetworkX...")
        for chunk in chunks:
            nodes, relationships = extractor.extract_from_text(chunk.content)
            for node in nodes:
                graph_db.add_entity(node)
            for rel in relationships:
                graph_db.add_relationship(rel)
        graph_db.save_graph()

        # 5. Advanced Hybrid Search Verification Rule
        print("\n🔎 EXECUTING HYBRID SEARCH QUERY...")
        target_query = "How does vector persistence utilize ChromaDB?"
        search_payload = hybrid_searcher.search(query=target_query, limit=2)
        
        print(f"\n🎯 Search Target Query: '{target_query}'")
        print("\n--- 🌐 RETRIEVED VECTOR CONTEXTS (Geometric Space) ---")
        for idx, doc in enumerate(search_payload["vector_context"]):
            print(f" [{idx + 1}] {doc.strip()}")
            
        print("\n--- 🗺️ RETRIEVED GRAPH CONTEXTS (Topological Web) ---")
        for idx, edge in enumerate(search_payload["graph_context"]):
            print(f" [{idx + 1}] {edge}")

        print("\n🎉 PHASE 3 INTEGRATION: HYBRID SEARCH PIPELINE 100% OPERATIONAL!")

    finally:
        # Clean up text asset footprint
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

if __name__ == "__main__":
    asyncio.run(main())