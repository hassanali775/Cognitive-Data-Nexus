import asyncio
import os
from app.services.document_processor import DocumentProcessorService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.graph_service import KnowledgeGraphService
from app.services.extraction_service import RuleBasedGraphExtractor

async def main():
    print("⚡ Starting Cognitive-Data-Nexus Phase 2 E2E Verification...")
    
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

    # Initialize complete localized architectural stack (Vector + Graph)
    doc_processor = DocumentProcessorService()
    chunker = ChunkingService(chunk_size=15, chunk_overlap=3)
    embedder = EmbeddingService()
    vector_db = VectorStoreService()
    graph_db = KnowledgeGraphService()
    extractor = RuleBasedGraphExtractor()

    try:
        # 1. Parse and Cleaning
        print("➡️  Executing Document Extractor Engine...")
        clean_text = await doc_processor.extract_text(test_file_path)
        
        # 2. Chunk Ingestion
        print("➡️  Running Recursive Sliding-Window Chunker...")
        chunks = chunker.create_chunks(document_id="doc_test_002", text=clean_text, metadata={"source": "graph_e2e"})
        print(f"📊 Generated {len(chunks)} structural context chunks.")

        # 3. Embedding Generation & Vector Injection
        print("➡️  Encoding and storing vector data in ChromaDB...")
        raw_texts = [c.content for c in chunks]
        embeddings = embedder.generate_embeddings_batch(raw_texts)
        vector_db.upsert_chunks(chunks, embeddings)

        # 4. Phase 2 Innovation: Knowledge Graph Extraction
        print("➡️  Processing structural semantic graph layers via NetworkX...")
        for chunk in chunks:
            nodes, relationships = extractor.extract_from_text(chunk.content)
            
            for node in nodes:
                graph_db.add_entity(node)
            for rel in relationships:
                graph_db.add_relationship(rel)
                
        graph_db.save_graph()
        print(f"✅ Knowledge Graph generated! Saved to: {graph_db.storage_path}")
        print(f"📈 Graph Metrics: {len(graph_db.graph.nodes)} unique nodes, {len(graph_db.graph.edges)} directed edges.")

        # Print out the extracted relations to verify visually
        print("\n🌐 Extracted Knowledge Graph Edge Connections:")
        for u, v, data in graph_db.graph.edges(data=True):
            print(f"   [{u}] --({data['type']})--> [{v}]")

        print("\n🎉 PHASE 2 INTEGRATION: KNOWLEDGE GRAPH INGESTION SUCCESSFUL!")

    finally:
        # Clean up text asset footprint
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

if __name__ == "__main__":
    asyncio.run(main())