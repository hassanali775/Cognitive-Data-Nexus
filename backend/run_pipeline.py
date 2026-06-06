import asyncio
import os
from app.services.document_processor import DocumentProcessorService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.graph_service import KnowledgeGraphService
from app.services.extraction_service import RuleBasedGraphExtractor
from app.services.hybrid_search import HybridSearchService
from app.services.llm_service import LocalLLMService

async def main():
    print("⚡ Executing Full Cognitive-Data-Nexus Production Pipeline Ingestion...")
    
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
        
    print(f"✅ Document Payload Staged: {test_file_path}")

    # Initialize complete localized architectural stack
    doc_processor = DocumentProcessorService()
    chunker = ChunkingService(chunk_size=15, chunk_overlap=3)
    embedder = EmbeddingService()
    vector_db = VectorStoreService()
    graph_db = KnowledgeGraphService()
    extractor = RuleBasedGraphExtractor()
    
    # Advanced Phase 3 & 4 Services
    hybrid_searcher = HybridSearchService(vector_store=vector_db, graph_store=graph_db, embedding_service=embedder)
    local_brain = LocalLLMService()

    try:
        # Ingestion Phases
        print("➡️  Parsing raw content matrices...")
        clean_text = await doc_processor.extract_text(test_file_path)
        
        print("➡️  Chunking and compiling semantic sliding windows...")
        chunks = chunker.create_chunks(document_id="doc_prod_001", text=clean_text, metadata={"source": "production"})

        print("➡️  Generating dense embedding vectors & synchronizing ChromaDB...")
        raw_texts = [c.content for c in chunks]
        embeddings = embedder.generate_embeddings_batch(raw_texts)
        vector_db.upsert_chunks(chunks, embeddings)

        print("➡️  Analyzing semantic connections & updating NetworkX Graph Topology...")
        for chunk in chunks:
            nodes, relationships = extractor.extract_from_text(chunk.content)
            for node in nodes:
                graph_db.add_entity(node)
            for rel in relationships:
                graph_db.add_relationship(rel)
        graph_db.save_graph()

        # Hybrid Context Synthesis
        target_query = "Explain how the core pipeline interacts with ChromaDB based on the architecture rules."
        print(f"\n🔎 Querying Hybrid Fusion Layer for: '{target_query}'")
        search_payload = hybrid_searcher.search(query=target_query, limit=2)

        # Local Generation Phase
        print("🤖 Invoking Localized Llama3 Inference Engine via Ollama...")
        final_response = local_brain.generate_answer(
            query=target_query,
            vector_context=search_payload["vector_context"],
            graph_context=search_payload["graph_context"]
        )

        print("\n======================= 🪐 COGNITIVE DATA NEXUS ANSWER =======================")
        print(final_response)
        print("==============================================================================")
        print("\n🎉 BACKEND ENGINE INTEGRATION: 100% COMPLETE AND SUCCESSFUL!")

    finally:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

if __name__ == "__main__":
    asyncio.run(main())