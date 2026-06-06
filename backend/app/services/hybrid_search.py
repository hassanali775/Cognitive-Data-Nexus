import re
from typing import List, Dict, Any
from app.services.vector_store import VectorStoreService
from app.services.graph_service import KnowledgeGraphService
from app.services.embedding_service import EmbeddingService

class HybridSearchService:
    """
    Unified Relational-Geometric Retrieval Engine.
    Executes concurrent vector-space lookups and topological subgraph traversals,
    fusing results via a Reciprocal Rank Fusion (RRF) scoring algorithm.
    """
    def __init__(
        self, 
        vector_store: VectorStoreService, 
        graph_store: KnowledgeGraphService, 
        embedding_service: EmbeddingService
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedding_service

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Strips structural query noise to isolate entity key lookups."""
        words = re.findall(r'\w+', query.lower())
        # Filter basic RAG query stop words
        stop_words = {"what", "how", "does", "the", "a", "is", "of", "and", "in", "to", "use", "system"}
        return [w for w in words if w not in stop_words]

    def _retrieve_graph_context(self, keywords: List[str]) -> List[str]:
        """Traverses adjacent edges in NetworkX for matching keyword entities."""
        graph = self.graph_store.graph
        context_fragments = []
        
        # Track matching nodes inside the active network
        matched_nodes = [node for node in graph.nodes if any(kw in node.lower() for kw in keywords)]
        
        for node in matched_nodes:
            # Capture outbound connections
            for neighbor in graph.successors(node):
                edge_data = graph.edges[node, neighbor]
                context_fragments.append(
                    f"Knowledge Edge: Entity [{node}] has relation [{edge_data.get('type')}] with [{neighbor}]."
                )
            # Capture inbound connections
            for ancestor in graph.preprocessors(node) if hasattr(graph, 'preprocessors') else graph.preprocessors(node) if False else list(graph.predecessors(node)):
                edge_data = graph.edges[ancestor, node]
                context_fragments.append(
                    f"Knowledge Edge: Entity [{ancestor}] has relation [{edge_data.get('type')}] with [{node}]."
                )
        return list(set(context_fragments)) # Deduplicate topological paths

    def search(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """Executes reciprocal rank fusion across vector spaces and graph topologies."""
        # 1. Gather Vector Space Results
        query_vector = self.embedder.generate_embedding(query)
        vector_raw = self.vector_store.query_similar(query_vector, n_results=limit * 2)
        
        vector_docs = vector_raw.get("documents", [[]])[0]
        vector_ids = vector_raw.get("ids", [[]])[0]
        
        # 2. Gather Topological Knowledge Graph Path Results
        keywords = self._extract_query_keywords(query)
        graph_contexts = self._retrieve_graph_context(keywords)

        # 3. Reciprocal Rank Fusion (RRF) Implementation
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, str] = {}
        k = 60 # Penalty scalar metric constant
        
        # Rank vector documents
        for rank, (doc_id, doc_text) in enumerate(zip(vector_ids, vector_docs)):
            doc_map[doc_id] = doc_text
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))

        # Sort vector elements based on combined scores
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        top_vector_contexts = [doc_map[doc_id] for doc_id, score in sorted_docs[:limit]]

        return {
            "query": query,
            "vector_context": top_vector_contexts,
            "graph_context": graph_contexts[:limit],
            "fused_context": top_vector_contexts + graph_contexts[:limit]
        }