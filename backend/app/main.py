"""
Cognitive Data Nexus - FastAPI Application Factory
Phase 5 Integrated: Multi-Dimensional GraphRAG & Inference Backend

Entry point for the REST API server.
Initializes services, middleware, and routes.
"""

import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

from app.config import get_settings, Environment
from app.logging_config import setup_logging, get_logger
from app.exceptions import CognitiveNexusBaseException

# Import our verified modular Phase 1-4 architecture services
from app.services.document_processor import DocumentProcessorService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.graph_service import KnowledgeGraphService
from app.services.extraction_service import RuleBasedGraphExtractor
from app.services.hybrid_search import HybridSearchService
from app.services.llm_service import LocalLLMService

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# ============================================================================
# PYDANTIC SCHEMAS FOR API RETRIEVAL
# ============================================================================

class QueryRequest(BaseModel):
    prompt: str
    limit: int = 2

# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup/shutdown)."""
    settings = get_settings()
    
    logger.info(
        "Application starting",
        extra={
            "event": "app_startup",
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "app_env": settings.app_env.value,
            "debug": settings.debug,
        }
    )
    
    logger.info("Application startup complete")
    yield  
    logger.info("Application shutting down")
    logger.info("Application shutdown complete", extra={"event": "app_shutdown"})


# ============================================================================
# APPLICATION FACTORY
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application with fused GraphRAG endpoints."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="Local document intelligence and graph-augmented RAG pipeline",
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )
    
    # ========================================================================
    # MIDDLEWARE
    # ========================================================================
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["localhost", "127.0.0.1"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1"],
        )
    
    # ========================================================================
    # SERVICE INSTANTIATION
    # ========================================================================
    
    UPLOAD_DIR = "./temp_uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    doc_processor = DocumentProcessorService()
    chunker = ChunkingService(chunk_size=15, chunk_overlap=3)
    embedder = EmbeddingService()
    vector_db = VectorStoreService()
    graph_db = KnowledgeGraphService()
    extractor = RuleBasedGraphExtractor()
    
    hybrid_searcher = HybridSearchService(
        vector_store=vector_db, 
        graph_store=graph_db, 
        embedding_service=embedder
    )
    local_brain = LocalLLMService()
    
    # ========================================================================
    # EXCEPTION HANDLERS
    # ========================================================================
    
    @app.exception_handler(CognitiveNexusBaseException)
    async def cognitive_nexus_exception_handler(request: Request, exc: CognitiveNexusBaseException):
        logger.warning(
            f"Application exception: {exc.error_code.value}",
            extra={
                "error_code": exc.error_code.value,
                "message": exc.message,
                "path": request.url.path,
            }
        )
        return JSONResponse(
            status_code=400,
            content={"error": exc.message, "error_code": exc.error_code.value},
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unexpected exception: {type(exc).__name__}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": str(exc) if settings.debug else "INTERNAL_001"},
        )
    
    # ========================================================================
    # CORE SYSTEM ROUTES
    # ========================================================================
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "app_name": settings.app_name, "app_version": settings.app_version}
    
    @app.get("/config")
    async def get_config():
        return settings.to_dict()
    
    # ========================================================================
    # FUSED PRODUCTION GRAPHRAG ENDPOINTS
    # ========================================================================
    
    @app.post("/api/ingest")
    async def ingest_document(file: UploadFile = File(...)):
        """Asynchronously parses, chunks, embeds, and indexes an uploaded document asset."""
        temp_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            clean_text = await doc_processor.extract_text(temp_path)
            if not clean_text:
                raise HTTPException(status_code=400, detail="Document contains no readable text stream.")

            doc_id = f"doc_{int(os.path.getmtime(temp_path))}"
            chunks = chunker.create_chunks(document_id=doc_id, text=clean_text, metadata={"filename": file.filename})

            raw_texts = [c.content for c in chunks]
            embeddings = embedder.generate_embeddings_batch(raw_texts)
            vector_db.upsert_chunks(chunks, embeddings)

            for chunk in chunks:
                nodes, relationships = extractor.extract_from_text(chunk.content)
                for node in nodes:
                    graph_db.add_entity(node)
                for rel in relationships:
                    graph_db.add_relationship(rel)
            graph_db.save_graph()

            logger.info(f"Successfully ingested file: {file.filename}, chunks: {len(chunks)}")
            return {
                "status": "success",
                "document_id": doc_id,
                "chunks_processed": len(chunks),
                "graph_metrics": {
                    "nodes": len(graph_db.graph.nodes),
                    "edges": len(graph_db.graph.edges)
                }
            }
        except Exception as e:
            logger.error(f"Ingestion Pipeline Faulted: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pipeline Ingestion Crash: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @app.post("/api/query")
    async def query_hybrid_engine(request: QueryRequest):
        """Executes a dual-space RRF search and passes context vectors to local Llama3 inference."""
        try:
            search_payload = hybrid_searcher.search(query=request.prompt, limit=request.limit)
            
            answer = local_brain.generate_answer(
                query=request.prompt,
                vector_context=search_payload["vector_context"],
                graph_context=search_payload["graph_context"]
            )
            
            return {
                "query": request.prompt,
                "answer": answer,
                "vector_context": search_payload["vector_context"],
                "graph_context": search_payload["graph_context"]
            }
        except Exception as e:
            logger.error(f"Query Routing Faulted: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Inference Routing Crash: {str(e)}")

    @app.get("/api/graph")
    def get_graph_data():
        """Returns the current raw state of the knowledge network for frontend visualization."""
        return {
            "nodes": [{"id": n, "type": d.get("type")} for n, d in graph_db.graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, "type": d.get("type")} for u, v, d in graph_db.graph.edges(data=True)]
        }
    
    logger.info(
        "Application created successfully with GraphRAG Extensions",
        extra={"routes_count": len(app.routes), "middleware_count": len(app.user_middleware)}
    )
    
    return app

# Instantiate application
app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
        log_config=None,
    )