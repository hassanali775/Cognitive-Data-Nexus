"""
Cognitive Data Nexus - FastAPI Application Factory
Phase 1: Ingestion, Chunking, Parsing

Entry point for the REST API server.
Initializes services, middleware, and routes.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import logging

from app.config import get_settings, Environment
from app.logging_config import setup_logging, get_logger
from app.exceptions import CognitiveNexusBaseException

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan (startup/shutdown).
    
    Phase 1: Initialize embeddings model on startup.
    Phase 2: Will add vector store and graph initialization.
    """
    settings = get_settings()
    
    # ========================================================================
    # STARTUP
    # ========================================================================
    
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
    
    # Phase 1: Initialize embedding model (lazy load on first use in service)
    # Phase 2: Will initialize vector store and graph databases
    
    logger.info("Application startup complete")
    
    yield  # Application runs here
    
    # ========================================================================
    # SHUTDOWN
    # ========================================================================
    
    logger.info("Application shutting down")
    
    # Phase 1: Cleanup is minimal (models are garbage collected)
    # Phase 2: Will add vector store and graph cleanup
    
    logger.info(
        "Application shutdown complete",
        extra={"event": "app_shutdown"}
    )


# ============================================================================
# APPLICATION FACTORY
# ============================================================================


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI instance
    """
    settings = get_settings()
    
    # ========================================================================
    # CREATE APP
    # ========================================================================
    
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
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["localhost", "127.0.0.1"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted hosts middleware
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1"],
        )
    
    # ========================================================================
    # EXCEPTION HANDLERS
    # ========================================================================
    
    @app.exception_handler(CognitiveNexusBaseException)
    async def cognitive_nexus_exception_handler(
        request: Request,
        exc: CognitiveNexusBaseException,
    ):
        """Handle application-specific exceptions."""
        logger.warning(
            f"Application exception: {exc.error_code.value}",
            extra={
                "error_code": exc.error_code.value,
                "message": exc.message,
                "context": exc.context,
                "path": request.url.path,
                "method": request.method,
            }
        )
        
        return JSONResponse(
            status_code=400,  # Phase 2: Use appropriate status codes per error
            content={
                "error": exc.message,
                "error_code": exc.error_code.value,
                "context": exc.context,
            },
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(
            f"Unexpected exception: {type(exc).__name__}",
            extra={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "path": request.url.path,
                "method": request.method,
            },
            exc_info=True,
        )
        
        if settings.debug:
            # Return detailed error in debug mode
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
        else:
            # Return generic error in production
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "error_code": "INTERNAL_001",
                },
            )
    
    # ========================================================================
    # HEALTH CHECK ROUTE
    # ========================================================================
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "app_name": settings.app_name,
            "app_version": settings.app_version,
        }
    
    # ========================================================================
    # CONFIG ROUTE (Phase 1 - info only)
    # ========================================================================
    
    @app.get("/config")
    async def get_config():
        """Get application configuration (non-sensitive fields)."""
        return settings.to_dict()
    
    # ========================================================================
    # API ROUTES (Phase 2 will add document, chunking, embedding routes)
    # ========================================================================
    
    # Phase 2: Add routes from app.routes
    # - POST /ingest - Upload and ingest document
    # - GET /documents - List documents
    # - GET /documents/{doc_id} - Get document details
    # - DELETE /documents/{doc_id} - Delete document
    # - POST /search - Execute search
    # etc.
    
    logger.info(
        "Application created successfully",
        extra={
            "routes_count": len(app.routes),
            "middleware_count": len(app.user_middleware),
        }
    )
    
    return app


# ============================================================================
# APPLICATION INSTANCE
# ============================================================================

app = create_app()


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
        log_config=None,  # Use our custom logging
    )
