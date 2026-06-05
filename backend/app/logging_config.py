"""
Cognitive Data Nexus - Logging Configuration
Phase 1: Ingestion, Chunking, Parsing

Structured logging setup with JSON support, file rotation, and metrics.
All logs are machine-readable for monitoring and debugging.
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from pythonjsonlogger import jsonlogger

from app.config import get_settings, LogLevel


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional context fields."""
    
    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record["timestamp"] = datetime.utcnow().isoformat()
        
        # Add module and function information
        log_record["module"] = record.name
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno
        
        # Add severity level
        log_record["level"] = record.levelname
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for console output."""
    
    FORMAT = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as readable text."""
        formatter = logging.Formatter(self.FORMAT, datefmt=self.DATE_FORMAT)
        return formatter.format(record)


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json or text)
        log_file: Optional log file path (with rotation)
    """
    settings = get_settings()
    
    # Use provided params or fall back to settings
    level = log_level or settings.log_level.value
    format_type = log_format or settings.log_format
    log_path = log_file or settings.log_file
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # ========================================================================
    # CONSOLE HANDLER
    # ========================================================================
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    
    if format_type == "json":
        console_formatter = StructuredFormatter()
    else:
        console_formatter = TextFormatter()
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # ========================================================================
    # FILE HANDLER (Optional)
    # ========================================================================
    
    if log_path:
        log_file_path = Path(log_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, level))
        
        # Always use JSON for file logging (machine-readable)
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # ========================================================================
    # CONFIGURE THIRD-PARTY LOGGERS
    # ========================================================================
    
    # Reduce verbosity of third-party libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    root_logger.info(
        "Logging configured",
        extra={
            "log_level": level,
            "log_format": format_type,
            "log_file": log_path,
            "app_name": settings.app_name,
            "app_version": settings.app_version,
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# ============================================================================
# PERFORMANCE LOGGING CONTEXT MANAGER
# ============================================================================


class PerformanceLogger:
    """Context manager for logging operation performance."""
    
    def __init__(self, logger: logging.Logger, operation: str, **extra_context):
        """
        Initialize performance logger.
        
        Args:
            logger: Logger instance
            operation: Name of operation being timed
            **extra_context: Additional context fields
        """
        self.logger = logger
        self.operation = operation
        self.extra_context = extra_context
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log performance metrics."""
        import time
        elapsed_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            self.logger.info(
                f"{self.operation} completed",
                extra={
                    "operation": self.operation,
                    "duration_ms": round(elapsed_ms, 2),
                    **self.extra_context,
                }
            )
        else:
            self.logger.error(
                f"{self.operation} failed",
                extra={
                    "operation": self.operation,
                    "duration_ms": round(elapsed_ms, 2),
                    "error_type": exc_type.__name__,
                    **self.extra_context,
                },
                exc_info=True,
            )
        
        return False  # Re-raise exception


# ============================================================================
# EXCEPTION LOGGING UTILITY
# ============================================================================


def log_exception(
    logger: logging.Logger,
    exc: Exception,
    context: Optional[dict] = None,
    level: str = "error",
) -> None:
    """
    Log an exception with structured context.
    
    Args:
        logger: Logger instance
        exc: Exception to log
        context: Additional context information
        level: Log level (error, warning, critical)
    """
    log_method = getattr(logger, level.lower())
    
    extra_data = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }
    
    if context:
        extra_data["context"] = context
    
    log_method(
        f"Exception: {type(exc).__name__}",
        extra=extra_data,
        exc_info=True,
    )


# ============================================================================
# METRICS LOGGING
# ============================================================================


class MetricsLogger:
    """Logger for application metrics."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize metrics logger."""
        self.logger = logger
    
    def log_document_processed(
        self,
        document_id: str,
        file_name: str,
        raw_text_length: int,
        processing_time_ms: float,
    ) -> None:
        """Log document processing metrics."""
        self.logger.info(
            "Document processed",
            extra={
                "event": "document_processed",
                "document_id": document_id,
                "file_name": file_name,
                "raw_text_length": raw_text_length,
                "processing_time_ms": round(processing_time_ms, 2),
            }
        )
    
    def log_chunks_created(
        self,
        document_id: str,
        chunk_count: int,
        total_tokens: int,
        processing_time_ms: float,
    ) -> None:
        """Log chunk creation metrics."""
        self.logger.info(
            "Chunks created",
            extra={
                "event": "chunks_created",
                "document_id": document_id,
                "chunk_count": chunk_count,
                "total_tokens": total_tokens,
                "processing_time_ms": round(processing_time_ms, 2),
                "avg_tokens_per_chunk": round(total_tokens / chunk_count, 1),
            }
        )
    
    def log_embeddings_computed(
        self,
        chunk_count: int,
        embedding_model: str,
        inference_time_ms: float,
        batch_size: int,
    ) -> None:
        """Log embedding computation metrics."""
        self.logger.info(
            "Embeddings computed",
            extra={
                "event": "embeddings_computed",
                "chunk_count": chunk_count,
                "embedding_model": embedding_model,
                "inference_time_ms": round(inference_time_ms, 2),
                "batch_size": batch_size,
                "avg_time_per_chunk_ms": round(inference_time_ms / chunk_count, 2),
            }
        )
    
    def log_search_query(
        self,
        query: str,
        search_method: str,
        result_count: int,
        search_time_ms: float,
    ) -> None:
        """Log search query metrics."""
        self.logger.info(
            "Search query executed",
            extra={
                "event": "search_query",
                "query_length": len(query),
                "search_method": search_method,
                "result_count": result_count,
                "search_time_ms": round(search_time_ms, 2),
            }
        )
    
    def log_graph_extracted(
        self,
        document_id: str,
        node_count: int,
        edge_count: int,
        extraction_time_ms: float,
    ) -> None:
        """Log graph extraction metrics."""
        self.logger.info(
            "Graph extracted",
            extra={
                "event": "graph_extracted",
                "document_id": document_id,
                "node_count": node_count,
                "edge_count": edge_count,
                "extraction_time_ms": round(extraction_time_ms, 2),
                "avg_edges_per_node": round(edge_count / node_count, 2) if node_count > 0 else 0,
            }
        )
