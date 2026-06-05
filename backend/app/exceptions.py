"""
Cognitive Data Nexus - Custom Exception Hierarchy
Phase 1: Ingestion, Chunking, Parsing

Provides fine-grained error boundaries for all service layers.
All exceptions inherit from CognitiveNexusBaseException for centralized handling.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(str, Enum):
    """Standardized error codes for client-facing responses."""
    
    # Document Processing
    DOCUMENT_PARSE_ERROR = "DOC_001"
    UNSUPPORTED_FORMAT = "DOC_002"
    FILE_CORRUPTED = "DOC_003"
    FILE_TOO_LARGE = "DOC_004"
    FILE_NOT_FOUND = "DOC_005"
    
    # Chunking
    CHUNKING_FAILED = "CHUNK_001"
    INVALID_CHUNK_PARAMS = "CHUNK_002"
    EMPTY_DOCUMENT = "CHUNK_003"
    
    # Embedding
    EMBEDDING_MODEL_LOAD_FAILED = "EMB_001"
    EMBEDDING_INFERENCE_FAILED = "EMB_002"
    EMBEDDING_OUT_OF_MEMORY = "EMB_003"
    INVALID_TEXT_ENCODING = "EMB_004"
    
    # Vector Store
    VECTOR_STORE_INIT_FAILED = "VS_001"
    VECTOR_STORE_WRITE_FAILED = "VS_002"
    VECTOR_STORE_READ_FAILED = "VS_003"
    VECTOR_STORE_CORRUPTED = "VS_004"
    COLLECTION_NOT_FOUND = "VS_005"
    
    # BM25 Service
    BM25_INDEX_FAILED = "BM25_001"
    BM25_SEARCH_FAILED = "BM25_002"
    BM25_CORRUPT_INDEX = "BM25_003"
    
    # Graph Service
    GRAPH_EXTRACTION_FAILED = "GRAPH_001"
    GRAPH_PERSISTENCE_FAILED = "GRAPH_002"
    GRAPH_LOAD_FAILED = "GRAPH_003"
    
    # General
    INTERNAL_SERVER_ERROR = "INTERNAL_001"
    INVALID_REQUEST = "INVALID_001"
    RESOURCE_CONFLICT = "CONFLICT_001"


class CognitiveNexusBaseException(Exception):
    """
    Base exception for all Cognitive Data Nexus errors.
    
    Provides structured error context for logging, debugging, and client responses.
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.original_exception = original_exception
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize exception to dictionary for API responses."""
        return {
            "error": self.message,
            "error_code": self.error_code.value,
            "context": self.context,
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"code={self.error_code.value}, "
            f"message={self.message}, "
            f"context={self.context})"
        )


# ============================================================================
# DOCUMENT PROCESSING EXCEPTIONS
# ============================================================================


class DocumentProcessingException(CognitiveNexusBaseException):
    """Base exception for document parsing failures."""
    pass


class DocumentParseException(DocumentProcessingException):
    """Raised when document parsing fails (invalid format, corruption, etc.)."""
    
    def __init__(
        self,
        message: str,
        file_name: str,
        format_type: str,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.DOCUMENT_PARSE_ERROR,
            context={
                "file_name": file_name,
                "format_type": format_type,
            },
            original_exception=original_exception,
        )


class UnsupportedFormatException(DocumentProcessingException):
    """Raised when document format is not supported."""
    
    def __init__(self, file_name: str, format_type: str, supported_formats: list):
        super().__init__(
            message=f"File format '{format_type}' is not supported",
            error_code=ErrorCode.UNSUPPORTED_FORMAT,
            context={
                "file_name": file_name,
                "format_type": format_type,
                "supported_formats": supported_formats,
            },
        )


class FileCorruptedException(DocumentProcessingException):
    """Raised when file is corrupted or unreadable."""
    
    def __init__(self, file_name: str, reason: str):
        super().__init__(
            message=f"File is corrupted or unreadable: {reason}",
            error_code=ErrorCode.FILE_CORRUPTED,
            context={
                "file_name": file_name,
                "reason": reason,
            },
        )


class FileTooLargeException(DocumentProcessingException):
    """Raised when file exceeds maximum size."""
    
    def __init__(self, file_name: str, file_size_mb: float, max_size_mb: float):
        super().__init__(
            message=f"File size ({file_size_mb:.2f}MB) exceeds limit ({max_size_mb:.2f}MB)",
            error_code=ErrorCode.FILE_TOO_LARGE,
            context={
                "file_name": file_name,
                "file_size_mb": file_size_mb,
                "max_size_mb": max_size_mb,
            },
        )


class FileNotFoundException(DocumentProcessingException):
    """Raised when file cannot be located."""
    
    def __init__(self, file_path: str):
        super().__init__(
            message=f"File not found: {file_path}",
            error_code=ErrorCode.FILE_NOT_FOUND,
            context={"file_path": file_path},
        )


# ============================================================================
# CHUNKING EXCEPTIONS
# ============================================================================


class ChunkingException(CognitiveNexusBaseException):
    """Base exception for chunking failures."""
    pass


class ChunkingFailedException(ChunkingException):
    """Raised when semantic chunking fails."""
    
    def __init__(
        self,
        message: str,
        document_id: str,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.CHUNKING_FAILED,
            context={"document_id": document_id},
            original_exception=original_exception,
        )


class InvalidChunkParametersException(ChunkingException):
    """Raised when chunking parameters are invalid."""
    
    def __init__(self, message: str, params: Dict[str, Any]):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_CHUNK_PARAMS,
            context={"invalid_params": params},
        )


class EmptyDocumentException(ChunkingException):
    """Raised when document yields no extractable text."""
    
    def __init__(self, document_id: str, file_name: str):
        super().__init__(
            message="Document contains no extractable text",
            error_code=ErrorCode.EMPTY_DOCUMENT,
            context={
                "document_id": document_id,
                "file_name": file_name,
            },
        )


# ============================================================================
# EMBEDDING EXCEPTIONS
# ============================================================================


class EmbeddingException(CognitiveNexusBaseException):
    """Base exception for embedding service failures."""
    pass


class EmbeddingModelLoadException(EmbeddingException):
    """Raised when embedding model fails to load."""
    
    def __init__(self, model_name: str, original_exception: Optional[Exception] = None):
        super().__init__(
            message=f"Failed to load embedding model: {model_name}",
            error_code=ErrorCode.EMBEDDING_MODEL_LOAD_FAILED,
            context={"model_name": model_name},
            original_exception=original_exception,
        )


class EmbeddingInferenceException(EmbeddingException):
    """Raised when embedding inference fails."""
    
    def __init__(
        self,
        message: str,
        batch_size: int,
        num_texts: int,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.EMBEDDING_INFERENCE_FAILED,
            context={
                "batch_size": batch_size,
                "num_texts": num_texts,
            },
            original_exception=original_exception,
        )


class EmbeddingOutOfMemoryException(EmbeddingException):
    """Raised when embedding inference runs out of memory."""
    
    def __init__(self, batch_size: int, suggested_batch_size: int):
        super().__init__(
            message="Embedding inference out of memory. Reduce batch size.",
            error_code=ErrorCode.EMBEDDING_OUT_OF_MEMORY,
            context={
                "current_batch_size": batch_size,
                "suggested_batch_size": suggested_batch_size,
            },
        )


class InvalidTextEncodingException(EmbeddingException):
    """Raised when text encoding is invalid."""
    
    def __init__(self, text_sample: str, encoding_error: str):
        super().__init__(
            message="Invalid text encoding encountered",
            error_code=ErrorCode.INVALID_TEXT_ENCODING,
            context={
                "text_sample": text_sample[:100],
                "encoding_error": encoding_error,
            },
        )


# ============================================================================
# VECTOR STORE EXCEPTIONS
# ============================================================================


class VectorStoreException(CognitiveNexusBaseException):
    """Base exception for vector store failures."""
    pass


class VectorStoreInitException(VectorStoreException):
    """Raised when vector store initialization fails."""
    
    def __init__(
        self,
        message: str,
        db_path: str,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VECTOR_STORE_INIT_FAILED,
            context={"db_path": db_path},
            original_exception=original_exception,
        )


class VectorStoreWriteException(VectorStoreException):
    """Raised when writing to vector store fails."""
    
    def __init__(
        self,
        message: str,
        collection_name: str,
        num_vectors: int,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VECTOR_STORE_WRITE_FAILED,
            context={
                "collection_name": collection_name,
                "num_vectors": num_vectors,
            },
            original_exception=original_exception,
        )


class VectorStoreReadException(VectorStoreException):
    """Raised when reading from vector store fails."""
    
    def __init__(
        self,
        message: str,
        collection_name: str,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VECTOR_STORE_READ_FAILED,
            context={"collection_name": collection_name},
            original_exception=original_exception,
        )


class VectorStoreCorruptedException(VectorStoreException):
    """Raised when vector store is corrupted."""
    
    def __init__(self, db_path: str, corruption_details: str):
        super().__init__(
            message="Vector store is corrupted or inaccessible",
            error_code=ErrorCode.VECTOR_STORE_CORRUPTED,
            context={
                "db_path": db_path,
                "corruption_details": corruption_details,
            },
        )


class CollectionNotFoundException(VectorStoreException):
    """Raised when requested collection does not exist."""
    
    def __init__(self, collection_name: str):
        super().__init__(
            message=f"Collection not found: {collection_name}",
            error_code=ErrorCode.COLLECTION_NOT_FOUND,
            context={"collection_name": collection_name},
        )


# ============================================================================
# BM25 EXCEPTIONS
# ============================================================================


class BM25Exception(CognitiveNexusBaseException):
    """Base exception for BM25 service failures."""
    pass


class BM25IndexingException(BM25Exception):
    """Raised when BM25 indexing fails."""
    
    def __init__(
        self,
        message: str,
        num_documents: int,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.BM25_INDEX_FAILED,
            context={"num_documents": num_documents},
            original_exception=original_exception,
        )


class BM25SearchException(BM25Exception):
    """Raised when BM25 search fails."""
    
    def __init__(self, message: str, query: str, original_exception: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.BM25_SEARCH_FAILED,
            context={"query": query},
            original_exception=original_exception,
        )


class BM25CorruptIndexException(BM25Exception):
    """Raised when BM25 index is corrupted."""
    
    def __init__(self, index_path: str):
        super().__init__(
            message="BM25 index is corrupted or unreadable",
            error_code=ErrorCode.BM25_CORRUPT_INDEX,
            context={"index_path": index_path},
        )


# ============================================================================
# GRAPH EXCEPTIONS
# ============================================================================


class GraphException(CognitiveNexusBaseException):
    """Base exception for graph service failures."""
    pass


class GraphExtractionException(GraphException):
    """Raised when entity-relation extraction fails."""
    
    def __init__(
        self,
        message: str,
        document_id: str,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.GRAPH_EXTRACTION_FAILED,
            context={"document_id": document_id},
            original_exception=original_exception,
        )


class GraphPersistenceException(GraphException):
    """Raised when graph persistence fails."""
    
    def __init__(
        self,
        message: str,
        graph_path: str,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.GRAPH_PERSISTENCE_FAILED,
            context={"graph_path": graph_path},
            original_exception=original_exception,
        )


class GraphLoadException(GraphException):
    """Raised when loading graph fails."""
    
    def __init__(
        self,
        message: str,
        graph_path: str,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.GRAPH_LOAD_FAILED,
            context={"graph_path": graph_path},
            original_exception=original_exception,
        )


# ============================================================================
# GENERAL EXCEPTIONS
# ============================================================================


class InternalServerException(CognitiveNexusBaseException):
    """Raised for unexpected internal errors."""
    
    def __init__(
        self,
        message: str,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            context={},
            original_exception=original_exception,
        )


class InvalidRequestException(CognitiveNexusBaseException):
    """Raised when request validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_REQUEST,
            context={"field": field} if field else {},
        )


class ResourceConflictException(CognitiveNexusBaseException):
    """Raised when resource already exists."""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            message=f"{resource_type} already exists: {resource_id}",
            error_code=ErrorCode.RESOURCE_CONFLICT,
            context={
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
        )
