"""
Cognitive Data Nexus - Configuration Management (Pydantic v2)
Phase 1: Ingestion, Chunking, Parsing

Environment-driven settings with validation, type safety, and sensible defaults.
All configuration is immutable after instantiation.
"""

from pathlib import Path
from typing import Optional, List
from enum import Enum

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment modes."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingDevice(str, Enum):
    """Device for embedding inference."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # macOS Metal Performance Shaders


class DocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "md"
    TXT = "txt"


class Settings(BaseSettings):
    """
    Cognitive Data Nexus configuration.
    
    Loads from environment variables with .env file support.
    All paths are created as absolute Path objects.
    """
    
    # ========================================================================
    # APPLICATION SETTINGS
    # ========================================================================
    
    app_name: str = Field(
        default="CognitiveDataNexus",
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0-phase1",
        description="Application version"
    )
    app_env: Environment = Field(
        default=Environment.PRODUCTION,
        description="Application environment (development/staging/production)"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # ========================================================================
    # LOGGING CONFIGURATION
    # ========================================================================
    
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Global logging level"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Optional log file path"
    )
    
    # ========================================================================
    # SERVER CONFIGURATION
    # ========================================================================
    
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API server port"
    )
    api_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of Uvicorn workers"
    )
    api_timeout_seconds: int = Field(
        default=300,
        ge=30,
        description="API request timeout in seconds"
    )
    
    # ========================================================================
    # DATA STORAGE PATHS
    # ========================================================================
    
    data_base_path: str = Field(
        default="./data",
        description="Base directory for all data storage"
    )
    raw_documents_path: Optional[str] = Field(
        default=None,
        description="Directory for raw uploaded documents"
    )
    processed_chunks_path: Optional[str] = Field(
        default=None,
        description="Directory for processed chunk metadata (JSON)"
    )
    embeddings_cache_path: Optional[str] = Field(
        default=None,
        description="Directory for Sentence-Transformers model cache"
    )
    chroma_db_path: Optional[str] = Field(
        default=None,
        description="ChromaDB persistent storage directory"
    )
    bm25_index_path: Optional[str] = Field(
        default=None,
        description="BM25 index storage directory"
    )
    graph_storage_path: Optional[str] = Field(
        default=None,
        description="NetworkX graph serialization directory"
    )
    
    # ========================================================================
    # DOCUMENT PROCESSING
    # ========================================================================
    
    allowed_formats: List[DocumentFormat] = Field(
        default=[
            DocumentFormat.PDF,
            DocumentFormat.DOCX,
            DocumentFormat.MARKDOWN,
        ],
        description="Allowed document formats"
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum file upload size in MB"
    )
    document_encoding: str = Field(
        default="utf-8",
        description="Default text encoding for documents"
    )
    
    # ========================================================================
    # CHUNKING CONFIGURATION
    # ========================================================================
    
    chunk_size_tokens: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Semantic chunk size in tokens"
    )
    chunk_overlap_tokens: int = Field(
        default=128,
        ge=0,
        le=512,
        description="Overlap between consecutive chunks in tokens"
    )
    chunk_strategy: str = Field(
        default="semantic_sliding_window",
        description="Chunking strategy to use"
    )
    
    # ========================================================================
    # EMBEDDING CONFIGURATION
    # ========================================================================
    
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-Transformers model name"
    )
    embedding_model_cache_dir: Optional[str] = Field(
        default=None,
        description="Custom cache directory for embedding models"
    )
    embedding_device: EmbeddingDevice = Field(
        default=EmbeddingDevice.CPU,
        description="Device for embedding inference (cpu/cuda/mps)"
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for embedding inference"
    )
    embedding_normalize: bool = Field(
        default=True,
        description="Normalize embedding vectors"
    )
    embedding_max_length: int = Field(
        default=384,
        ge=1,
        description="Maximum length for embedding inputs"
    )
    
    # ========================================================================
    # VECTOR STORE CONFIGURATION (CHROMADB)
    # ========================================================================
    
    chroma_collection_prefix: str = Field(
        default="doc_",
        description="Prefix for ChromaDB collection names"
    )
    chroma_metric: str = Field(
        default="cosine",
        description="Distance metric for ChromaDB (cosine/l2/ip)"
    )
    chroma_persist_directory: Optional[str] = Field(
        default=None,
        description="ChromaDB persist directory (defaults to chroma_db_path)"
    )
    
    # ========================================================================
    # BM25 CONFIGURATION
    # ========================================================================
    
    bm25_k1: float = Field(
        default=1.5,
        ge=0.0,
        description="BM25 k1 parameter (term frequency saturation)"
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 b parameter (length normalization)"
    )
    bm25_min_term_frequency: int = Field(
        default=1,
        ge=1,
        description="Minimum term frequency for BM25 indexing"
    )
    
    # ========================================================================
    # GRAPH CONFIGURATION
    # ========================================================================
    
    graph_format: str = Field(
        default="gml",
        description="Graph serialization format (gml/graphml/pickle)"
    )
    enable_ner: bool = Field(
        default=True,
        description="Enable Named Entity Recognition for graph building"
    )
    enable_relation_extraction: bool = Field(
        default=True,
        description="Enable relation extraction for graph building"
    )
    
    # ========================================================================
    # PERFORMANCE & OPTIMIZATION
    # ========================================================================
    
    max_workers_document_processing: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Max workers for async document processing"
    )
    max_workers_embedding: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Max workers for async embedding computation"
    )
    use_memory_cache: bool = Field(
        default=True,
        description="Enable in-memory caching for frequently accessed data"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Cache TTL in seconds"
    )
    
    # ========================================================================
    # PYDANTIC SETTINGS CONFIGURATION
    # ========================================================================
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown env vars
    )
    
    # ========================================================================
    # VALIDATORS
    # ========================================================================
    
    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info):
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size_tokens", 512)
        if chunk_size and v >= chunk_size:
            raise ValueError(
                f"chunk_overlap_tokens ({v}) must be less than "
                f"chunk_size_tokens ({chunk_size})"
            )
        return v
    
    @field_validator("embedding_batch_size")
    @classmethod
    def validate_batch_size(cls, v: int, info):
        """Warn if batch size is too large for typical hardware."""
        if v > 256:
            import warnings
            warnings.warn(
                f"Large embedding batch size ({v}) may cause OOM. "
                "Consider reducing if on limited hardware.",
                UserWarning
            )
        return v
    
    @field_validator("bm25_b")
    @classmethod
    def validate_bm25_b(cls, v: float):
        """Validate BM25 b parameter."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("bm25_b must be between 0.0 and 1.0")
        return v
    
    @model_validator(mode="after")
    def setup_default_paths(self):
        """Set default paths for data storage if not explicitly provided."""
        base_path = Path(self.data_base_path)
        
        if not self.raw_documents_path:
            self.raw_documents_path = str(base_path / "raw")
        if not self.processed_chunks_path:
            self.processed_chunks_path = str(base_path / "processed")
        if not self.embeddings_cache_path:
            self.embeddings_cache_path = str(base_path / "embeddings")
        if not self.chroma_db_path:
            self.chroma_db_path = str(base_path / "chroma_db")
        if not self.bm25_index_path:
            self.bm25_index_path = str(base_path / "bm25_index")
        if not self.graph_storage_path:
            self.graph_storage_path = str(base_path / "graph")
        if not self.chroma_persist_directory:
            self.chroma_persist_directory = self.chroma_db_path
        if not self.embedding_model_cache_dir:
            self.embedding_model_cache_dir = self.embeddings_cache_path
        
        return self
    
    @model_validator(mode="after")
    def create_directories(self):
        """Ensure all data directories exist."""
        directories = [
            self.raw_documents_path,
            self.processed_chunks_path,
            self.embeddings_cache_path,
            self.chroma_db_path,
            self.bm25_index_path,
            self.graph_storage_path,
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
        
        return self
    
    # ========================================================================
    # COMPUTED PROPERTIES
    # ========================================================================
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == Environment.DEVELOPMENT
    
    @property
    def raw_documents_dir(self) -> Path:
        """Get raw documents directory as Path object."""
        return Path(self.raw_documents_path)
    
    @property
    def processed_chunks_dir(self) -> Path:
        """Get processed chunks directory as Path object."""
        return Path(self.processed_chunks_path)
    
    @property
    def embeddings_cache_dir(self) -> Path:
        """Get embeddings cache directory as Path object."""
        return Path(self.embeddings_cache_path)
    
    @property
    def chroma_db_dir(self) -> Path:
        """Get ChromaDB directory as Path object."""
        return Path(self.chroma_db_path)
    
    @property
    def bm25_index_dir(self) -> Path:
        """Get BM25 index directory as Path object."""
        return Path(self.bm25_index_path)
    
    @property
    def graph_storage_dir(self) -> Path:
        """Get graph storage directory as Path object."""
        return Path(self.graph_storage_path)
    
    @property
    def allowed_formats_str(self) -> List[str]:
        """Get allowed formats as string list."""
        return [fmt.value for fmt in self.allowed_formats]
    
    def to_dict(self) -> dict:
        """Serialize settings to dictionary (sensitive data masked)."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "app_env": self.app_env.value,
            "debug": self.debug,
            "log_level": self.log_level.value,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "api_workers": self.api_workers,
            "embedding_model": self.embedding_model_name,
            "embedding_device": self.embedding_device.value,
            "chunk_size_tokens": self.chunk_size_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "allowed_formats": self.allowed_formats_str,
        }


# Singleton instance
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get or create the global Settings instance.
    
    Uses lazy initialization pattern for pytest fixtures.
    """
    global settings
    if settings is None:
        settings = Settings()
    return settings


def reset_settings() -> None:
    """Reset settings instance (useful for testing)."""
    global settings
    settings = None
