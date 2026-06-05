"""
Cognitive Data Nexus - Backend Application
Phase 1: Ingestion, Chunking, Parsing

FastAPI-based document intelligence and RAG pipeline.
Fully local, zero cloud API dependencies.
"""

__version__ = "1.0.0-phase1"
__author__ = "Cognitive Data Nexus Team"
__description__ = "Local document intelligence and graph-augmented RAG pipeline"

from app.config import get_settings, Settings
from app.exceptions import CognitiveNexusBaseException
from app.logging_config import get_logger, setup_logging

__all__ = [
    "get_settings",
    "Settings",
    "CognitiveNexusBaseException",
    "get_logger",
    "setup_logging",
]
