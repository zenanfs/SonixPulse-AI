"""
SonixPulse AI - Multimodal Ingestion Package
=============================================
Exporta utilidades e ingestores efímeros para URLs públicas y fuentes multimedia.
"""

from src.ingestion.multimodal_ingestion import (
    MultimodalIngestor,
    ingest_multimodal_url,
    ingest_url,
    find_ffmpeg,
)

__all__ = [
    "MultimodalIngestor",
    "ingest_multimodal_url",
    "ingest_url",
    "find_ffmpeg",
]
