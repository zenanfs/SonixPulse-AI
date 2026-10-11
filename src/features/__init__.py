"""
SonixPulse AI - Multimodal Feature Extraction Package
=====================================================
Módulos deterministas y vectorizados de extracción de características en CPU:
- extract_visual_features: Métricas visuales de brillo, contraste, entropía, saturación y Canny.
- extract_audio_features: Métricas acústicas de RMS, cruces por cero, tempo BPM y 13 coeficientes MFCC.
- extract_text_features: Métricas lingüísticas de legibilidad (INFLESZ, Fernández-Huerta) y sentimiento VADER.
"""

from src.features.visual_extractor import (
    extract_visual_features,
    DEFAULT_VISUAL_FEATURES,
    FEATURE_NAMES_VISUAL,
)
from src.features.audio_extractor import (
    extract_audio_features,
    DEFAULT_AUDIO_FEATURES,
    FEATURE_NAMES_AUDIO,
)
from src.features.text_extractor import (
    extract_text_features,
    DEFAULT_TEXT_FEATURES,
    FEATURE_NAMES_TEXT,
)

# Lista unificada de todas las características continuas (para normalización con StandardScaler)
ALL_CONTINUOUS_FEATURE_NAMES = (
    FEATURE_NAMES_VISUAL + FEATURE_NAMES_AUDIO + FEATURE_NAMES_TEXT
)

__all__ = [
    "extract_visual_features",
    "extract_audio_features",
    "extract_text_features",
    "DEFAULT_VISUAL_FEATURES",
    "DEFAULT_AUDIO_FEATURES",
    "DEFAULT_TEXT_FEATURES",
    "FEATURE_NAMES_VISUAL",
    "FEATURE_NAMES_AUDIO",
    "FEATURE_NAMES_TEXT",
    "ALL_CONTINUOUS_FEATURE_NAMES",
]
