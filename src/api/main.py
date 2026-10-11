#!/usr/bin/env python3
"""
SonixPulse AI - Multimodal Virality Ingestion & SHAP Explainability API (Phase 5)
================================================================================
Servidor FastAPI de alto rendimiento para predicción de viralidad P90 y explicabilidad
multimodal mediante SHAP TreeExplainer y modelos XGBoost calibrados.

Endpoints:
- GET  /api/health: Diagnóstico de salud, versión 3.0.0-beta y estado del modelo.
- POST /api/predict-virality: Inferencia multimodal vía URL (YouTube/TikTok), presets rápidos o texto crudo.
- Servidor estático montado en la raíz '/' sirviendo el dashboard web interactivo.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Garantizar path del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("SonixPulse.API")

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models.virality_explainer import ViralityExplainer, DEFAULT_MODEL_PATH
from src.ingestion.multimodal_ingestion import ingest_url
from src.features import (
    extract_visual_features,
    extract_audio_features,
    extract_text_features,
    ALL_CONTINUOUS_FEATURE_NAMES,
)

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="SonixPulse AI - Multimodal Virality Predictor",
    description="Motor de predicción e interpretabilidad SHAP de viralidad P90 para contenido multimedia.",
    version="3.0.0-beta",
)

# Configuración de CORS para soporte web local y clientes externos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton global para ViralityExplainer
_EXPLAINER: Optional[ViralityExplainer] = None


def get_explainer() -> ViralityExplainer:
    """Retorna una instancia única en memoria del explicador SHAP."""
    global _EXPLAINER
    if _EXPLAINER is None:
        logger.info("Cargando modelo virality_xgb_model.joblib e inicializando ViralityExplainer...")
        _EXPLAINER = ViralityExplainer()
    return _EXPLAINER


# Definición de presets heurísticos y sintéticos calibrados
PRESETS: Dict[str, Dict[str, Any]] = {
    "viral_hit": {
        "title": "Preset: Hit Viral de Alto Impacto (TikTok/Reels)",
        "features": {
            "visual_brightness_mean": 155.0,
            "visual_contrast_rms": 82.0,
            "visual_color_entropy": 6.9,
            "visual_saturation_mean": 145.0,
            "visual_canny_edge_density": 0.14,
            "audio_rms_mean": 0.22,
            "audio_zcr_mean": 0.08,
            "audio_bpm_tempo": 128.0,
            "audio_mfcc_1": -120.0,
            "audio_mfcc_2": 118.0,
            "audio_mfcc_3": -10.0,
            "audio_mfcc_4": 30.0,
            "audio_mfcc_5": 2.0,
            "audio_mfcc_6": 15.0,
            "audio_mfcc_7": -5.0,
            "audio_mfcc_8": 8.0,
            "audio_mfcc_9": -2.0,
            "audio_mfcc_10": 4.0,
            "audio_mfcc_11": -1.0,
            "audio_mfcc_12": 2.0,
            "audio_mfcc_13": -0.5,
            "text_inflesz_score": 78.0,
            "text_fernandez_huerta_score": 82.0,
            "text_vader_compound": 0.88,
            "text_word_count": 12.0,
            "text_raw": "¡Alerta de temazo! La canción que va a dominar el verano en todas las discos #viral #trending",
        },
    },
    "podcast": {
        "title": "Preset: Podcast Conversacional / Entrevista Técnica",
        "features": {
            "visual_brightness_mean": 115.0,
            "visual_contrast_rms": 36.0,
            "visual_color_entropy": 4.6,
            "visual_saturation_mean": 48.0,
            "visual_canny_edge_density": 0.03,
            "audio_rms_mean": 0.05,
            "audio_zcr_mean": 0.035,
            "audio_bpm_tempo": 90.0,
            "audio_mfcc_1": -210.0,
            "audio_mfcc_2": 70.0,
            "audio_mfcc_3": -25.0,
            "audio_mfcc_4": 15.0,
            "audio_mfcc_5": -10.0,
            "audio_mfcc_6": 8.0,
            "audio_mfcc_7": -12.0,
            "audio_mfcc_8": 3.0,
            "audio_mfcc_9": -8.0,
            "audio_mfcc_10": 1.0,
            "audio_mfcc_11": -4.0,
            "audio_mfcc_12": 0.5,
            "audio_mfcc_13": -2.0,
            "text_inflesz_score": 88.0,
            "text_fernandez_huerta_score": 85.0,
            "text_vader_compound": 0.15,
            "text_word_count": 28.0,
            "text_raw": "Lectura continua y análisis técnico sobre neurociencia cognitiva y hábitos diarios en el estudio.",
        },
    },
    "indie_melancholic": {
        "title": "Preset: Música Indie Melancólica / Balada Lofi",
        "features": {
            "visual_brightness_mean": 88.0,
            "visual_contrast_rms": 28.0,
            "visual_color_entropy": 4.1,
            "visual_saturation_mean": 38.0,
            "visual_canny_edge_density": 0.025,
            "audio_rms_mean": 0.04,
            "audio_zcr_mean": 0.028,
            "audio_bpm_tempo": 80.0,
            "audio_mfcc_1": -240.0,
            "audio_mfcc_2": 60.0,
            "audio_mfcc_3": -30.0,
            "audio_mfcc_4": 10.0,
            "audio_mfcc_5": -12.0,
            "audio_mfcc_6": 5.0,
            "audio_mfcc_7": -15.0,
            "audio_mfcc_8": 2.0,
            "audio_mfcc_9": -10.0,
            "audio_mfcc_10": 0.5,
            "audio_mfcc_11": -5.0,
            "audio_mfcc_12": 0.2,
            "audio_mfcc_13": -3.0,
            "text_inflesz_score": 52.0,
            "text_fernandez_huerta_score": 50.0,
            "text_vader_compound": -0.42,
            "text_word_count": 18.0,
            "text_raw": "Lluvia silenciosa y recuerdos solitarios en una noche fría sin rumbo ni despedidas.",
        },
    },
}


# =====================================================================
# MODELOS PYDANTIC PARA REQUEST Y RESPONSE
# =====================================================================

class ViralityPredictRequest(BaseModel):
    url: Optional[str] = Field(None, description="URL pública de YouTube, Shorts, TikTok, etc.")
    text: Optional[str] = Field(None, description="Texto o guión del contenido para extracción lingüística.")
    preset: Optional[str] = Field(None, description="Preset rápido: 'viral_hit', 'podcast', 'indie_melancholic'")
    features: Optional[Dict[str, Any]] = Field(None, description="Diccionario opcional con características ya calculadas.")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    model_artifact: str
    model_path: str
    features_count: int
    p90_calibrated: bool


# =====================================================================
# ENDPOINTS REST DE LA API
# =====================================================================

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Diagnóstico de salud del servicio y verificación del modelo de viralidad."""
    model_path_exists = DEFAULT_MODEL_PATH.exists()
    model_loaded = False
    try:
        expl = get_explainer()
        model_loaded = expl is not None and expl.classifier is not None
    except Exception as exc:
        logger.error("Error al verificar modelo en health check: %s", exc)

    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": "SonixPulse AI Multimodal Virality Engine",
        "version": "3.0.0-beta",
        "model_loaded": model_loaded,
        "model_artifact": DEFAULT_MODEL_PATH.name,
        "model_path": str(DEFAULT_MODEL_PATH),
        "features_count": len(ALL_CONTINUOUS_FEATURE_NAMES),
        "p90_calibrated": True,
    }


@app.post("/api/predict-virality")
def predict_virality(req: ViralityPredictRequest):
    """
    Infiere la probabilidad de pertenecer al P90 viral y computa la interpretabilidad SHAP.
    Soporta URLs públicas, presets predefinidos o análisis textual.
    """
    try:
        explainer = get_explainer()
        fallback_applied = False
        fallback_reason: Optional[str] = None
        target_url: Optional[str] = req.url
        metadata: Dict[str, Any] = {}
        features_dict: Dict[str, Any] = {}

        # 1. Caso A: Preset preconfigurado
        if req.preset and req.preset.lower() in PRESETS:
            preset_key = req.preset.lower()
            preset_data = PRESETS[preset_key]
            features_dict = preset_data["features"]
            metadata = {
                "title": preset_data["title"],
                "preset": preset_key,
                "source": "preset_simulator",
            }
            target_url = f"preset://{preset_key}"

        # 2. Caso B: Ingesta desde URL pública
        elif req.url and req.url.strip():
            clean_url = req.url.strip()
            logger.info("Procesando ingesta multimodal para URL: %s", clean_url)
            ingest_result = ingest_url(clean_url)

            features_dict = ingest_result.get("features", {})
            metadata = ingest_result.get("metadata", {})
            fallback_applied = metadata.get("fallback_applied", False)
            fallback_reason = metadata.get("fallback_reason")
            target_url = clean_url

        # 3. Caso C: Características suministradas directamente
        elif req.features and isinstance(req.features, dict):
            features_dict = req.features
            metadata = {
                "title": "Vector de Características Directo",
                "source": "custom_features",
            }
            target_url = "custom://features"

        # 4. Caso D: Texto o Guión libre
        elif req.text and req.text.strip():
            raw_text = req.text.strip()
            text_feats = extract_text_features(raw_text)
            visual_feats = extract_visual_features(None)
            audio_feats = extract_audio_features(None)

            features_dict = {
                **visual_feats,
                **audio_feats,
                **text_feats,
                "text_raw": raw_text,
            }
            metadata = {
                "title": raw_text[:50] + ("..." if len(raw_text) > 50 else ""),
                "source": "raw_text_input",
            }
            target_url = "raw://text"

        # 5. Fallback por defecto: Viral Hit
        else:
            preset_data = PRESETS["viral_hit"]
            features_dict = preset_data["features"]
            metadata = {
                "title": "Preset Predeterminado (Viral Hit)",
                "source": "default_preset",
            }
            target_url = "preset://viral_hit"

        # Ejecución de la inferencia y cálculo de SHAP
        explanation = explainer.explain_prediction(features_dict)

        probability = float(explanation["probability"])
        probability_percentage = round(probability * 100.0, 1)
        is_viral_p90 = bool(explanation["is_viral_p90"])
        radar_scores = explanation.get("radar_scores", {"visual": 50.0, "acoustic": 50.0, "textual": 50.0})
        top_positive_factors = explanation.get("top_positive_factors", [])
        top_negative_factors = explanation.get("top_negative_factors", [])
        recommendations = explanation.get("recommendations", [])

        return {
            "success": True,
            "url": target_url,
            "metadata": metadata,
            "probability": probability,
            "probability_percentage": probability_percentage,
            "is_viral_p90": is_viral_p90,
            "radar_scores": radar_scores,
            "top_positive_factors": top_positive_factors,
            "top_negative_factors": top_negative_factors,
            "recommendations": recommendations,
            "fallback_applied": fallback_applied,
            "fallback_reason": fallback_reason,
        }

    except Exception as exc:
        logger.error("Error en /api/predict-virality: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en inferencia multimodal: {str(exc)}",
        )


# =====================================================================
# MONTAJE ESTÁTICO DEL FRONTEND
# =====================================================================
web_dir = PROJECT_ROOT / "web"
if web_dir.exists() and web_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")
else:
    logger.warning("Directorio 'web/' no encontrado en %s. No se montó frontend estático.", web_dir)


if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando SonixPulse AI Virality REST Engine en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
