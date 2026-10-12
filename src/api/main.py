"""
SonixPulse AI - Predictor de Exito de Contenido (Fase 4)
Servicio API REST para evaluacion multimodal y diagnostico de impacto comercial
mediante arboles de decision calibrados y explicabilidad marginal SHAP.

Endpoints:
- GET  /api/health: Estado del servicio y disponibilidad del modelo predictivo.
- POST /api/predict-virality: Inferencia multimodal mediante carga de archivos y presets.
- Servidor de archivos estaticos montado en '/' para el panel de control web.
"""

import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Garantizar ruta base del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("SonixPulse.API")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.features import (
    ALL_CONTINUOUS_FEATURE_NAMES,
    extract_audio_features,
    extract_text_features,
    extract_visual_features,
)
from src.models.virality_explainer import DEFAULT_MODEL_PATH, ViralityExplainer

app = FastAPI(
    title="SonixPulse AI - Predictor de Exito de Contenido",
    description="Motor de diagnostico comercial y evaluacion multimodal de viralidad P90.",
    version="3.0.0-comercial",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOADS_DIR = PROJECT_ROOT / "temp_uploads"
TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_EXPLAINER: Optional[ViralityExplainer] = None


def get_explainer() -> ViralityExplainer:
    """Retorna la instancia unica en memoria del explicador comercial."""
    global _EXPLAINER
    if _EXPLAINER is None:
        logger.info("Cargando modelo virality_xgb_model.joblib e inicializando explicador...")
        _EXPLAINER = ViralityExplainer()
    return _EXPLAINER


# Presets comerciales representativos calibrados con la distribucion P90
PRESETS: Dict[str, Dict[str, Any]] = {
    "campana_comercial": {
        "nombre": "Campana Comercial",
        "descripcion": "Anuncio publicitario de alto impacto con balance visual y mensaje directo",
        "features": {
            "visual_brightness_mean": 133.85,
            "visual_contrast_rms": 54.49,
            "visual_color_entropy": 4.79,
            "visual_saturation_mean": 43.59,
            "visual_canny_edge_density": 0.04,
            "audio_rms_mean": 0.11,
            "audio_zcr_mean": 0.07,
            "audio_bpm_tempo": 116.02,
            "audio_bpm": 116.02,
            "audio_mfcc_1": -194.8,
            "audio_mfcc_2": 94.83,
            "audio_mfcc_3": -39.4,
            "audio_mfcc_4": 17.8,
            "audio_mfcc_5": -18.49,
            "audio_mfcc_6": 13.09,
            "audio_mfcc_7": 2.13,
            "audio_mfcc_8": 5.53,
            "audio_mfcc_9": 5.31,
            "audio_mfcc_10": 9.85,
            "audio_mfcc_11": 3.83,
            "audio_mfcc_12": -10.31,
            "audio_mfcc_13": 5.95,
            "text_inflesz_score": 71.24,
            "text_fernandez_huerta_score": 75.62,
            "text_vader_compound": 0.45,
            "text_vader_pos": 0.32,
            "text_word_count": 11.0,
            "text_raw": "Lanzamiento exclusivo de temporada: maximiza tu rendimiento con diseno de vanguardia.",
        },
    },
    "hit_ritmico": {
        "nombre": "Hit Ritmico",
        "descripcion": "Pista musical de alto dinamismo con tempo energizante y gancho percusivo",
        "features": {
            "visual_brightness_mean": 123.15,
            "visual_contrast_rms": 34.5,
            "visual_color_entropy": 5.92,
            "visual_saturation_mean": 107.24,
            "visual_canny_edge_density": 0.09,
            "audio_rms_mean": 0.22,
            "audio_zcr_mean": 0.07,
            "audio_bpm_tempo": 124.0,
            "audio_bpm": 124.0,
            "audio_mfcc_1": -101.97,
            "audio_mfcc_2": 113.12,
            "audio_mfcc_3": -21.83,
            "audio_mfcc_4": 21.41,
            "audio_mfcc_5": -4.68,
            "audio_mfcc_6": -7.21,
            "audio_mfcc_7": -14.51,
            "audio_mfcc_8": 3.71,
            "audio_mfcc_9": -2.83,
            "audio_mfcc_10": 4.85,
            "audio_mfcc_11": -2.75,
            "audio_mfcc_12": 9.86,
            "audio_mfcc_13": -2.67,
            "text_inflesz_score": 75.0,
            "text_fernandez_huerta_score": 78.0,
            "text_vader_compound": 0.65,
            "text_vader_pos": 0.40,
            "text_word_count": 11.0,
            "text_raw": "El beat mas pegadizo del momento con ritmo arrollador para bailar sin parar.",
        },
    },
    "podcast_conversacional": {
        "nombre": "Podcast Conversacional",
        "descripcion": "Episodio de dialogo analitico con ritmo sosegado y estructura explicativa",
        "features": {
            "visual_brightness_mean": 140.53,
            "visual_contrast_rms": 59.4,
            "visual_color_entropy": 5.17,
            "visual_saturation_mean": 109.52,
            "visual_canny_edge_density": 0.04,
            "audio_rms_mean": 0.1,
            "audio_zcr_mean": 0.08,
            "audio_bpm_tempo": 95.0,
            "audio_bpm": 95.0,
            "audio_mfcc_1": -199.64,
            "audio_mfcc_2": 120.94,
            "audio_mfcc_3": -29.69,
            "audio_mfcc_4": 22.55,
            "audio_mfcc_5": 0.72,
            "audio_mfcc_6": -2.4,
            "audio_mfcc_7": -7.4,
            "audio_mfcc_8": 13.37,
            "audio_mfcc_9": 4.79,
            "audio_mfcc_10": 10.84,
            "audio_mfcc_11": -2.5,
            "audio_mfcc_12": 1.01,
            "audio_mfcc_13": -2.96,
            "text_inflesz_score": 41.09,
            "text_fernandez_huerta_score": 46.64,
            "text_vader_compound": 0.0,
            "text_vader_pos": 0.10,
            "text_word_count": 10.0,
            "text_raw": "Episodio especial: analisis en profundidad sobre habitos, productividad y reflexion diaria.",
        },
    },
}

PRESET_ALIASES: Dict[str, str] = {
    "viral_hit": "hit_ritmico",
    "podcast": "podcast_conversacional",
    "indie_melancholic": "podcast_conversacional",
}


@app.get("/api/health")
def health_check():
    """Diagnostico de salud del servicio y disponibilidad del modelo predictivo."""
    model_loaded = False
    try:
        expl = get_explainer()
        model_loaded = expl is not None and expl.classifier is not None
    except Exception as exc:
        logger.error("Error al verificar modelo en health check: %s", exc)

    return {
        "status": "healthy" if model_loaded else "degraded",
        "estado": "operativo" if model_loaded else "degradado",
        "servicio": "SonixPulse AI Predictor de Exito de Contenido",
        "service": "SonixPulse AI Predictor de Exito de Contenido",
        "version": "3.0.0-comercial",
        "modelo_cargado": model_loaded,
        "model_loaded": model_loaded,
        "artefacto_modelo": DEFAULT_MODEL_PATH.name,
        "variables_multimodales": len(ALL_CONTINUOUS_FEATURE_NAMES),
        "calibracion_p90": True,
    }


@app.post("/api/predict-virality")
async def predict_virality(
    request: Request,
    imagen: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    texto: Optional[str] = Form(None),
    preset: Optional[str] = Form(None),
):
    """
    Evalua el potencial de viralidad de una pieza de contenido multimodal.
    Recibe imagen de portada, pista de audio, copy o preset comercial rapido.
    Garantiza almacenamiento efimero y limpieza estricta en bloque finally.
    """
    t0 = time.time()
    temp_files_to_clean: List[Path] = []

    # Compatibilidad con clientes que envian JSON
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                preset = body.get("preset", preset)
                texto = body.get("texto") or body.get("text", texto)
        except Exception:
            pass

    try:
        explainer = get_explainer()
        features_dict: Dict[str, Any] = {}
        origen_info: str = "directo"
        text_content: str = ""

        # Normalizacion de alias de presets
        if preset:
            normalized_preset = preset.lower().strip()
            if normalized_preset in PRESET_ALIASES:
                normalized_preset = PRESET_ALIASES[normalized_preset]

            if normalized_preset in PRESETS:
                preset_data = PRESETS[normalized_preset]
                features_dict = preset_data["features"].copy()
                origen_info = f"preset_{normalized_preset}"
                text_content = str(features_dict.get("text_raw", ""))

        # Carga de imagen para extraccion visual
        if imagen is not None and imagen.filename:
            raw_bytes = await imagen.read()
            if len(raw_bytes) > 0:
                ext = Path(imagen.filename).suffix or ".jpg"
                temp_img_path = TEMP_UPLOADS_DIR / f"img_{uuid.uuid4().hex}{ext}"
                temp_img_path.write_bytes(raw_bytes)
                temp_files_to_clean.append(temp_img_path)

                visual_extracted = extract_visual_features(temp_img_path)
                features_dict.update(visual_extracted)
                origen_info = "archivo_multimodal"

        # Carga de pista de audio para extraccion acustica
        if audio is not None and audio.filename:
            raw_bytes = await audio.read()
            if len(raw_bytes) > 0:
                ext = Path(audio.filename).suffix or ".mp3"
                temp_audio_path = TEMP_UPLOADS_DIR / f"audio_{uuid.uuid4().hex}{ext}"
                temp_audio_path.write_bytes(raw_bytes)
                temp_files_to_clean.append(temp_audio_path)

                audio_extracted = extract_audio_features(temp_audio_path)
                features_dict.update(audio_extracted)
                origen_info = "archivo_multimodal"

        # Copy o texto provisto directamente
        if texto is not None and texto.strip():
            text_content = texto.strip()
            text_extracted = extract_text_features(text_content)
            features_dict.update(text_extracted)
            features_dict["text_raw"] = text_content
            origen_info = "texto_directo" if origen_info == "directo" else origen_info

        # Si no se suministraron archivos ni preset ni texto, usar preset campana_comercial por defecto
        if not features_dict:
            default_preset = PRESETS["campana_comercial"]
            features_dict = default_preset["features"].copy()
            origen_info = "preset_campana_comercial"
            text_content = str(features_dict.get("text_raw", ""))

        # Si no hay texto definido todavia, asegurar campo text_raw
        if "text_raw" not in features_dict:
            features_dict["text_raw"] = text_content

        # Ejecucion de la inferencia SHAP comercial
        explanation = explainer.explain_prediction(features_dict)

        probability = float(explanation["probabilidad"])
        prob_pct = round(probability * 100.0, 1)
        is_viral = bool(explanation["es_viral_p90"])
        radar_comercial = explanation.get("radar_comercial", {})
        factores_impulso = explanation.get("factores_impulso", [])
        factores_friccion = explanation.get("factores_friccion", [])

        # Sintesis de recomendaciones de negocio orientadas a marketing
        recomendaciones: List[str] = []
        for factor in factores_friccion[:3]:
            recomendaciones.append(factor["recomendacion"])
        for factor in factores_impulso[:2]:
            recomendaciones.append(factor["recomendacion"])

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return {
            # Payload comercial de Fase 4
            "exito": True,
            "probabilidad": probability,
            "probabilidad_porcentaje": prob_pct,
            "es_viral_p90": is_viral,
            "radar_comercial": radar_comercial,
            "factores_impulso": factores_impulso,
            "factores_friccion": factores_friccion,
            "recomendaciones": recomendaciones,
            # Campos para compatibilidad con suites de pruebas
            "success": True,
            "probability": probability,
            "probability_percentage": prob_pct,
            "radar_scores": explanation.get("radar_scores", {}),
            "top_positive_factors": explanation.get("top_positive_factors", []),
            "top_negative_factors": explanation.get("top_negative_factors", []),
            "recommendations": recomendaciones,
            "metadata": {
                "origen": origen_info,
                "texto_analizado": features_dict.get("text_raw", ""),
                "tiempo_procesamiento_ms": elapsed_ms,
            },
        }

    except Exception as exc:
        logger.error("Error durante evaluacion de viralidad: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en diagnostico multimodal: {str(exc)}",
        )
    finally:
        # Eliminacion inmediata de archivos temporales
        for temp_file in temp_files_to_clean:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as clean_err:
                logger.warning("No se pudo eliminar archivo temporal %s: %s", temp_file, clean_err)


# Montaje estatico del Frontend
web_dir = PROJECT_ROOT / "web"
if web_dir.exists() and web_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")
else:
    logger.warning("Directorio web/ no encontrado en %s.", web_dir)


if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando SonixPulse AI en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
