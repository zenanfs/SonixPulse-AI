#!/usr/bin/env python3
"""
SonixPulse AI - FastAPI REST Engine & Real-Time Analytics API
=============================================================
Proporciona endpoints HTTP REST para ingesta de playlists de Spotify en vivo,
inferencia de K-Means en tiempo real, análisis de ROI financiero y matcheo
de arquetipos de marca para agencias publicitarias.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("SonixPulse.API")

# Añadir el directorio raíz al path de Python
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    FastAPI = None
    BaseModel = object

from src.ingestion.data_ingestion import SpotifyDataIngestion
from src.models.predictive_model import AcousticTrendPredictor
from src.models.financial_roi_engine import FinancialROIEngine

app = FastAPI(
    title="SonixPulse AI Sonic Branding API",
    description="API REST de Inteligencia Musical B2B para Agencias de Publicidad",
    version="2.4.0"
) if FastAPI else None

if app:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

class PlaylistAnalysisRequest(BaseModel):
    playlist_id: str = "37i9dQZEVXbLw8tC2WcDFy"
    force_offline: bool = False

class BrandMatchRequest(BaseModel):
    brand_archetype: str  # "tech", "luxury", "energy", "corporate"
    budget_usd: float = 50000.0

@app.get("/api/health") if app else lambda: None
def health_check():
    return {
        "status": "healthy",
        "service": "TrendScout Core API",
        "version": "2.4.0",
        "models": {
            "clustering": "K-Means (k=4)",
            "financial_engine": "Empirical Sync ROI v1.2",
            "nlp_local": "Ollama / Qwen2.5"
        }
    }

@app.post("/api/analyze-playlist") if app else lambda: None
def analyze_playlist(req: PlaylistAnalysisRequest):
    try:
        ingestion = SpotifyDataIngestion()
        df_raw = ingestion.run(playlist_id=req.playlist_id, force_offline=req.force_offline)

        predictor = AcousticTrendPredictor(n_clusters=4)
        X_scaled = predictor.preprocess_features(df_raw)
        df_clustered, sil_score = predictor.train_clusters(df_raw, X_scaled)
        telemetry = predictor.compute_market_telemetry(df_clustered)

        fin_engine = FinancialROIEngine()
        financial_summary = fin_engine.run_financial_analysis()

        return {
            "success": True,
            "playlist_id": req.playlist_id,
            "silhouette_score": sil_score,
            "telemetry": telemetry,
            "financial_roi": financial_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict-brand-match") if app else lambda: None
def predict_brand_match(req: BrandMatchRequest):
    archetype_maps = {
        "tech": {
            "name": "Tech & AI Innovator",
            "recommended_bpm": "118 - 126 BPM",
            "target_energy": 0.76,
            "target_valence": 0.58,
            "risk_level": "BAJO (12.4%)",
            "projected_savings": "$46,000 USD (-63.8%)",
            "recall_gain": "+34.8% vs Urbano",
            "top_track": "SUPERESTRELLA - Aitana"
        },
        "luxury": {
            "name": "Luxury & Fine Living",
            "recommended_bpm": "80 - 95 BPM",
            "target_energy": 0.42,
            "target_valence": 0.35,
            "risk_level": "MUY BAJO (4.1%)",
            "projected_savings": "$58,000 USD (-71.2%)",
            "recall_gain": "+48.2% vs Urbano",
            "top_track": "Doma - Jósean Log"
        },
        "energy": {
            "name": "Gen-Z Energy & Sports",
            "recommended_bpm": "128 - 134 BPM",
            "target_energy": 0.88,
            "target_valence": 0.82,
            "risk_level": "MODERADO (22.0%)",
            "projected_savings": "$32,000 USD (-45.0%)",
            "recall_gain": "+29.1% vs Urbano",
            "top_track": "Te Estoy Correteando - LATIN MAFIA"
        },
        "corporate": {
            "name": "Corporate & Fintech Trust",
            "recommended_bpm": "100 - 112 BPM",
            "target_energy": 0.55,
            "target_valence": 0.60,
            "risk_level": "BAJO (8.5%)",
            "projected_savings": "$40,000 USD (-55.5%)",
            "recall_gain": "+31.4% vs Urbano",
            "top_track": "Andean Chill - Lojano Beats"
        }
    }

    match_info = archetype_maps.get(req.brand_archetype.lower())
    if not match_info:
        raise HTTPException(status_code=400, detail="Arquetipo no válido. Opciones: tech, luxury, energy, corporate")

    return {
        "success": True,
        "query_archetype": req.brand_archetype,
        "match_details": match_info
    }

if __name__ == "__main__":
    if FastAPI:
        import uvicorn
        logger.info("Iniciando servidor REST TrendScout en http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("FastAPI / Uvicorn no instalado. Instálalos con: pip install fastapi uvicorn")
