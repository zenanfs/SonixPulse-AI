#!/usr/bin/env python3
"""
SonixPulse AI - Offline Lyric NLP Sentiment Analysis (Ollama Local Integration)
=================================================================================
Utiliza un LLM local (Ollama: qwen2.5 / gemma2) mediante API HTTP local
para analizar el sentimiento, carga emocional y vibras temáticas de letras
de canciones sin enviar datos a APIs comerciales de terceros.
"""

import os
import json
import logging
import requests
import argparse
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("SonixPulse.LyricNLP")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

class OllamaLyricAnalyzer:
    """Analizador de sentimiento y carga lírica con modelos locales mediante Ollama."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host
        self.model = model

    def check_connection(self) -> bool:
        """Comprueba si el servidor local de Ollama está activo."""
        try:
            res = requests.get(f"{self.host}/api/tags", timeout=2)
            if res.status_code == 200:
                logger.info(f"Ollama detectado en {self.host}. Modelos disponibles: {[m['name'] for m in res.json().get('models', [])]}")
                return True
        except Exception:
            logger.warning(f"No se pudo conectar a Ollama en {self.host}. Se usará el analizador heurístico local.")
        return False

    def analyze_lyric(self, track_name: str, artist: str, lyric_snippet: str) -> Dict[str, Any]:
        """Envía el snippet de letra a Ollama para extraer vectores emocionales en JSON."""
        prompt = f"""
        Actúa como un analista de Sonic Branding y Musicología.
        Analiza el siguiente fragmento de letra de la canción "{track_name}" de {artist}:

        "{lyric_snippet}"

        Responde ÚNICAMENTE con un objeto JSON válido con la siguiente estructura:
        {{
            "primary_emotion": "Euphoria | Nostalgia | Sensuality | Rebellion | Melancholy | Confidence",
            "emotional_intensity_score": 0.0 a 1.0,
            "themes": ["lista", "de", "temas"],
            "sonic_branding_fit": "Descripción breve de a qué tipo de marca conviene esta vibra lírica",
            "predicted_cultural_resonance_6m": "High | Medium | Low"
        }}
        """

        if self.check_connection():
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
                res = requests.post(f"{self.host}/api/generate", json=payload, timeout=15)
                if res.status_code == 200:
                    response_json = json.loads(res.json().get("response", "{}"))
                    logger.info(f"Análisis NLP exitoso con Ollama ({self.model})")
                    return response_json
            except Exception as e:
                logger.error(f"Error en inferencia Ollama: {e}")

        # Fallback Heurístico Local
        return self._heuristic_fallback(track_name, lyric_snippet)

    def _heuristic_fallback(self, track_name: str, lyric_snippet: str) -> Dict[str, Any]:
        """Fallback heurístico determinista si Ollama no está encendido."""
        lyric_lower = lyric_snippet.lower()
        if any(w in lyric_lower for w in ["bailar", "fiesta", "perreo", "dembow", "noche", "coqueta"]):
            return {
                "primary_emotion": "Euphoria / Sensuality",
                "emotional_intensity_score": 0.88,
                "themes": ["Hedonismo", "Vida Nocturna", "Atracción"],
                "sonic_branding_fit": "Marcas de bebidas alcohólicas, snacks y ropa juvenil",
                "predicted_cultural_resonance_6m": "Medium (Saturado)"
            }
        else:
            return {
                "primary_emotion": "Nostalgia / Melancholy",
                "emotional_intensity_score": 0.74,
                "themes": ["Introspección", "Autenticidad", "Memoria"],
                "sonic_branding_fit": "Marcas de tecnología, automóviles, seguros y café boutique",
                "predicted_cultural_resonance_6m": "High (Tendencia Emergente)"
            }

def main():
    parser = argparse.ArgumentParser(description="TrendScout - Análisis Lírico con Ollama Local")
    parser.add_argument("--track", type=str, default="BbY WOW", help="Nombre de la canción")
    parser.add_argument("--artist", type=str, default="KAROL G", help="Artista")
    parser.add_argument("--lyric", type=str, default="Qué chimba de noche, perreando hasta abajo con las babys", help="Fragmento de letra")

    args = parser.parse_args()

    analyzer = OllamaLyricAnalyzer()
    res = analyzer.analyze_lyric(args.track, args.artist, args.lyric)

    print("\n--- Resultado del Análisis NLP de Letra ---")
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
