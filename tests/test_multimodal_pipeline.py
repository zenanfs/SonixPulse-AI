#!/usr/bin/env python3
"""
Test Suite Unificado: Pipeline Multimodal SonixPulse AI
======================================================
Valida exhaustivamente los 5 pilares del sistema multimodal:
1. Extracción de características visuales en memoria (OpenCV/skimage).
2. Extracción de características acústicas sintéticas (librosa/soundfile).
3. Extracción de métricas lingüísticas (INFLESZ, Fernández-Huerta, VADER).
4. Inferencia con XGBoost y explicabilidad con SHAP TreeExplainer.
5. Llamada integral a la API REST de FastAPI (/api/predict-virality).
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Añadir raíz del proyecto al path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.visual_extractor import extract_visual_features, DEFAULT_VISUAL_FEATURES
from src.features.audio_extractor import extract_audio_features, DEFAULT_AUDIO_FEATURES
from src.features.text_extractor import extract_text_features, DEFAULT_TEXT_FEATURES
from src.features import ALL_CONTINUOUS_FEATURE_NAMES
from src.models.virality_explainer import ViralityExplainer
from src.api.main import app
from fastapi.testclient import TestClient


class TestMultimodalPipeline(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_01_visual_extractor_synthetic(self):
        """1. Test de extracción visual con imagen sintética en memoria."""
        # Crear imagen sintética RGB con gradiente y formas (120x120x3)
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        img[:60, :, 0] = 200   # canal R
        img[60:, :, 1] = 180   # canal G
        img[:, 40:80, 2] = 220 # canal B

        features = extract_visual_features(img)

        expected_keys = [
            "visual_brightness_mean",
            "visual_contrast_rms",
            "visual_color_entropy",
            "visual_saturation_mean",
            "visual_canny_edge_density"
        ]
        for key in expected_keys:
            self.assertIn(key, features, f"Falta la métrica visual: {key}")
            self.assertIsInstance(features[key], float)
            self.assertFalse(np.isnan(features[key]), f"Valor NaN detectado en {key}")

        # Probar fallback con entrada None
        fallback_features = extract_visual_features(None)
        self.assertEqual(len(fallback_features), 5)
        self.assertEqual(fallback_features["visual_brightness_mean"], DEFAULT_VISUAL_FEATURES["visual_brightness_mean"])

    def test_02_audio_extractor_synthetic(self):
        """2. Test de extracción acústica con señal sintética en memoria."""
        sr = 22050
        duration = 1.5  # 1.5 segundos
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Tono senoidal a 440 Hz modulado con armónico a 880 Hz
        signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)

        features = extract_audio_features(signal, sr=sr)

        from src.features.audio_extractor import FEATURE_NAMES_AUDIO
        for key in FEATURE_NAMES_AUDIO:
            self.assertIn(key, features, f"Falta la métrica de audio: {key}")
            self.assertIsInstance(features[key], float)
            self.assertFalse(np.isnan(features[key]), f"Valor NaN detectado en {key}")

        # Probar fallback con entrada None
        fallback_features = extract_audio_features(None)
        self.assertEqual(len(fallback_features), 16)
        self.assertEqual(fallback_features["audio_bpm_tempo"], DEFAULT_AUDIO_FEATURES["audio_bpm_tempo"])

    def test_03_text_extractor_heuristics(self):
        """3. Test de extracción textual con INFLESZ, Fernández-Huerta y VADER."""
        sample_text = (
            "Este es un video viral absolutamente increíble y fascinante sobre inteligencia artificial. "
            "Aprende cómo transformar tu estrategia hoy mismo sin complicaciones."
        )

        features = extract_text_features(sample_text)

        expected_numeric_keys = [
            "text_inflesz_score",
            "text_fernandez_huerta_score",
            "text_vader_compound",
            "text_word_count"
        ]

        for key in expected_numeric_keys:
            self.assertIn(key, features, f"Falta la métrica de texto: {key}")
            self.assertIsInstance(features[key], float)
            self.assertFalse(np.isnan(features[key]), f"Valor NaN detectado en {key}")

        self.assertIn("text_raw", features)
        self.assertIsInstance(features["text_raw"], str)
        self.assertGreater(features["text_word_count"], 5)

        # Probar fallback con texto vacío
        fallback_features = extract_text_features("")
        self.assertEqual(fallback_features["text_word_count"], 0.0)
        self.assertIn("text_inflesz_score", fallback_features)

    def test_04_virality_classifier_and_shap_explainer(self):
        """4. Test de inferencia con XGBoost y calibración de explicabilidad SHAP."""
        explainer = ViralityExplainer()

        # Generar vector completo de 25 características continuas + texto
        synthetic_features = {}
        for key in ALL_CONTINUOUS_FEATURE_NAMES:
            if "visual" in key:
                synthetic_features[key] = 120.0 if "brightness" in key else 0.4
            elif "audio" in key:
                synthetic_features[key] = 125.0 if "bpm" in key else 0.15
            else:
                synthetic_features[key] = 65.0 if "score" in key else 0.5
        synthetic_features["text_raw"] = "Acelerador definitivo de viralidad multimodal en redes"

        explanation = explainer.explain_prediction(synthetic_features)

        self.assertIn("probability", explanation)
        self.assertIn("is_viral_p90", explanation)
        self.assertIn("radar_scores", explanation)
        self.assertIn("top_positive_factors", explanation)
        self.assertIn("top_negative_factors", explanation)
        self.assertIn("recommendations", explanation)

        self.assertIsInstance(explanation["probability"], float)
        self.assertGreaterEqual(explanation["probability"], 0.0)
        self.assertLessEqual(explanation["probability"], 1.0)
        self.assertIsInstance(explanation["is_viral_p90"], bool)

        # Verificar ejes de radar 3D normalizados
        radar = explanation["radar_scores"]
        self.assertIn("visual", radar)
        self.assertIn("acoustic", radar)
        self.assertIn("textual", radar)
        for val in radar.values():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 100.0)

    def test_05_api_predict_virality_endpoint(self):
        """5. Test de endpoints de la API REST (/api/health y /api/predict-virality)."""
        # 1. Health check
        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        data_health = res_health.json()
        self.assertEqual(data_health["status"], "healthy")
        self.assertTrue(data_health.get("model_loaded", False))

        # 2. Predict con Preset Viral Hit
        res_viral = self.client.post("/api/predict-virality", json={"preset": "viral_hit"})
        self.assertEqual(res_viral.status_code, 200)
        data_viral = res_viral.json()
        self.assertTrue(data_viral["success"])
        self.assertIn("probability", data_viral)
        self.assertIn("radar_scores", data_viral)
        self.assertIn("top_positive_factors", data_viral)

        # 3. Predict con Preset Podcast
        res_podcast = self.client.post("/api/predict-virality", json={"preset": "podcast"})
        self.assertEqual(res_podcast.status_code, 200)
        self.assertTrue(res_podcast.json()["success"])

        # 4. Predict con texto directo
        res_text = self.client.post("/api/predict-virality", json={"text": "Nuevo video corto de tendencia"})
        self.assertEqual(res_text.status_code, 200)
        self.assertTrue(res_text.json()["success"])

        # 5. Servicio estático de Frontend en raíz
        res_index = self.client.get("/")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn("SonixPulse AI", res_index.text)


if __name__ == "__main__":
    unittest.main()
