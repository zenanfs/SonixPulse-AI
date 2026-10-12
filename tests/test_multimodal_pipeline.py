#!/usr/bin/env python3
"""
Suite unificada de pruebas para el pipeline multimodal de SonixPulse AI.
Verifica la extraccion de caracteristicas en memoria, inferencia con XGBoost,
traduccion comercial SHAP y los endpoints de la API REST con carga directa.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.visual_extractor import extract_visual_features, DEFAULT_VISUAL_FEATURES
from src.features.audio_extractor import extract_audio_features, DEFAULT_AUDIO_FEATURES
from src.features.text_extractor import extract_text_features, DEFAULT_TEXT_FEATURES
from src.features import ALL_CONTINUOUS_FEATURE_NAMES
from src.models.virality_explainer import ViralityExplainer
from src.models.commercial_translator import traducir_factores_shap, calcular_radar_comercial
from src.api.main import app
from fastapi.testclient import TestClient


class TestMultimodalPipeline(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_01_visual_extractor_synthetic(self):
        # Array sintetico RGB de 120x120
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        img[:60, :, 0] = 200
        img[60:, :, 1] = 180
        img[:, 40:80, 2] = 220

        features = extract_visual_features(img)

        metricas_esperadas = [
            "visual_brightness_mean",
            "visual_contrast_rms",
            "visual_color_entropy",
            "visual_saturation_mean",
            "visual_canny_edge_density"
        ]
        for metrica in metricas_esperadas:
            self.assertIn(metrica, features)
            self.assertIsInstance(features[metrica], float)
            self.assertFalse(np.isnan(features[metrica]))

        fallback = extract_visual_features(None)
        self.assertEqual(len(fallback), 5)
        self.assertEqual(fallback["visual_brightness_mean"], DEFAULT_VISUAL_FEATURES["visual_brightness_mean"])

    def test_02_audio_extractor_synthetic(self):
        sr = 22050
        duracion = 1.0
        t = np.linspace(0, duracion, int(sr * duracion), endpoint=False)
        senoidal = 0.5 * np.sin(2 * np.pi * 440 * t)

        features = extract_audio_features(senoidal, sr=sr)

        from src.features.audio_extractor import FEATURE_NAMES_AUDIO
        for metrica in FEATURE_NAMES_AUDIO:
            self.assertIn(metrica, features)
            self.assertIsInstance(features[metrica], float)
            self.assertFalse(np.isnan(features[metrica]))

        fallback = extract_audio_features(None)
        self.assertEqual(len(fallback), 16)
        self.assertEqual(fallback["audio_bpm_tempo"], DEFAULT_AUDIO_FEATURES["audio_bpm_tempo"])

    def test_03_text_extractor_heuristics(self):
        texto_muestra = "Nueva campana de lanzamiento con enfoque en estilo y autenticidad para redes."
        features = extract_text_features(texto_muestra)

        metricas_texto = [
            "text_inflesz_score",
            "text_fernandez_huerta_score",
            "text_vader_compound",
            "text_word_count"
        ]
        for metrica in metricas_texto:
            self.assertIn(metrica, features)
            self.assertIsInstance(features[metrica], float)

        self.assertIn("text_raw", features)
        self.assertGreater(features["text_word_count"], 5)

        fallback = extract_text_features("")
        self.assertEqual(fallback["text_word_count"], 0.0)

    def test_04_virality_classifier_and_shap_explainer(self):
        explainer = ViralityExplainer()

        # Vector sintetico completo de 25 variables continuas mas texto crudo
        muestra_features = {}
        for key in ALL_CONTINUOUS_FEATURE_NAMES:
            if "visual" in key:
                muestra_features[key] = 120.0 if "brightness" in key else 0.4
            elif "audio" in key:
                muestra_features[key] = 128.0 if "bpm" in key else 0.15
            else:
                muestra_features[key] = 65.0 if "score" in key else 0.5
        muestra_features["text_raw"] = "Lanzamiento exclusivo para plataformas digitales"

        explicacion = explainer.explain_prediction(muestra_features)

        self.assertIn("probabilidad", explicacion)
        self.assertIn("es_viral_p90", explicacion)
        self.assertIn("radar_comercial", explicacion)
        self.assertIn("factores_impulso", explicacion)
        self.assertIn("factores_friccion", explicacion)

        # Verificacion del radar de 5 ejes comerciales
        radar = explicacion["radar_comercial"]
        ejes_esperados = ["atractivo_visual", "energia_ritmo", "claridad_mensaje", "conexion_emocional", "retencion_estimada"]
        for eje in ejes_esperados:
            self.assertIn(eje, radar)
            self.assertGreaterEqual(radar[eje], 0.0)
            self.assertLessEqual(radar[eje], 100.0)

    def test_05_api_predict_virality_direct_upload(self):
        # 1. Health check
        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        data_health = res_health.json()
        self.assertEqual(data_health["estado"], "operativo")
        self.assertTrue(data_health["modelo_cargado"])

        # 2. Preset comercial
        res_preset = self.client.post("/api/predict-virality", data={"preset": "hit_ritmico"})
        self.assertEqual(res_preset.status_code, 200)
        data_preset = res_preset.json()
        self.assertTrue(data_preset["exito"])
        self.assertIn("radar_comercial", data_preset)
        self.assertIn("factores_impulso", data_preset)

        # 3. Carga directa con archivos temporales en memoria
        dummy_img = (
            b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff"
            b"\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00"
            b"\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
        )
        res_upload = self.client.post(
            "/api/predict-virality",
            files={"imagen": ("portada.gif", dummy_img, "image/gif")},
            data={"texto": "Campana visual de alto impacto con diseno moderno."}
        )
        self.assertEqual(res_upload.status_code, 200)
        data_upload = res_upload.json()
        self.assertTrue(data_upload["exito"])
        self.assertIn("radar_comercial", data_upload)

        # 4. Servidor estatico en raiz
        res_index = self.client.get("/")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn("SonixPulse AI", res_index.text)


if __name__ == "__main__":
    unittest.main()
