"""
Pruebas unitarias para el modulo de traduccion comercial y radar de 5 ejes.
Verifica cumplimiento estricto de directrices de estilo y consistencia matematica.
"""

import unittest
from src.models.commercial_translator import (
    get_commercial_name,
    compute_commercial_radar,
    generate_business_recommendation,
    format_commercial_factors,
    COMMERCIAL_FEATURE_NAMES,
)
from src.models.virality_explainer import ViralityExplainer


class TestCommercialTranslator(unittest.TestCase):
    """
    Suite de validacion para la capa de interpretacion de negocio.
    """

    def setUp(self):
        self.prohibited_words = ["crucial", "potenciar", "desentrañar", "desentranar", "tapiz"]

    def _assert_no_emojis(self, text: str):
        for char in text:
            # Rango unicode tipico de emojis
            self.assertFalse(
                0x1F300 <= ord(char) <= 0x1FAD6 or 0x1F600 <= ord(char) <= 0x1F64F,
                f"Emoji detectado en cadena: {text}"
            )

    def test_01_feature_name_mapping(self):
        """Valida mapeo de nombres tecnicos a terminologia comercial."""
        self.assertEqual(get_commercial_name("visual_contrast_rms"), "Contraste Visual")
        self.assertEqual(get_commercial_name("visual_brightness_mean"), "Luminosidad de Portada")
        self.assertEqual(get_commercial_name("visual_color_entropy"), "Variedad Visual")
        self.assertEqual(get_commercial_name("visual_saturation_mean"), "Intensidad del Color")
        self.assertEqual(get_commercial_name("visual_canny_edge_density"), "Detalle y Nitidez")
        self.assertEqual(get_commercial_name("audio_rms_mean"), "Fuerza del Audio")
        self.assertEqual(get_commercial_name("audio_zcr_mean"), "Presencia de Frecuencias Altas")
        self.assertEqual(get_commercial_name("audio_bpm_tempo"), "Ritmo y Velocidad Musical")
        self.assertEqual(get_commercial_name("num__audio_mfcc_3"), "Complejidad Timbrica")
        self.assertEqual(get_commercial_name("text_inflesz_score"), "Fluidez de Lectura")
        self.assertEqual(get_commercial_name("text_fernandez_huerta_score"), "Sencillez del Texto")
        self.assertEqual(get_commercial_name("text_vader_compound"), "Impacto Emocional")
        self.assertEqual(get_commercial_name("text_word_count"), "Longitud del Mensaje")
        self.assertEqual(get_commercial_name("text__ritmo"), "Gancho en Texto: ritmo")

    def test_02_radar_5_axes_bounds(self):
        """Valida que los 5 ejes del radar comercial esten acotados entre 0 y 100."""
        features = {
            "visual_contrast_rms": 52.0,
            "visual_saturation_mean": 110.0,
            "visual_color_entropy": 6.2,
            "visual_canny_edge_density": 0.10,
            "audio_rms_mean": 0.18,
            "audio_bpm_tempo": 124.0,
            "audio_zcr_mean": 0.07,
            "text_inflesz_score": 75.0,
            "text_fernandez_huerta_score": 72.0,
            "text_vader_compound": 0.60,
            "text_word_count": 14.0,
        }

        radar = compute_commercial_radar(features)
        expected_axes = [
            "atractivo_visual",
            "energia_ritmo",
            "claridad_mensaje",
            "conexion_emocional",
            "retencion_estimada",
        ]

        for axis in expected_axes:
            self.assertIn(axis, radar)
            val = radar[axis]
            self.assertIsInstance(val, float)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 100.0)

    def test_03_recommendations_human_style(self):
        """Valida ausencia de emojis y cliches de IA en las recomendaciones."""
        features_to_test = [
            ("visual_contrast_rms", 1.2),
            ("visual_contrast_rms", -0.8),
            ("audio_bpm_tempo", -1.5),
            ("audio_rms_mean", -0.5),
            ("text_vader_compound", 0.7),
            ("text__sorpresa", -0.3),
        ]

        for feat, val in features_to_test:
            rec = generate_business_recommendation(feat, val)
            self._assert_no_emojis(rec)
            for bad in self.prohibited_words:
                self.assertNotIn(bad, rec.lower(), f"Cliche prohibido '{bad}' en: {rec}")

    def test_04_explainer_integration(self):
        """Valida integracion de ViralityExplainer con el esquema comercial."""
        explainer = ViralityExplainer()
        sample = {
            "visual_contrast_rms": 40.0,
            "visual_brightness_mean": 120.0,
            "visual_saturation_mean": 80.0,
            "audio_rms_mean": 0.15,
            "audio_bpm_tempo": 128.0,
            "text_raw": "anuncio promocional directo",
        }

        res = explainer.explain_prediction(sample)

        # Claves requeridas por la fase 3
        self.assertIn("probabilidad", res)
        self.assertIn("es_viral_p90", res)
        self.assertIn("radar_comercial", res)
        self.assertIn("factores_impulso", res)
        self.assertIn("factores_friccion", res)

        self.assertIsInstance(res["probabilidad"], float)
        self.assertIsInstance(res["es_viral_p90"], bool)
        self.assertIsInstance(res["radar_comercial"], dict)
        self.assertEqual(len(res["radar_comercial"]), 5)

        for factor in res["factores_impulso"] + res["factores_friccion"]:
            self.assertIn("caracteristica", factor)
            self.assertIn("nombre_comercial", factor)
            self.assertIn("impacto", factor)
            self.assertIn("recomendacion", factor)
            self._assert_no_emojis(factor["recomendacion"])
            for bad in self.prohibited_words:
                self.assertNotIn(bad, factor["recomendacion"].lower())


if __name__ == "__main__":
    unittest.main()
