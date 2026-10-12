"""
SonixPulse AI - Tests unitarios para extractores de caracteristicas deterministicos en CPU.
Valida compatibilidad con rutas, arrays numpy, buffers en memoria (bytes y BytesIO) y fallbacks.
"""

import io
import unittest
import cv2
import numpy as np
import soundfile as sf

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
    count_syllables_word_spanish,
    split_sentences_spanish,
    DEFAULT_TEXT_FEATURES,
    FEATURE_NAMES_TEXT,
)
from src.features import ALL_CONTINUOUS_FEATURE_NAMES


class TestVisualExtractor(unittest.TestCase):
    def test_fallback_on_none_and_missing(self):
        res_none = extract_visual_features(None)
        self.assertEqual(res_none, DEFAULT_VISUAL_FEATURES)

        res_missing = extract_visual_features("ruta/falsa/inexistente.png")
        self.assertEqual(res_missing, DEFAULT_VISUAL_FEATURES)

    def test_numpy_array_input(self):
        img_uint8 = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        res = extract_visual_features(img_uint8)
        self.assertEqual(set(res.keys()), set(FEATURE_NAMES_VISUAL))
        for v in res.values():
            self.assertTrue(np.isfinite(v))

        # Array monocromatico 2D
        img_gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        res_gray = extract_visual_features(img_gray)
        self.assertEqual(set(res_gray.keys()), set(FEATURE_NAMES_VISUAL))

        # Array float [0, 1]
        img_float = np.random.rand(50, 50, 3).astype(np.float32)
        res_float = extract_visual_features(img_float)
        self.assertEqual(set(res_float.keys()), set(FEATURE_NAMES_VISUAL))

    def test_bytes_and_bytesio_input(self):
        raw_img = np.full((64, 64, 3), 120, dtype=np.uint8)
        cv2.circle(raw_img, (32, 32), 16, (255, 0, 0), -1)
        ok, encoded = cv2.imencode(".png", raw_img)
        self.assertTrue(ok)
        img_bytes = encoded.tobytes()

        # Entrada bytes
        res_bytes = extract_visual_features(img_bytes)
        self.assertEqual(set(res_bytes.keys()), set(FEATURE_NAMES_VISUAL))
        for v in res_bytes.values():
            self.assertTrue(np.isfinite(v))

        # Entrada BytesIO
        res_bio = extract_visual_features(io.BytesIO(img_bytes))
        self.assertEqual(set(res_bio.keys()), set(FEATURE_NAMES_VISUAL))


class TestAudioExtractor(unittest.TestCase):
    def test_fallback_on_none_and_short(self):
        res_none = extract_audio_features(None)
        self.assertEqual(res_none, DEFAULT_AUDIO_FEATURES)

        # Senal menor al minimo de 512 muestras
        short_sig = np.zeros(256, dtype=np.float32)
        res_short = extract_audio_features(short_sig)
        self.assertEqual(res_short, DEFAULT_AUDIO_FEATURES)

    def test_numpy_array_mono_and_stereo(self):
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig_mono = (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)

        res_mono = extract_audio_features(sig_mono, sr=sr)
        self.assertEqual(set(res_mono.keys()), set(FEATURE_NAMES_AUDIO))
        self.assertEqual(len(res_mono), 16)
        for v in res_mono.values():
            self.assertTrue(np.isfinite(v))

        # Estereo (2, N) y (N, 2)
        sig_stereo_ch_first = np.vstack([sig_mono, sig_mono * 0.5])
        res_ch_first = extract_audio_features(sig_stereo_ch_first, sr=sr)
        self.assertEqual(set(res_ch_first.keys()), set(FEATURE_NAMES_AUDIO))

        sig_stereo_samples_first = sig_stereo_ch_first.T
        res_samples_first = extract_audio_features(sig_stereo_samples_first, sr=sr)
        self.assertEqual(set(res_samples_first.keys()), set(FEATURE_NAMES_AUDIO))

    def test_bytes_and_bytesio_input(self):
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sig = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        bio = io.BytesIO()
        sf.write(bio, sig, sr, format="WAV")
        raw_bytes = bio.getvalue()

        # Entrada bytes
        res_bytes = extract_audio_features(raw_bytes, sr=sr)
        self.assertEqual(set(res_bytes.keys()), set(FEATURE_NAMES_AUDIO))
        self.assertEqual(len(res_bytes), 16)
        for v in res_bytes.values():
            self.assertTrue(np.isfinite(v))

        # Entrada BytesIO
        bio.seek(0)
        res_bio = extract_audio_features(bio, sr=sr)
        self.assertEqual(set(res_bio.keys()), set(FEATURE_NAMES_AUDIO))


class TestTextExtractor(unittest.TestCase):
    def test_fallback_on_none_and_empty(self):
        res_none = extract_text_features(None)
        self.assertEqual(res_none, DEFAULT_TEXT_FEATURES)

        res_empty = extract_text_features("   ")
        self.assertEqual(res_empty, DEFAULT_TEXT_FEATURES)

    def test_syllable_heuristics_spanish(self):
        test_cases = {
            "sol": 1,
            "pan": 1,
            "hola": 2,
            "cancion": 2,
            "canción": 2,
            "poesía": 4,
            "computadora": 5,
            "guau": 1,
            "paraguay": 3,
        }
        for word, expected in test_cases.items():
            self.assertEqual(
                count_syllables_word_spanish(word),
                expected,
                f"Fallo silabico en '{word}'",
            )

    def test_sentence_splitting_and_metrics(self):
        text = "¡Esto cambia todo! El beat mas viral del verano en TikTok."
        sentences = split_sentences_spanish(text)
        self.assertEqual(len(sentences), 2)

        res = extract_text_features(text)
        self.assertIn("text_inflesz_score", res)
        self.assertIn("text_fernandez_huerta_score", res)
        self.assertIn("text_vader_compound", res)
        self.assertIn("text_word_count", res)
        self.assertIn("text_raw", res)
        self.assertEqual(res["text_word_count"], 11.0)
        self.assertTrue(isinstance(res["text_raw"], str))


class TestUnifiedFeatureContract(unittest.TestCase):
    def test_total_continuous_features_count(self):
        self.assertEqual(len(FEATURE_NAMES_VISUAL), 5)
        self.assertEqual(len(FEATURE_NAMES_AUDIO), 16)
        self.assertEqual(len(FEATURE_NAMES_TEXT), 4)
        self.assertEqual(len(ALL_CONTINUOUS_FEATURE_NAMES), 25)
        self.assertEqual(
            ALL_CONTINUOUS_FEATURE_NAMES,
            FEATURE_NAMES_VISUAL + FEATURE_NAMES_AUDIO + FEATURE_NAMES_TEXT,
        )


if __name__ == "__main__":
    unittest.main()
