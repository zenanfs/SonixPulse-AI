#!/usr/bin/env python3
"""
SonixPulse AI - Tests for Multimodal Ingestion and Master Dataset Builder (Phase 3)
===================================================================================
Pruebas automatizadas para validar:
1. Resiliencia de la ingesta efímera (yt-dlp, ffmpeg, fallback transparente).
2. Estructura y nombres de características retornadas.
3. Generación determinista del dataset maestro calibrado P90.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import ALL_CONTINUOUS_FEATURE_NAMES
from src.features.dataset_builder import build_multimodal_dataset, DEFAULT_OUTPUT_PATH
from src.ingestion.multimodal_ingestion import MultimodalIngestor, ingest_multimodal_url, find_ffmpeg


class TestMultimodalIngestionAndDataset(unittest.TestCase):

    def test_ffmpeg_detection(self):
        """Verifica que find_ffmpeg localice la instalación de FFmpeg del sistema."""
        ffmpeg_bin = find_ffmpeg()
        self.assertIsNotNone(ffmpeg_bin, "FFmpeg debe ser detectado en Chocolatey o en PATH")
        self.assertTrue(Path(ffmpeg_bin).exists())

    def test_ephemeral_ingestion_fallback_transparent(self):
        """Verifica que URLs inaccesibles o privadas activen el fallback sin lanzar excepciones."""
        ingestor = MultimodalIngestor()
        result = ingestor.ingest_url("https://invalid-video-url-for-test.example")

        self.assertIn("features", result)
        self.assertIn("metadata", result)

        meta = result["metadata"]
        self.assertTrue(meta["fallback_applied"])
        self.assertIsNotNone(meta["fallback_reason"])

        feats = result["features"]
        self.assertIn("text_raw", feats)

        # Verificar que contiene exactamente las 25 características continuas
        for col in ALL_CONTINUOUS_FEATURE_NAMES:
            self.assertIn(col, feats, f"Característica {col} no encontrada en features")

        self.assertEqual(len(ALL_CONTINUOUS_FEATURE_NAMES), 25)

    def test_master_dataset_builder_schema_and_p90_calibration(self):
        """Verifica la generación determinista del dataset y el corte exacto P90 (90:10)."""
        df = build_multimodal_dataset(n_samples=200, seed=42, output_path=None)

        self.assertEqual(len(df), 200)
        self.assertEqual(len(df.columns), 27)  # 25 continuas + text_raw + y_clf

        # Verificar ausencia de valores nulos o infinitos
        self.assertEqual(df.isnull().sum().sum(), 0)

        # Verificar calibración P90 exacta
        counts = df["y_clf"].value_counts().to_dict()
        self.assertEqual(counts.get(1), 20, "Exactamente 20 muestras deben ser clase positiva (10%)")
        self.assertEqual(counts.get(0), 180, "Exactamente 180 muestras deben ser clase negativa (90%)")

        # Verificar columnas continuas
        for col in ALL_CONTINUOUS_FEATURE_NAMES:
            self.assertIn(col, df.columns)
            self.assertTrue(np.issubdtype(df[col].dtype, np.number))

        self.assertIn("text_raw", df.columns)
        self.assertIn("y_clf", df.columns)

    def test_master_dataset_csv_file_persistence(self):
        """Verifica que el archivo CSV exista en data/processed/multimodal_master_dataset.csv."""
        self.assertTrue(DEFAULT_OUTPUT_PATH.exists())
        df = pd.read_csv(DEFAULT_OUTPUT_PATH)
        self.assertEqual(len(df), 200)
        self.assertEqual(len(df.columns), 27)
        self.assertEqual((df["y_clf"] == 1).sum(), 20)
        self.assertEqual((df["y_clf"] == 0).sum(), 180)


if __name__ == "__main__":
    unittest.main()
