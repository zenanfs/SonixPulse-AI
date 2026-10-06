#!/usr/bin/env python3
"""
TrendScout - Automated Test Suite
=================================
Pruebas unitarias e integrales para verificar:
1. Ingesta de datos y validez de esquema CSV.
2. Modelo de clustering K-Means y matriz de dispersión.
3. Motor de ROI financiero y fórmulas de licencias.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Añadir el directorio raíz al path de Python
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.data_ingestion import SpotifyDataIngestion
from src.models.predictive_model import AcousticTrendPredictor
from src.models.financial_roi_engine import FinancialROIEngine

class TestTrendScoutPipeline(unittest.TestCase):

    def setUp(self):
        self.ingestion = SpotifyDataIngestion()
        self.predictor = AcousticTrendPredictor(n_clusters=4)
        self.financial_engine = FinancialROIEngine()

    def test_data_ingestion_schema(self):
        """Verifica que la ingesta devuelva las 50 canciones y las columnas requeridas."""
        df = self.ingestion.generate_synthetic_benchmark(50)
        self.assertEqual(len(df), 50)
        required_cols = ['track_name', 'artist_name', 'danceability', 'energy', 'valence', 'tempo', 'acousticness']
        for col in required_cols:
            self.assertIn(col, df.columns)

    def test_kmeans_clustering_convergence(self):
        """Verifica que K-Means genere etiquetas de cluster válidas entre 0 y 3."""
        df = self.ingestion.generate_synthetic_benchmark(50)
        X_scaled = self.predictor.preprocess_features(df)
        df_clustered, sil_score = self.predictor.train_clusters(df, X_scaled)
        
        self.assertIn('cluster_id', df_clustered.columns)
        unique_clusters = set(df_clustered['cluster_id'].unique())
        self.assertTrue(unique_clusters.issubset({0, 1, 2, 3}))
        self.assertGreaterEqual(sil_score, -1.0)

    def test_financial_roi_calculations(self):
        """Verifica que las fórmulas empíricas de costo de licencia y recall sean coherentes."""
        cost_hit = self.financial_engine.calculate_sync_licensing_cost(95, "RED_ALERT_SATURATED")
        cost_indie = self.financial_engine.calculate_sync_licensing_cost(60, "WHITE_SPACE_OPPORTUNITY")
        
        # El tema saturado debe costar significativamente más que el tema no saturado
        self.assertGreater(cost_hit, cost_indie)
        
        # Probar degradación de recall
        recall_saturated = self.financial_engine.calculate_brand_recall_delta(68.5)
        recall_virgin = self.financial_engine.calculate_brand_recall_delta(8.1)
        self.assertGreater(recall_virgin, recall_saturated)

if __name__ == '__main__':
    unittest.main()
