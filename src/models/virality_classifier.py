#!/usr/bin/env python3
"""
SonixPulse AI - Multimodal Virality Classifier (Phase 4)
========================================================
Entrena un clasificador XGBoost calibrado sobre el dataset multimodal maestro
para predecir la probabilidad de viralidad bajo el umbral P90 (90:10 desbalance).

Pipeline:
- ColumnTransformer:
  * StandardScaler para 25 características continuas (visuales, acústicas, texto).
  * TfidfVectorizer (500 n-gramas 1-2) para 'text_raw'.
- División estratificada 80/20 (stratify=y_clf, random_state=42).
- scale_pos_weight dinámico calculado en train set (~9.0).
- XGBClassifier (n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss').
- Serialización completa en models/virality_xgb_model.joblib.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Asegurar path raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.features import ALL_CONTINUOUS_FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SonixPulse.ViralityClassifier")

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "multimodal_master_dataset.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "virality_xgb_model.joblib"


def build_pipeline() -> ColumnTransformer:
    """
    Construye el ColumnTransformer multimodal:
    - 25 variables continuas normalizadas con StandardScaler.
    - 'text_raw' vectorizado con TF-IDF (500 max_features, n-gramas 1-2).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                ALL_CONTINUOUS_FEATURE_NAMES,
            ),
            (
                "text",
                TfidfVectorizer(max_features=500, ngram_range=(1, 2)),
                "text_raw",
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def train_virality_model(
    dataset_path: Path = DATASET_PATH,
    output_path: Path = MODEL_OUTPUT_PATH,
) -> Dict[str, Any]:
    """
    Carga el dataset maestro, ajusta el pipeline multimodal, entrena XGBClassifier
    con scale_pos_weight dinámico y serializa el artefacto.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset maestro no encontrado en: {dataset_path}")

    logger.info("Cargando dataset multimodal maestro desde: %s", dataset_path)
    df = pd.read_csv(dataset_path)

    # Validar columnas requeridas
    missing_cols = [c for c in ALL_CONTINUOUS_FEATURE_NAMES + ["text_raw", "y_clf"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing_cols}")

    X = df[ALL_CONTINUOUS_FEATURE_NAMES + ["text_raw"]]
    y = df["y_clf"].astype(int)

    logger.info("Dataset shape: %s | Distribución y_clf: %s", X.shape, dict(y.value_counts()))

    # División estratificada 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    logger.info("Split 80/20: Train=%d muestras, Test=%d muestras", len(y_train), len(y_test))

    # Preprocesamiento con ColumnTransformer
    preprocessor = build_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Cálculo dinámico de scale_pos_weight para desbalance P90 90:10
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    if pos_count == 0:
        raise ValueError("No hay ejemplos de la clase positiva (y_clf=1) en train set.")

    scale_pos_weight = float(neg_count / pos_count)
    logger.info(
        "Cálculo scale_pos_weight dinámico: %d / %d = %.4f",
        neg_count,
        pos_count,
        scale_pos_weight,
    )

    # Inicialización y entrenamiento de XGBClassifier
    classifier = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    logger.info("Entrenando XGBClassifier con n_estimators=100, max_depth=3, lr=0.1...")
    classifier.fit(X_train_proc, y_train)

    # Evaluación en Test Set
    y_test_pred = classifier.predict(X_test_proc)
    y_test_proba = classifier.predict_proba(X_test_proc)[:, 1]

    roc_auc = float(roc_auc_score(y_test, y_test_proba))
    f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
    precision = float(precision_score(y_test, y_test_pred, zero_division=0))
    recall = float(recall_score(y_test, y_test_pred, zero_division=0))

    logger.info("=" * 60)
    logger.info("EVALUACIÓN DEL MODELO VIRALITY XGB:")
    logger.info("ROC-AUC Score: %.4f", roc_auc)
    logger.info("F1-Score:      %.4f", f1)
    logger.info("Precision:     %.4f | Recall: %.4f", precision, recall)
    logger.info("Matriz de Confusión:\n%s", confusion_matrix(y_test, y_test_pred))
    logger.info("Reporte de Clasificación:\n%s", classification_report(y_test, y_test_pred, zero_division=0))
    logger.info("=" * 60)

    # Nombres de características generados por ColumnTransformer
    feature_names = preprocessor.get_feature_names_out().tolist()

    # Creación del artefacto serializable
    model_bundle = {
        "preprocessor": preprocessor,
        "classifier": classifier,
        "feature_names_continuous": list(ALL_CONTINUOUS_FEATURE_NAMES),
        "all_feature_names": feature_names,
        "metadata": {
            "scale_pos_weight": scale_pos_weight,
            "roc_auc": roc_auc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "total_samples": len(df),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "n_features_proc": len(feature_names),
            "random_state": 42,
        },
    }

    # Guardar en disco
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, output_path, compress=3)
    logger.info("Artefacto serializado exitosamente en: %s", output_path)

    return model_bundle


if __name__ == "__main__":
    train_virality_model()
