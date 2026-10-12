"""
SonixPulse AI - Predictive Models and Explainability Package
============================================================
Módulos de modelado predictivo, optimización financiera y explicabilidad táctica:
- ViralityExplainer: Interpretabilidad SHAP multivariada en tiempo real (Visual, Acústico, Texto).
- train_virality_model, build_pipeline: Pipeline de clasificación de viralidad P90 (XGBoost).
- AcousticTrendPredictor: Modelado no supervisado de saturación de mercado y clusters K-Means.
- FinancialROIEngine: Motor econométrico de ROI y licencias sincronizadas.
"""

from src.models.virality_classifier import (
    train_virality_model,
    build_pipeline,
)
from src.models.virality_explainer import (
    ViralityExplainer,
)
from src.models.commercial_translator import (
    get_commercial_name,
    compute_commercial_radar,
    calcular_radar_comercial,
    generate_business_recommendation,
    format_commercial_factors,
    traducir_factores_shap,
    obtener_nombre_comercial,
    COMMERCIAL_FEATURE_NAMES,
)
from src.models.predictive_model import (
    AcousticTrendPredictor,
)
from src.models.financial_roi_engine import (
    FinancialROIEngine,
)

__all__ = [
    "train_virality_model",
    "build_pipeline",
    "ViralityExplainer",
    "get_commercial_name",
    "compute_commercial_radar",
    "calcular_radar_comercial",
    "generate_business_recommendation",
    "format_commercial_factors",
    "traducir_factores_shap",
    "obtener_nombre_comercial",
    "COMMERCIAL_FEATURE_NAMES",
    "AcousticTrendPredictor",
    "FinancialROIEngine",
]

