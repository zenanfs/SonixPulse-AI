#!/usr/bin/env python3
"""
SonixPulse AI - Real-Time Virality Tactical Explainer (Phase 4)
==============================================================
Módulo de explicabilidad e interpretabilidad en tiempo real mediante SHAP TreeExplainer
para el clasificador multimodal de viralidad P90 (XGBClassifier).

Funcionalidades:
- Carga del artefacto serializado `models/virality_xgb_model.joblib`.
- Compatibilidad transparente para serialización XGBoost 3.x / SHAP.
- Inferencia y cálculo de SHAP values deterministas por muestra.
- Extracción de aceleradores (top positivos) y fricciones (top negativos).
- Recomendaciones tácticas de optimización en español para creadores y marcas.
- Cálculo de scores normalizados (0-100) para los ejes Visual, Acústico y Textual
  diseñados para renderizado en el Radar 3D del Frontend.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Asegurar path raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import shap
import shap.explainers._tree as shap_tree

from src.features import (
    ALL_CONTINUOUS_FEATURE_NAMES,
    DEFAULT_VISUAL_FEATURES,
    DEFAULT_AUDIO_FEATURES,
    DEFAULT_TEXT_FEATURES,
    FEATURE_NAMES_VISUAL,
    FEATURE_NAMES_AUDIO,
    FEATURE_NAMES_TEXT,
)

logger = logging.getLogger("SonixPulse.ViralityExplainer")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "virality_xgb_model.joblib"

# Parche de compatibilidad para XGBoost 3.x / SHAP UBJSON base_score bracket parsing
_orig_decode_ubjson = getattr(shap_tree, "decode_ubjson_buffer", None)
if _orig_decode_ubjson is not None:
    def _patched_decode_ubjson(fd):
        jmodel = _orig_decode_ubjson(fd)
        try:
            bs = jmodel.get("learner", {}).get("learner_model_param", {}).get("base_score")
            if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                jmodel["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
        except Exception:
            pass
        return jmodel
    shap_tree.decode_ubjson_buffer = _patched_decode_ubjson


class ViralityExplainer:
    """
    Explicador táctico de viralidad multivariado en tiempo real basado en SHAP TreeExplainer.
    """

    # Diccionario de pautas tácticas por característica
    TACTICAL_GUIDELINES: Dict[str, Dict[str, str]] = {
        "visual_contrast_rms": {
            "pos": "Excelente rango dinámico y contraste visual; favorece la retención visual sostenida.",
            "neg": "Bajo contraste visual; incrementa la iluminación y la diferencia tonal entre planos iniciales."
        },
        "visual_saturation_mean": {
            "pos": "Paleta de color vibrante y saturada; genera mayor impacto en feeds móviles saturados.",
            "neg": "Colores deslavados o baja saturación; aplica gradación de color más viva y atractiva."
        },
        "visual_color_entropy": {
            "pos": "Alta riqueza cromática y variedad visual que estimula el procesamiento sensorial.",
            "neg": "Composición cromática monótona; introduce mayor diversidad de elementos en el encuadre."
        },
        "visual_brightness_mean": {
            "pos": "Nivel de luminosidad óptimo para visualización nítida en pantallas OLED/LCD.",
            "neg": "Tomas oscuras o subexpuestas; ajusta la exposición para evitar abandono en los primeros segundos."
        },
        "visual_canny_edge_density": {
            "pos": "Densidad de bordes y texturas equilibrada; nitidez visual sobresaliente.",
            "neg": "Imagen borrosa o con bajo nivel de detalle; mejora el foco y la definición visual."
        },
        "audio_rms_mean": {
            "pos": "Presión acústica (RMS/Loudness) competitiva con los estándares virales de TikTok/Reels.",
            "neg": "Volumen percibido bajo; masteriza el audio a -14 a -9 LUFS para mayor presencia sonora."
        },
        "audio_bpm": {
            "pos": "Tempo rítmico energizante en la ventana ideal de engagement y repetición.",
            "neg": "Tempo desfasado del rango óptimo (115-130 BPM); ajusta el beat para sincronización coreográfica."
        },
        "audio_bpm_tempo": {
            "pos": "Tempo rítmico energizante en la ventana ideal de engagement y repetición (115-130 BPM).",
            "neg": "Tempo desfasado del rango óptimo (115-130 BPM); ajusta el beat para sincronización coreográfica."
        },
        "audio_zcr_mean": {
            "pos": "Excelente brillo en frecuencias agudas y transitorios rítmicos marcados.",
            "neg": "Transitorios opacos; ecualiza o resalta elementos percusivos y ataques sonoros."
        },
        "audio_mfcc_1": {
            "pos": "Resonancia y balance espectral de graves bien ecualizado.",
            "neg": "Desbalance en frecuencias graves; ecualiza la pegada del bajo/kick para mayor impacto."
        },
        "audio_mfcc_2": {
            "pos": "Cuerpo medio-grave consistente que da calidez y peso al track.",
            "neg": "Falta de definición en medios; refuerza la presencia del instrumento principal o voz."
        },
        "text_inflesz_score": {
            "pos": "Gancho textual altamente accesible y de comprensión instantánea.",
            "neg": "Gancho con sintaxis densa; simplifica las oraciones para lectura rápida en pantallas pequeñas."
        },
        "text_fernandez_huerta_score": {
            "pos": "Estructura lingüística fluida y directa que engancha de inmediato al usuario.",
            "neg": "Texto difícil de digerir; reduce palabras complejas y usa frases contundentes."
        },
        "text_vader_compound": {
            "pos": "Carga emocional polarizada que estimula la curiosidad y la interacción de la audiencia.",
            "neg": "Tono neutral o apagado; añade emoción, intriga o un llamado a la acción más contundente."
        },
        "text_vader_pos": {
            "pos": "Afinidad de emoción positiva que fomenta el compartir y la empatía social.",
            "neg": "Escaso optimismo o entusiasmo percibido en el hook de texto."
        },
    }

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el artefacto del modelo en: {self.model_path}. "
                "Ejecuta previamente `python src/models/virality_classifier.py`."
            )

        logger.info("Cargando modelo serializado desde: %s", self.model_path)
        bundle = joblib.load(self.model_path)

        self.preprocessor = bundle["preprocessor"]
        self.classifier = bundle["classifier"]
        self.feature_names_continuous = bundle.get("feature_names_continuous", list(ALL_CONTINUOUS_FEATURE_NAMES))
        self.all_feature_names = bundle.get("all_feature_names", [])
        self.metadata = bundle.get("metadata", {})

        # Inicialización de SHAP TreeExplainer
        logger.info("Calibrando SHAP TreeExplainer con XGBClassifier...")
        self.explainer = shap.TreeExplainer(self.classifier)
        logger.info("ViralityExplainer listo con %d características.", len(self.all_feature_names))

    def _normalize_input(self, features_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Garantiza que el diccionario de entrada contenga todas las 25 variables
        continuas y 'text_raw', rellenando valores faltantes con defaults predeterminados.
        """
        row: Dict[str, Any] = {}

        # Default fallback continuo
        default_continuous = {
            **DEFAULT_VISUAL_FEATURES,
            **DEFAULT_AUDIO_FEATURES,
            **DEFAULT_TEXT_FEATURES,
        }

        for feat in self.feature_names_continuous:
            val = features_dict.get(feat, default_continuous.get(feat, 0.0))
            try:
                row[feat] = float(val) if val is not None else 0.0
            except (ValueError, TypeError):
                row[feat] = float(default_continuous.get(feat, 0.0))

        # Texto crudo
        raw_text = features_dict.get("text_raw", "")
        row["text_raw"] = str(raw_text) if raw_text is not None else ""

        return pd.DataFrame([row])

    def _generate_tactical_recommendation(self, feature_name: str, shap_val: float) -> str:
        """
        Traduce el impacto SHAP de una característica a una recomendación táctica accionable en español.
        """
        clean_name = feature_name.replace("num__", "")
        guideline = self.TACTICAL_GUIDELINES.get(clean_name)
        if guideline:
            return guideline["pos"] if shap_val > 0 else guideline["neg"]

        # Si es una característica de texto TF-IDF
        if feature_name.startswith("text__"):
            token = feature_name.replace("text__", "")
            if shap_val > 0:
                return f"El gancho o término '{token}' aporta alta tracción y engagement en la audiencia."
            else:
                return f"El término o combinación '{token}' reduce el gancho viral; reformula o sustituye por sinónimos de mayor impacto."

        # Timbre acústico MFCC
        if clean_name.startswith("audio_mfcc_"):
            mfcc_num = clean_name.replace("audio_mfcc_", "")
            if shap_val > 0:
                return f"Timbre espectral equilibrado en banda armónica MFCC-{mfcc_num}; sonoridad atractiva para la audiencia."
            else:
                return f"Ajustar ecualización en banda espectral MFCC-{mfcc_num}; limpiar frecuencias resonantes que saturan la mezcla."

        # Genérico para otras métricas continuas
        if shap_val > 0:
            return f"Parámetro {clean_name} aporta impulso positivo hacia la viralidad."
        else:
            return f"Optimizar parámetro {clean_name} para reducir la fricción en la retención."

    def explain_prediction(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explica la predicción de viralidad para una pieza de contenido dada.

        Parámetros:
        - features_dict: Diccionario conteniendo las 25 variables continuas y 'text_raw'.

        Retorna:
        - probability: Probabilidad predicha de pertenecer al P90 viral [0.0, 1.0].
        - is_viral_p90: Booleano de clasificación (True si >= 0.5).
        - top_positive_factors: Lista de los 5 factores con mayor impacto SHAP positivo.
        - top_negative_factors: Lista de los 5 factores con mayor impacto SHAP negativo.
        - radar_scores: Diccionario con puntajes normalizados (0-100) para Visual, Acoustic y Textual.
        - all_shap_contributions: Diccionario mapeando cada feature con su valor SHAP.
        - recommendations: Lista de recomendaciones tácticas generadas en español.
        """
        # Preparación del dato
        df_input = self._normalize_input(features_dict)
        X_proc = self.preprocessor.transform(df_input)

        # Inferencia de probabilidad
        proba_array = self.classifier.predict_proba(X_proc)
        probability = float(proba_array[0, 1])
        is_viral = bool(probability >= 0.5)

        # Cálculo de valores SHAP
        shap_explanation = self.explainer(X_proc)
        shap_vals = np.asarray(shap_explanation.values)[0]  # vector 1D

        # Nombres de variables
        names = self.all_feature_names if len(self.all_feature_names) == len(shap_vals) else [
            f"feat_{i}" for i in range(len(shap_vals))
        ]

        # Consolidar contribuciones
        all_contributions: Dict[str, float] = {}
        contributions_list: List[Dict[str, Any]] = []

        visual_shap_sum = 0.0
        acoustic_shap_sum = 0.0
        textual_shap_sum = 0.0

        for feat, val in zip(names, shap_vals):
            val_f = float(val)
            all_contributions[feat] = round(val_f, 5)
            contributions_list.append({
                "feature": feat,
                "clean_name": feat.replace("num__", "").replace("text__", ""),
                "shap_value": round(val_f, 5),
                "impact": "positive" if val_f > 0 else "negative",
            })

            # Agregación por eje sensorial
            if "visual" in feat:
                visual_shap_sum += val_f
            elif "audio" in feat:
                acoustic_shap_sum += val_f
            elif "text" in feat:
                textual_shap_sum += val_f

        # Ordenar factores
        sorted_positives = [item for item in contributions_list if item["shap_value"] > 0]
        sorted_positives.sort(key=lambda x: x["shap_value"], reverse=True)

        sorted_negatives = [item for item in contributions_list if item["shap_value"] < 0]
        sorted_negatives.sort(key=lambda x: x["shap_value"])

        top_positive = sorted_positives[:5]
        top_negative = sorted_negatives[:5]

        # Generación de recomendaciones tácticas legibles
        recommendations: List[str] = []
        for item in top_negative[:3]:
            rec = self._generate_tactical_recommendation(item["feature"], item["shap_value"])
            recommendations.append(f"[Fricción táctica] {rec}")

        for item in top_positive[:2]:
            rec = self._generate_tactical_recommendation(item["feature"], item["shap_value"])
            recommendations.append(f"[Fortaleza clave] {rec}")

        # Cálculo de Radar Scores (0 a 100) para los ejes Visual, Acústico y Textual
        # Base neutral en 50.0; la escala logística mapea [-3, 3] a [5, 95] limpiamente
        def _calc_axis_score(shap_sum: float) -> float:
            score = 100.0 / (1.0 + np.exp(-1.5 * shap_sum))
            return round(float(np.clip(score, 5.0, 98.0)), 1)

        radar_scores = {
            "visual": _calc_axis_score(visual_shap_sum),
            "acoustic": _calc_axis_score(acoustic_shap_sum),
            "textual": _calc_axis_score(textual_shap_sum),
        }

        return {
            "probability": round(probability, 4),
            "is_viral_p90": is_viral,
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative,
            "radar_scores": radar_scores,
            "all_shap_contributions": all_contributions,
            "recommendations": recommendations,
        }


if __name__ == "__main__":
    # Test sintético local si el modelo ya está entrenado
    print("Inicializando ViralityExplainer...")
    explainer = ViralityExplainer()

    # Muestra sintética representativa
    sample_data = {
        "visual_brightness_mean": 128.5,
        "visual_contrast_rms": 45.2,
        "visual_color_entropy": 6.8,
        "visual_saturation_mean": 95.0,
        "visual_canny_edge_density": 0.08,
        "audio_rms_mean": 0.22,
        "audio_zcr_mean": 0.09,
        "audio_bpm": 124.0,
        "audio_mfcc_1": -180.0,
        "audio_mfcc_2": 95.0,
        "audio_mfcc_3": -12.0,
        "audio_mfcc_4": 18.0,
        "audio_mfcc_5": 5.0,
        "audio_mfcc_6": -8.0,
        "audio_mfcc_7": 12.0,
        "audio_mfcc_8": -4.0,
        "audio_mfcc_9": 6.0,
        "audio_mfcc_10": -2.0,
        "audio_mfcc_11": 4.0,
        "audio_mfcc_12": -1.0,
        "audio_mfcc_13": 3.0,
        "text_inflesz_score": 72.5,
        "text_fernandez_huerta_score": 75.0,
        "text_vader_compound": 0.65,
        "text_vader_pos": 0.35,
        "text_raw": "¡ESTO CAMBIA TODO! El nuevo drop viral con energía insuperable",
    }

    result = explainer.explain_prediction(sample_data)
    print("\n--- RESULTADO DE EXPLICABILIDAD VIRAL ---")
    print(f"Probabilidad Viral P90: {result['probability'] * 100:.1f}% | ¿Es Viral?: {result['is_viral_p90']}")
    print(f"Radar Scores 3D: {result['radar_scores']}")
    print(f"Top 3 Factores Positivos: {[f['feature'] for f in result['top_positive_factors'][:3]]}")
    print(f"Top 3 Factores Negativos: {[f['feature'] for f in result['top_negative_factors'][:3]]}")
    print("Recomendaciones Tácticas:")
    for rec in result["recommendations"]:
        print(f" - {rec}")
