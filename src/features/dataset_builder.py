#!/usr/bin/env python3
"""
SonixPulse AI - Multimodal Master Dataset Builder
=================================================
Generador determinista del dataset maestro multimodal calibrado según el umbral P90
(10% de clase positiva y_clf=1, 90% de clase negativa y_clf=0).

Estructura del Dataset:
- Exactamente las 25 características continuas definidas en `ALL_CONTINUOUS_FEATURE_NAMES`
  (5 visuales, 16 acústicas, 4 lingüísticas).
- Columna 'text_raw' con ganchos y títulos variados en español.
- Columna objetivo 'y_clf' (int: 0 o 1).

Calibración P90:
Modela un score latente de viralidad continuo multi-sensorial combinando:
- Eje Visual: Contraste RMS, saturación HSV, entropía de color y densidad de bordes.
- Eje Acústico: Nivel RMS medio (loudness), tempo BPM en zona bailable y MFCCs.
- Eje Lingüístico: Intensidad afectiva VADER, legibilidad INFLESZ y longitud óptima.
El umbral de corte P90 garantiza que exactamente el 10% de los datos represente hits virales.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features import (
    ALL_CONTINUOUS_FEATURE_NAMES,
    FEATURE_NAMES_VISUAL,
    FEATURE_NAMES_AUDIO,
    FEATURE_NAMES_TEXT,
    extract_text_features,
)

logger = logging.getLogger("SonixPulse.DatasetBuilder")

DEFAULT_OUTPUT_PATH = Path("data/processed/multimodal_master_dataset.csv")

# Banco representativo de ganchos y títulos en español clasificados por intención comunicativa
SPANISH_TITLES_POOL: List[str] = [
    # Ganchos de alta viralidad / curiosidad / emoción
    "¡ESTO CAMBIA TODO! El truco secreto que los productores no quieren que sepas",
    "¿Por qué todos están obsesionados con este nuevo beat en TikTok?",
    "El drop más potente del año: Escúchalo con audífonos al 100%",
    "No cometas este error si quieres que tu contenido explote en redes",
    "Cuando la base entra justo a tiempo y la energía de la sala se descontrola",
    "Los 3 secretos definitivos para sonar como un artista de grandes ligas",
    "¡Increíble temazo viral que no vas a poder parar de cantar todo el día!",
    "Si te sabes este trend completo tienes el nivel de energía más alto",
    "La colaboración más esperada del año por fin vio la luz en plataformas",
    "Probando el audio más viral de la semana con mis mejores amigos",
    "Cómo logré más de un millón de reproducciones con una sola melodía",
    "¡Alerta de temazo! La canción que va a dominar el verano en todas las discos",
    "Bailando el trend latino del momento bajo la lluvia en Quito",
    "El secreto de los 808s que hace retumbar cualquier bocina",
    "Reaccionando por primera vez al estreno musical más viral de la temporada",
    "Esto es lo que pasa cuando mezclas reggaetón clásico con sintetizadores del futuro",
    "¡Nadie vio venir este remix! Escucha la versión extendida ya mismo",
    "El coro más pegajoso que vas a escuchar en todo el mes de octubre",
    "Reto viral: Intenta no bailar con los primeros cinco segundos de este tema",
    "Desde el estudio: Creando un éxito urbano en menos de diez minutos",

    # Publicaciones estándar / promocionales / mid-tier
    "Nuevo sencillo disponible en todas las plataformas digitales de streaming",
    "Detrás de cámaras durante la filmación oficial del videoclip musical",
    "Sesión de grabación acústica en vivo desde nuestro estudio independiente",
    "Agradeciendo todo el apoyo de la comunidad en esta nueva etapa creativa",
    "Explorando nuevas progresiones de acordes en teclado analógico",
    "Lanzamiento del tercer corte promocional del álbum de estudio",
    "Entrevista exclusiva hablando sobre el proceso de composición y letras",
    "Ensayando el repertorio en vivo antes de la gira de conciertos nacional",
    "Un vistazo íntimo a la producción de voces y afinación en cabina",
    "Compartiendo un adelanto instrumental antes del lanzamiento oficial",
    "Remasterización digital de clásicos del pop latino para coleccionistas",
    "Música relajante para concentrarse y trabajar durante largas jornadas",
    "Sesión de preguntas y respuestas sobre el futuro de la banda",
    "Consejos prácticos para mezclar guitarras acústicas en proyectos caseros",
    "Celebrando el primer aniversario de nuestro primer EP independiente",

    # Contenido neutral / técnico / archivo institucional
    "Resumen técnico del balance acústico y respuesta de frecuencia en recintos",
    "Informe de métricas mensuales sobre retención de usuarios en transmisiones",
    "Prueba de calibración estandarizada para micrófonos de condensador",
    "Registro sonoro número doce archivado para análisis de ruido de fondo",
    "Documentación metodológica sobre algoritmos de análisis espectral digital",
    "Lectura continua de las actas de reunión del comité de evaluación",
    "Pruebas de latencia en la transmisión de paquetes de audio multicanal",
    "Ensayo técnico sobre dispersión sonora en auditorios de planta rectangular",
    "Registro documental de mediciones de presión sonora en decibeles ponderados",
    "Informe descriptivo de procedimientos operativos para digitalización de cintas",
]


def build_multimodal_dataset(
    n_samples: int = 200,
    seed: int = 42,
    output_path: Optional[Path] = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Construye determinísticamente el dataset maestro multimodal calibrado con P90.

    Args:
        n_samples: Número de filas a generar (mínimo recomendado: 150-200).
        seed: Semilla para el generador pseudoaleatorio de NumPy.
        output_path: Ruta destino para guardar el archivo CSV.

    Returns:
        pd.DataFrame con exactamente las 25 características continuas, 'text_raw' y 'y_clf'.
    """
    if n_samples < 50:
        raise ValueError(f"n_samples debe ser al menos 50, recibido: {n_samples}")

    rng = np.random.default_rng(seed=seed)

    # 1. GENERACIÓN DETERMINISTA DE CARACTERÍSTICAS VISUALES (5)
    visual_brightness = rng.normal(loc=132.0, scale=28.0, size=n_samples).clip(25.0, 245.0)
    visual_contrast = rng.normal(loc=54.0, scale=16.0, size=n_samples).clip(15.0, 115.0)
    visual_entropy = rng.normal(loc=5.3, scale=0.75, size=n_samples).clip(2.5, 7.8)
    visual_saturation = rng.normal(loc=92.0, scale=34.0, size=n_samples).clip(15.0, 230.0)
    visual_canny = (rng.beta(a=2.0, b=15.0, size=n_samples) * 0.35 + 0.02).clip(0.01, 0.35)

    # 2. GENERACIÓN DETERMINISTA DE CARACTERÍSTICAS ACÚSTICAS (16)
    audio_rms = rng.normal(loc=0.11, scale=0.038, size=n_samples).clip(0.02, 0.32)
    audio_zcr = rng.normal(loc=0.058, scale=0.018, size=n_samples).clip(0.01, 0.22)

    # Distribución realista de tempo con picos en rangos bailables de pop y urbano (95 - 130 BPM)
    tempo_anchors = [95.0, 100.0, 108.0, 116.0, 120.0, 124.0, 128.0, 132.0, 85.0]
    tempo_probs = [0.18, 0.16, 0.14, 0.14, 0.16, 0.08, 0.06, 0.04, 0.04]
    base_tempo = rng.choice(tempo_anchors, size=n_samples, p=tempo_probs)
    audio_tempo = (base_tempo + rng.normal(0.0, 1.8, size=n_samples)).clip(60.0, 180.0)

    # 13 Coeficientes MFCC basados en perfiles acústicos típicos
    mfcc_1 = rng.normal(loc=-165.0, scale=40.0, size=n_samples)
    mfcc_2 = rng.normal(loc=95.0, scale=22.0, size=n_samples)
    mfcc_3 = rng.normal(loc=-16.0, scale=16.0, size=n_samples)
    mfcc_4 = rng.normal(loc=26.0, scale=13.0, size=n_samples)
    mfcc_5 = rng.normal(loc=-4.0, scale=11.0, size=n_samples)
    mfcc_6 = rng.normal(loc=12.0, scale=9.5, size=n_samples)
    mfcc_7 = rng.normal(loc=-8.0, scale=8.5, size=n_samples)
    mfcc_8 = rng.normal(loc=6.0, scale=7.5, size=n_samples)
    mfcc_9 = rng.normal(loc=-4.5, scale=6.5, size=n_samples)
    mfcc_10 = rng.normal(loc=3.2, scale=5.5, size=n_samples)
    mfcc_11 = rng.normal(loc=-2.1, scale=4.8, size=n_samples)
    mfcc_12 = rng.normal(loc=1.5, scale=4.2, size=n_samples)
    mfcc_13 = rng.normal(loc=-1.2, scale=3.8, size=n_samples)

    # 3. GENERACIÓN DE TEXTO CRUDO Y CARACTERÍSTICAS LINGÜÍSTICAS (4)
    # Muestreo con reemplazo del banco diverso en español
    text_pool_indices = rng.integers(low=0, high=len(SPANISH_TITLES_POOL), size=n_samples)
    texts_raw = [SPANISH_TITLES_POOL[idx] for idx in text_pool_indices]

    # Extracción real y coherente de características para cada texto
    inflesz_scores = []
    huerta_scores = []
    vader_compounds = []
    word_counts = []

    for text in texts_raw:
        feats = extract_text_features(text)
        inflesz_scores.append(feats["text_inflesz_score"])
        huerta_scores.append(feats["text_fernandez_huerta_score"])
        vader_compounds.append(feats["text_vader_compound"])
        word_counts.append(feats["text_word_count"])

    inflesz_arr = np.array(inflesz_scores, dtype=np.float64)
    huerta_arr = np.array(huerta_scores, dtype=np.float64)
    vader_arr = np.array(vader_compounds, dtype=np.float64)
    words_arr = np.array(word_counts, dtype=np.float64)

    # 4. MODELADO DEL SCORE LATENTE DE VIRALIDAD Y CALIBRACIÓN P90
    # Función auxiliar para estandarización z-score segura
    def _standardize(arr: np.ndarray) -> np.ndarray:
        s = np.std(arr)
        return (arr - np.mean(arr)) / (s if s > 1e-7 else 1.0)

    # Señal visual: Alto contraste, alta saturación, nitidez (canny) y brillo equilibrado
    z_contrast = _standardize(visual_contrast)
    z_saturation = _standardize(visual_saturation)
    z_canny = _standardize(visual_canny)
    z_entropy = _standardize(visual_entropy)
    score_visual = (
        0.35 * z_contrast
        + 0.35 * z_saturation
        + 0.15 * z_canny
        + 0.15 * z_entropy
    )

    # Señal acústica: Mayor nivel RMS (loudness), tempo en rango bailable (95-128 BPM), MFCC2
    tempo_affinity = 1.0 - np.clip(np.abs(audio_tempo - 116.0) / 45.0, 0.0, 1.0)
    z_rms = _standardize(audio_rms)
    z_tempo = _standardize(tempo_affinity)
    z_mfcc2 = _standardize(mfcc_2)
    score_audio = (
        0.45 * z_rms
        + 0.35 * z_tempo
        + 0.20 * z_mfcc2
    )

    # Señal lingüística: Polaridad emocional VADER (magnitud y positividad), legibilidad y brevedad
    emotional_potency = np.abs(vader_arr) + np.maximum(0.0, vader_arr)
    z_emotion = _standardize(emotional_potency)
    z_inflesz = _standardize(inflesz_arr)
    length_penalty = 1.0 - np.clip(np.abs(words_arr - 11.0) / 12.0, 0.0, 1.0)
    z_length = _standardize(length_penalty)
    score_text = (
        0.45 * z_emotion
        + 0.30 * z_inflesz
        + 0.25 * z_length
    )

    # Score latente continuo unificado con ruido estocástico controlado
    stochastic_noise = rng.normal(loc=0.0, scale=0.35, size=n_samples)
    latent_virality_score = (
        0.35 * score_visual
        + 0.40 * score_audio
        + 0.25 * score_text
        + stochastic_noise
    )

    # Calibración P90: Exactamente el 10% de las muestras asignadas a la clase positiva (y_clf = 1)
    k_positives = int(np.round(0.10 * n_samples))
    top_indices = np.argsort(latent_virality_score)[-k_positives:]

    y_clf = np.zeros(n_samples, dtype=int)
    y_clf[top_indices] = 1

    p90_threshold = float(np.percentile(latent_virality_score, 90.0))
    logger.info(
        "Calibración P90 completada: Umbral P90 = %.4f | Positivos (1): %d (%.1f%%) | Negativos (0): %d (%.1f%%)",
        p90_threshold,
        k_positives,
        (k_positives / n_samples) * 100.0,
        n_samples - k_positives,
        ((n_samples - k_positives) / n_samples) * 100.0,
    )

    # 5. CONSTRUCCIÓN DEL DATAFRAME CON LAS COLUMNAS EXACTAS
    data_dict: Dict[str, Any] = {
        # 5 Visuales
        "visual_brightness_mean": visual_brightness,
        "visual_contrast_rms": visual_contrast,
        "visual_color_entropy": visual_entropy,
        "visual_saturation_mean": visual_saturation,
        "visual_canny_edge_density": visual_canny,
        # 16 Acústicas
        "audio_rms_mean": audio_rms,
        "audio_zcr_mean": audio_zcr,
        "audio_bpm_tempo": audio_tempo,
        "audio_mfcc_1": mfcc_1,
        "audio_mfcc_2": mfcc_2,
        "audio_mfcc_3": mfcc_3,
        "audio_mfcc_4": mfcc_4,
        "audio_mfcc_5": mfcc_5,
        "audio_mfcc_6": mfcc_6,
        "audio_mfcc_7": mfcc_7,
        "audio_mfcc_8": mfcc_8,
        "audio_mfcc_9": mfcc_9,
        "audio_mfcc_10": mfcc_10,
        "audio_mfcc_11": mfcc_11,
        "audio_mfcc_12": mfcc_12,
        "audio_mfcc_13": mfcc_13,
        # 4 Lingüísticas
        "text_inflesz_score": inflesz_arr,
        "text_fernandez_huerta_score": huerta_arr,
        "text_vader_compound": vader_arr,
        "text_word_count": words_arr,
        # Texto crudo y variable objetivo
        "text_raw": texts_raw,
        "y_clf": y_clf,
    }

    df = pd.DataFrame(data_dict)

    # Verificación de columnas frente a ALL_CONTINUOUS_FEATURE_NAMES
    expected_continuous = list(ALL_CONTINUOUS_FEATURE_NAMES)
    for col in expected_continuous:
        if col not in df.columns:
            raise ValueError(f"Falta columna obligatoria en el dataset: {col}")

    expected_total_cols = expected_continuous + ["text_raw", "y_clf"]
    df = df[expected_total_cols]

    # Guardado si se especifica la ruta
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False, encoding="utf-8")
        logger.info("Dataset maestro multimodal guardado en: %s", out_p.resolve())

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    print("=" * 60)
    print("SonixPulse AI - Generando Multimodal Master Dataset (Calibración P90)")
    print("=" * 60)

    dataset = build_multimodal_dataset(n_samples=200, seed=42)

    print(f"\n[OK] Filas totales generadas: {len(dataset)}")
    print(f"[OK] Columnas totales: {len(dataset.columns)}")
    print(f"[OK] Variables numéricas continuas ({len(ALL_CONTINUOUS_FEATURE_NAMES)}):")
    for idx, c in enumerate(ALL_CONTINUOUS_FEATURE_NAMES, start=1):
        print(f"  {idx:2d}. {c}")
    print(f"  26. text_raw (tipo: {dataset['text_raw'].dtype})")
    print(f"  27. y_clf (tipo: {dataset['y_clf'].dtype})")

    counts = dataset["y_clf"].value_counts().to_dict()
    pos_count = counts.get(1, 0)
    neg_count = counts.get(0, 0)
    total = len(dataset)
    print("\n[OK] Distribución de Clases Calibrada (P90):")
    print(f"  Clase 0 (Estándar): {neg_count} muestras ({(neg_count / total) * 100:.1f}%)")
    print(f"  Clase 1 (Viral P90): {pos_count} muestras ({(pos_count / total) * 100:.1f}%)")

    dest_file = Path("data/processed/multimodal_master_dataset.csv")
    print(f"\n[OK] Archivo guardado: {dest_file.resolve()} ({dest_file.stat().st_size} bytes)")
    print("=" * 60)
