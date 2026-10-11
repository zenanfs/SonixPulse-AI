#!/usr/bin/env python3
"""
SonixPulse AI - Deterministic CPU Visual Feature Extractor
==========================================================
Extrae características visuales deterministas, vectorizadas y ligeras en CPU
a partir de imágenes (portadas / miniaturas / frames).
Operaciones sobre OpenCV y NumPy, tolerante a fallos con valores de respaldo neutrales.
"""

import logging
from pathlib import Path
from typing import Dict, Union, Optional
import cv2
import numpy as np

logger = logging.getLogger("SonixPulse.VisualExtractor")

# Valores de respaldo neutrales (en caso de archivo faltante, corrupto o None)
DEFAULT_VISUAL_FEATURES: Dict[str, float] = {
    "visual_brightness_mean": 128.0,
    "visual_contrast_rms": 45.0,
    "visual_color_entropy": 5.0,
    "visual_saturation_mean": 70.0,
    "visual_canny_edge_density": 0.05,
}

FEATURE_NAMES_VISUAL = [
    "visual_brightness_mean",
    "visual_contrast_rms",
    "visual_color_entropy",
    "visual_saturation_mean",
    "visual_canny_edge_density",
]


def extract_visual_features(
    image_input: Union[str, Path, np.ndarray, None] = None
) -> Dict[str, float]:
    """Extrae 5 métricas visuales continuas a partir de una ruta de imagen o numpy array.

    Métricas calculadas:
    1. visual_brightness_mean: Brillo medio de luminancia (escala 0-255).
    2. visual_contrast_rms: Contraste RMS (desviación estándar de luminancia).
    3. visual_color_entropy: Entropía de Shannon del color sobre histograma normalizado.
    4. visual_saturation_mean: Saturación media en espacio de color HSV.
    5. visual_canny_edge_density: Proporción de píxeles activos con detector Canny.

    Args:
        image_input: Ruta al archivo de imagen (str o Path), numpy array de imagen, o None.

    Returns:
        Diccionario plano con las 5 claves float. Si la imagen es None o falla,
        devuelve valores neutrales por defecto.
    """
    if image_input is None:
        return DEFAULT_VISUAL_FEATURES.copy()

    try:
        img_bgr: Optional[np.ndarray] = None

        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.is_file():
                logger.warning("Archivo de imagen no encontrado: %s. Usando fallback neutral.", img_path)
                return DEFAULT_VISUAL_FEATURES.copy()

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("Fallo al decodificar la imagen: %s. Usando fallback neutral.", img_path)
                return DEFAULT_VISUAL_FEATURES.copy()

        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                logger.warning("Array de imagen vacío recibido. Usando fallback neutral.")
                return DEFAULT_VISUAL_FEATURES.copy()

            arr = image_input.copy()
            # Si el array es flotante en rango [0, 1], escalar a uint8 [0, 255]
            if np.issubdtype(arr.dtype, np.floating):
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.clip(0, 255).astype(np.uint8)

            if arr.ndim == 2:
                # Monocromático (H, W) -> Convertir a BGR
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 3:
                if arr.shape[2] == 3:
                    img_bgr = arr
                elif arr.shape[2] == 4:
                    img_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                elif arr.shape[2] == 1:
                    img_bgr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
                else:
                    logger.warning("Dimensiones de canales inesperadas: %s. Usando fallback.", arr.shape)
                    return DEFAULT_VISUAL_FEATURES.copy()
            else:
                logger.warning("Array con dimensiones no soportadas: %s. Usando fallback.", arr.shape)
                return DEFAULT_VISUAL_FEATURES.copy()

        else:
            logger.warning("Tipo de entrada no soportado para imagen: %s. Usando fallback.", type(image_input))
            return DEFAULT_VISUAL_FEATURES.copy()

        # 1. Luminancia / Escala de grises para brillo y contraste
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness_mean = float(np.mean(gray))
        contrast_rms = float(np.std(gray))

        # 2. Entropía de Shannon del color sobre histograma normalizado
        # -sum(p * log2(p + 1e-12))
        hist, _ = np.histogram(img_bgr, bins=256, range=(0, 256))
        p = hist.astype(np.float64) / (np.sum(hist) + 1e-12)
        nonzero_p = p[p > 0]
        color_entropy = float(-np.sum(nonzero_p * np.log2(nonzero_p + 1e-12)))

        # 3. Saturación media en espacio de color HSV
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        saturation_mean = float(np.mean(hsv[:, :, 1]))

        # 4. Densidad de bordes Canny (sigma=1.0, umbrales 50 y 150)
        # Aplicamos suavizado gaussiano sigma=1.0 para emular el parámetro sigma del detector
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
        edges = cv2.Canny(blurred, 50, 150)
        canny_edge_density = float(np.count_nonzero(edges) / edges.size)

        return {
            "visual_brightness_mean": brightness_mean,
            "visual_contrast_rms": contrast_rms,
            "visual_color_entropy": color_entropy,
            "visual_saturation_mean": saturation_mean,
            "visual_canny_edge_density": canny_edge_density,
        }

    except Exception as exc:
        logger.error("Error imprevisto en extract_visual_features: %s. Retornando valores neutrales.", exc)
        return DEFAULT_VISUAL_FEATURES.copy()
