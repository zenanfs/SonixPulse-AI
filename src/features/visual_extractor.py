"""
SonixPulse AI - Deterministic CPU Visual Feature Extractor
Modulo de extraccion de caracteristicas visuales continuas sobre CPU a partir
de rutas de archivo, arrays numpy o buffers de bytes en memoria.
Operaciones vectorizadas sobre OpenCV y NumPy con valores neutrales de respaldo.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
import numpy as np

logger = logging.getLogger("SonixPulse.VisualExtractor")

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
    image_input: Union[str, Path, np.ndarray, bytes, bytearray, io.BytesIO, Any, None] = None
) -> Dict[str, float]:
    """Extrae 5 metricas visuales continuas a partir de ruta, array o buffer de bytes.

    Metricas calculadas:
    1. visual_brightness_mean: Brillo medio de luminancia en escala 0 a 255.
    2. visual_contrast_rms: Contraste RMS (desviacion estandar de luminancia).
    3. visual_color_entropy: Entropia de Shannon del color sobre histograma normalizado.
    4. visual_saturation_mean: Saturacion media en espacio de color HSV.
    5. visual_canny_edge_density: Proporcion de pixeles activos con detector Canny.

    Args:
        image_input: Ruta al archivo (str/Path), numpy array, bytes/BytesIO, o None.

    Returns:
        Diccionario plano con las 5 claves continuas en formato float.
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

        elif isinstance(image_input, (bytes, bytearray, memoryview)):
            if len(image_input) == 0:
                logger.warning("Buffer de imagen vacio recibido. Usando fallback neutral.")
                return DEFAULT_VISUAL_FEATURES.copy()

            buf = np.frombuffer(image_input, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.warning("Fallo al decodificar buffer de imagen con cv2.imdecode. Usando fallback neutral.")
                return DEFAULT_VISUAL_FEATURES.copy()

        elif hasattr(image_input, "read"):
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            data = image_input.getvalue() if hasattr(image_input, "getvalue") else image_input.read()
            if not data:
                logger.warning("Stream de imagen vacio recibido. Usando fallback neutral.")
                return DEFAULT_VISUAL_FEATURES.copy()

            buf = np.frombuffer(data, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.warning("Fallo al decodificar stream de imagen con cv2.imdecode. Usando fallback neutral.")
                return DEFAULT_VISUAL_FEATURES.copy()

        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                logger.warning("Array de imagen vacio recibido. Usando fallback neutral.")
                return DEFAULT_VISUAL_FEATURES.copy()

            arr = image_input.copy()
            # Escalar a uint8 si los datos son flotantes en [0, 1] o fuera de rango uint8
            if np.issubdtype(arr.dtype, np.floating):
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.clip(0, 255).astype(np.uint8)

            if arr.ndim == 2:
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 3:
                if arr.shape[2] == 3:
                    img_bgr = arr
                elif arr.shape[2] == 4:
                    img_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                elif arr.shape[2] == 1:
                    img_bgr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
                else:
                    logger.warning("Canales de imagen no soportados: %s. Usando fallback.", arr.shape)
                    return DEFAULT_VISUAL_FEATURES.copy()
            else:
                logger.warning("Dimension de array no soportada para imagen: %s. Usando fallback.", arr.shape)
                return DEFAULT_VISUAL_FEATURES.copy()

        else:
            logger.warning("Tipo de entrada no soportado para imagen: %s. Usando fallback.", type(image_input))
            return DEFAULT_VISUAL_FEATURES.copy()

        if img_bgr is None or img_bgr.size == 0:
            return DEFAULT_VISUAL_FEATURES.copy()

        # 1. Luminancia / Escala de grises para brillo y contraste
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness_mean = float(np.mean(gray))
        contrast_rms = float(np.std(gray))

        # 2. Entropia de Shannon del color sobre histograma normalizado
        hist, _ = np.histogram(img_bgr, bins=256, range=(0, 256))
        p = hist.astype(np.float64) / (np.sum(hist) + 1e-12)
        nonzero_p = p[p > 0]
        color_entropy = float(-np.sum(nonzero_p * np.log2(nonzero_p + 1e-12)))

        # 3. Saturacion media en espacio HSV
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        saturation_mean = float(np.mean(hsv[:, :, 1]))

        # 4. Densidad de bordes Canny con suavizado gaussiano previo (sigma=1.0)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
        edges = cv2.Canny(blurred, 50, 150)
        canny_edge_density = float(np.count_nonzero(edges) / edges.size)

        return {
            "visual_brightness_mean": brightness_mean if np.isfinite(brightness_mean) else DEFAULT_VISUAL_FEATURES["visual_brightness_mean"],
            "visual_contrast_rms": contrast_rms if np.isfinite(contrast_rms) else DEFAULT_VISUAL_FEATURES["visual_contrast_rms"],
            "visual_color_entropy": color_entropy if np.isfinite(color_entropy) else DEFAULT_VISUAL_FEATURES["visual_color_entropy"],
            "visual_saturation_mean": saturation_mean if np.isfinite(saturation_mean) else DEFAULT_VISUAL_FEATURES["visual_saturation_mean"],
            "visual_canny_edge_density": canny_edge_density if np.isfinite(canny_edge_density) else DEFAULT_VISUAL_FEATURES["visual_canny_edge_density"],
        }

    except Exception as exc:
        logger.error("Error imprevisto en extract_visual_features: %s. Retornando valores neutrales.", exc)
        return DEFAULT_VISUAL_FEATURES.copy()
