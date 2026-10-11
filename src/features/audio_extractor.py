#!/usr/bin/env python3
"""
SonixPulse AI - Deterministic CPU Audio Feature Extractor
=========================================================
Extrae características acústicas deterministas, vectorizadas y ligeras en CPU
a partir de archivos de audio (WAV, MP3, etc.) o arrays de señal.
Ingesta mono a 22050 Hz mediante librosa.
Tolerante a fallos con valores de respaldo neutrales.
"""

import logging
from pathlib import Path
from typing import Dict, Union, Optional
import numpy as np
import librosa

logger = logging.getLogger("SonixPulse.AudioExtractor")

TARGET_SAMPLE_RATE = 22050
N_MFCC = 13

# Valores de respaldo neutrales en caso de fallo, archivo no encontrado o señal nula
DEFAULT_AUDIO_FEATURES: Dict[str, float] = {
    "audio_rms_mean": 0.08,
    "audio_zcr_mean": 0.05 + 1e-4,
    "audio_bpm_tempo": 120.0,
    **{f"audio_mfcc_{i}": 0.0 for i in range(1, N_MFCC + 1)},
}

FEATURE_NAMES_AUDIO = list(DEFAULT_AUDIO_FEATURES.keys())


def extract_audio_features(
    audio_input: Union[str, Path, np.ndarray, None] = None,
    sr: int = TARGET_SAMPLE_RATE,
) -> Dict[str, float]:
    """Extrae las 16 características acústicas normalizadas en tiempo continuo.

    Métricas calculadas:
    1. audio_rms_mean: Media temporal del nivel RMS de amplitud.
    2. audio_rms_std: Desviación estándar temporal del nivel RMS.
    3. audio_zcr_mean: Tasa media de cruces por cero con offset estabilizador (+1e-4).
    4. audio_bpm_tempo: Estimación del tempo rítmico en BPM.
    5-17. audio_mfcc_1 a audio_mfcc_13: 13 coeficientes MFCC promediados en el tiempo.

    Args:
        audio_input: Ruta al archivo de audio (str/Path), numpy array 1D/2D, o None.
        sr: Frecuencia de muestreo objetivo (por defecto 22050 Hz).

    Returns:
        Diccionario plano con claves float estandarizadas. Si la entrada es None,
        el archivo no existe o ocurre un error, devuelve valores de respaldo neutrales.
    """
    if audio_input is None:
        return DEFAULT_AUDIO_FEATURES.copy()

    try:
        y: Optional[np.ndarray] = None

        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.is_file():
                logger.warning("Archivo de audio no encontrado: %s. Usando fallback neutral.", audio_path)
                return DEFAULT_AUDIO_FEATURES.copy()

            # Ingesta mono a 22050 Hz
            y, _ = librosa.load(str(audio_path), sr=sr, mono=True)

        elif isinstance(audio_input, np.ndarray):
            if audio_input.size == 0:
                logger.warning("Array de audio vacío recibido. Usando fallback neutral.")
                return DEFAULT_AUDIO_FEATURES.copy()

            arr = audio_input.astype(np.float32)
            # Manejo de canales estéreo/multicanal -> convertir a mono
            if arr.ndim == 2:
                if arr.shape[0] < arr.shape[1]:
                    # Forma (canales, muestras)
                    y = librosa.to_mono(arr)
                else:
                    # Forma (muestras, canales)
                    y = librosa.to_mono(arr.T)
            elif arr.ndim == 1:
                y = arr
            else:
                logger.warning("Array de audio con dimensión inesperada: %s. Usando fallback.", arr.ndim)
                return DEFAULT_AUDIO_FEATURES.copy()
        else:
            logger.warning("Tipo de entrada no soportado para audio: %s. Usando fallback.", type(audio_input))
            return DEFAULT_AUDIO_FEATURES.copy()

        # Validación de señal mínima (requiere al menos un frame FFT estándar de 512 muestras)
        if y is None or len(y) < 512 or not np.isfinite(y).all():
            logger.warning("Señal de audio demasiado corta o con valores infinitos/NaN. Usando fallback.")
            return DEFAULT_AUDIO_FEATURES.copy()

        # 1. Nivel RMS medio de amplitud
        rms = librosa.feature.rms(y=y)
        audio_rms_mean = float(np.mean(rms))

        # 2. Cruces por cero con offset estabilizador (+1e-4)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        audio_zcr_mean = float(np.mean(zcr) + 1e-4)

        # 3. Estimación de BPM (tempo)
        tempo_arr = librosa.feature.tempo(y=y, sr=sr)
        if hasattr(tempo_arr, "__len__"):
            audio_bpm_tempo = float(tempo_arr[0]) if len(tempo_arr) > 0 else 120.0
        else:
            audio_bpm_tempo = float(tempo_arr)

        if not np.isfinite(audio_bpm_tempo) or audio_bpm_tempo <= 0.0:
            audio_bpm_tempo = 120.0

        # 4. 13 Coeficientes MFCC promediados temporalmente
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_means = np.mean(mfcc, axis=1)

        result: Dict[str, float] = {
            "audio_rms_mean": audio_rms_mean,
            "audio_zcr_mean": audio_zcr_mean,
            "audio_bpm_tempo": audio_bpm_tempo,
        }

        for i in range(1, N_MFCC + 1):
            val = float(mfcc_means[i - 1])
            result[f"audio_mfcc_{i}"] = val if np.isfinite(val) else 0.0

        return result

    except Exception as exc:
        logger.error("Error imprevisto en extract_audio_features: %s. Retornando valores neutrales.", exc)
        return DEFAULT_AUDIO_FEATURES.copy()
