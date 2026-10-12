"""
SonixPulse AI - Deterministic CPU Audio Feature Extractor
Modulo de extraccion de caracteristicas acusticas continuas sobre CPU a partir
de rutas de archivo, arrays de senal (1D/2D) o buffers en memoria (bytes/BytesIO).
Ingesta mono a 22050 Hz con tolerancia a fallos y valores de respaldo neutrales.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union
import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger("SonixPulse.AudioExtractor")

TARGET_SAMPLE_RATE = 22050
N_MFCC = 13

# Valores neutrales de respaldo para las 16 caracteristicas continuas de audio
DEFAULT_AUDIO_FEATURES: Dict[str, float] = {
    "audio_rms_mean": 0.08,
    "audio_zcr_mean": 0.05 + 1e-4,
    "audio_bpm_tempo": 120.0,
    **{f"audio_mfcc_{i}": 0.0 for i in range(1, N_MFCC + 1)},
}

FEATURE_NAMES_AUDIO = list(DEFAULT_AUDIO_FEATURES.keys())


def extract_audio_features(
    audio_input: Union[str, Path, np.ndarray, bytes, bytearray, io.BytesIO, Any, None] = None,
    sr: int = TARGET_SAMPLE_RATE,
) -> Dict[str, float]:
    """Extrae las 16 caracteristicas acusticas deterministas en tiempo continuo.

    Metricas calculadas:
    1. audio_rms_mean: Media temporal del nivel RMS de amplitud.
    2. audio_zcr_mean: Tasa media de cruces por cero con offset estabilizador (+1e-4).
    3. audio_bpm_tempo: Estimacion del tempo ritmico en BPM.
    4-16. audio_mfcc_1 a audio_mfcc_13: 13 coeficientes MFCC promediados en el tiempo.

    Args:
        audio_input: Ruta al archivo (str/Path), numpy array 1D/2D, bytes/BytesIO, o None.
        sr: Frecuencia de muestreo objetivo (por defecto 22050 Hz).

    Returns:
        Diccionario plano con las 16 claves continuas de audio.
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

            y, _ = librosa.load(str(audio_path), sr=sr, mono=True)

        elif isinstance(audio_input, (bytes, bytearray, memoryview, io.BytesIO, io.BufferedIOBase)) or hasattr(audio_input, "read"):
            if isinstance(audio_input, (bytes, bytearray, memoryview)):
                bio = io.BytesIO(audio_input)
                raw_bytes = bytes(audio_input)
            else:
                bio = audio_input
                if hasattr(bio, "seek"):
                    bio.seek(0)
                raw_bytes = bio.read() if hasattr(bio, "read") else bytes(bio)
                if hasattr(bio, "seek"):
                    bio.seek(0)

            if len(raw_bytes) == 0:
                logger.warning("Buffer de audio vacio recibido. Usando fallback neutral.")
                return DEFAULT_AUDIO_FEATURES.copy()

            # Intento de lectura directa en memoria con soundfile
            try:
                data, sample_rate = sf.read(bio)
                if data.ndim == 2:
                    if data.shape[0] <= data.shape[1]:
                        y = librosa.to_mono(data.astype(np.float32))
                    else:
                        y = librosa.to_mono(data.astype(np.float32).T)
                else:
                    y = data.astype(np.float32)

                if sample_rate != sr:
                    y = librosa.resample(y=y, orig_sr=sample_rate, target_sr=sr)
            except Exception:
                # Fallback a archivo temporal si soundfile en memoria no reconoce el contenedor
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                        tmp.write(raw_bytes)
                        tmp_path = Path(tmp.name)
                    y, _ = librosa.load(str(tmp_path), sr=sr, mono=True)
                except Exception as decode_exc:
                    logger.warning("Fallo al decodificar audio desde buffer: %s. Usando fallback.", decode_exc)
                    return DEFAULT_AUDIO_FEATURES.copy()
                finally:
                    if tmp_path and tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass

        elif isinstance(audio_input, np.ndarray):
            if audio_input.size == 0:
                logger.warning("Array de audio vacio recibido. Usando fallback neutral.")
                return DEFAULT_AUDIO_FEATURES.copy()

            arr = audio_input.astype(np.float32)
            if arr.ndim == 2:
                if arr.shape[0] <= arr.shape[1]:
                    y = librosa.to_mono(arr)
                else:
                    y = librosa.to_mono(arr.T)
            elif arr.ndim == 1:
                y = arr
            else:
                logger.warning("Array de audio con dimension no soportada: %s. Usando fallback.", arr.ndim)
                return DEFAULT_AUDIO_FEATURES.copy()

        else:
            logger.warning("Tipo de entrada no soportado para audio: %s. Usando fallback.", type(audio_input))
            return DEFAULT_AUDIO_FEATURES.copy()

        # Validacion de senal minima para calculo espectral
        if y is None or len(y) < 512 or not np.isfinite(y).all():
            logger.warning("Senal de audio demasiado corta o con valores infinitos/NaN. Usando fallback.")
            return DEFAULT_AUDIO_FEATURES.copy()

        # 1. Nivel RMS medio de amplitud
        rms = librosa.feature.rms(y=y)
        audio_rms_mean = float(np.mean(rms))

        # 2. Cruces por cero con offset estabilizador (+1e-4)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        audio_zcr_mean = float(np.mean(zcr) + 1e-4)

        # 3. Estimacion de BPM (tempo)
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
            "audio_rms_mean": audio_rms_mean if np.isfinite(audio_rms_mean) else DEFAULT_AUDIO_FEATURES["audio_rms_mean"],
            "audio_zcr_mean": audio_zcr_mean if np.isfinite(audio_zcr_mean) else DEFAULT_AUDIO_FEATURES["audio_zcr_mean"],
            "audio_bpm_tempo": audio_bpm_tempo if np.isfinite(audio_bpm_tempo) else DEFAULT_AUDIO_FEATURES["audio_bpm_tempo"],
        }

        for i in range(1, N_MFCC + 1):
            val = float(mfcc_means[i - 1])
            result[f"audio_mfcc_{i}"] = val if np.isfinite(val) else 0.0

        return result

    except Exception as exc:
        logger.error("Error imprevisto en extract_audio_features: %s. Retornando valores neutrales.", exc)
        return DEFAULT_AUDIO_FEATURES.copy()
