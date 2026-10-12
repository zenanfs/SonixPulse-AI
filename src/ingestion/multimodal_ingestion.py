#!/usr/bin/env python3
"""
SonixPulse AI - Ephemeral Multimodal Ingestion Pipeline
========================================================
Módulo de ingesta efímera y determinista para URLs públicas (YouTube, Shorts, TikTok).
Utiliza la API nativa de yt-dlp y ffmpeg para descargar en memoria temporal (tempfile.TemporaryDirectory)
el audio en formato WAV y la miniatura del contenido.

Resiliencia y Fallback Transparente:
En caso de fallo en la descarga (bloqueo por bot/captcha, restricción geográfica,
video privado o error de red), captura la excepción de forma segura, marca
`fallback_applied=True` con el motivo correspondiente, e invoca los extractores
con valores neutros/sintéticos para garantizar continuidad operativa sin lanzar error 500.

Garantía de Limpieza:
Todos los archivos temporales se descargan dentro de un contexto efímero
y se eliminan de disco inmediatamente tras la extracción de características.
"""

import os
import sys
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yt_dlp

from src.features import (
    extract_visual_features,
    extract_audio_features,
    extract_text_features,
    ALL_CONTINUOUS_FEATURE_NAMES,
)

logger = logging.getLogger("SonixPulse.MultimodalIngestion")


def find_ffmpeg() -> Optional[str]:
    """Localiza el binario de FFmpeg en el PATH del sistema o en rutas estándar portables.

    Returns:
        Ruta absoluta al ejecutable ffmpeg si se encuentra, de lo contrario None.
    """
    # 1. Búsqueda prioritaria en PATH del sistema (multiplataforma)
    sys_path = shutil.which("ffmpeg")
    if sys_path:
        return sys_path

    # 2. Ruta estándar de Chocolatey / Windows mediante variable de entorno ProgramData
    program_data = os.environ.get("ProgramData", "C:/ProgramData")
    choco_path = Path(program_data) / "chocolatey" / "bin" / "ffmpeg.exe"
    if choco_path.exists() and choco_path.is_file():
        return str(choco_path)

    # 3. Rutas estándar en sistemas Unix / Linux / macOS
    for unix_bin in (Path("/usr/bin/ffmpeg"), Path("/usr/local/bin/ffmpeg")):
        if unix_bin.exists() and unix_bin.is_file():
            return str(unix_bin)

    return None


class MultimodalIngestor:
    """Ingestor efímero de contenido multimedia desde URLs públicas."""

    def __init__(self, ffmpeg_path: Optional[str] = None):
        """Inicializa el ingestor detectando FFmpeg automáticamente si no se provee.

        Args:
            ffmpeg_path: Ruta opcional al binario ffmpeg.
        """
        self.ffmpeg_path = ffmpeg_path or find_ffmpeg()
        if self.ffmpeg_path:
            logger.info("FFmpeg localizado para yt-dlp en: %s", self.ffmpeg_path)
        else:
            logger.warning("FFmpeg no encontrado en el sistema. Algunas conversiones de audio podrían degradarse.")

    def ingest_url(self, url: str) -> Dict[str, Any]:
        """Descarga de forma efímera los componentes de una URL y extrae sus características.

        Args:
            url: Enlace público del contenido (YouTube, Shorts, TikTok, etc.).

        Returns:
            Dict con dos claves principales:
            - 'features': Diccionario con las 25 variables continuas + 'text_raw'.
            - 'metadata': Diccionario con 'title', 'url', 'fallback_applied', 'fallback_reason'.
        """
        if not url or not isinstance(url, str) or not url.strip():
            return self._build_fallback(url=url or "", reason="URL vacía o no válida")

        clean_url = url.strip()

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                ydl_opts: Dict[str, Any] = {
                    "format": "bestaudio/best",
                    "outtmpl": str(tmp_path / "media_ephemeral.%(ext)s"),
                    "writethumbnail": True,
                    "noplaylist": True,
                    "quiet": True,
                    "no_warnings": True,
                    "socket_timeout": 15,
                    "postprocessors": [
                        {
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "wav",
                            "preferredquality": "192",
                        },
                        {
                            "key": "FFmpegThumbnailsConvertor",
                            "format": "jpg",
                            "when": "before_dl",
                        },
                    ],
                }

                if self.ffmpeg_path:
                    ydl_opts["ffmpeg_location"] = self.ffmpeg_path

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(clean_url, download=True)

                if info_dict is None:
                    return self._build_fallback(clean_url, "No se obtuvieron metadatos del medio")

                # Extracción de metadatos de texto
                title = info_dict.get("title") or "Contenido sin título"
                description = info_dict.get("description") or ""
                text_raw = f"{title}. {description}".strip() if description else title

                # Localización del archivo de audio WAV en el directorio temporal
                audio_file: Optional[Path] = None
                wav_candidates = list(tmp_path.glob("*.wav"))
                if wav_candidates:
                    audio_file = wav_candidates[0]
                else:
                    # Fallback a cualquier otro formato de audio si FFmpeg no convirtió
                    other_audio = [
                        f for f in tmp_path.glob("*")
                        if f.suffix.lower() in {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".webm"}
                    ]
                    if other_audio:
                        audio_file = other_audio[0]

                # Localización de la miniatura de imagen
                image_file: Optional[Path] = None
                img_candidates = [
                    f for f in tmp_path.glob("*")
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                ]
                if img_candidates:
                    image_file = img_candidates[0]

                # Extracción multimodal
                visual_features = extract_visual_features(image_file)
                audio_features = extract_audio_features(audio_file)
                text_features = extract_text_features(text_raw)

                # Construir el diccionario unificado de características
                features = self._combine_features(visual_features, audio_features, text_features, text_raw)

                metadata = {
                    "title": title,
                    "url": clean_url,
                    "fallback_applied": False,
                    "fallback_reason": None,
                }

                return {
                    "features": features,
                    "metadata": metadata,
                }

        except Exception as exc:
            logger.warning(
                "Excepción en descarga/extracción de '%s': %s. Aplicando fallback transparente.",
                clean_url,
                exc,
            )
            return self._build_fallback(clean_url, str(exc))

    def _build_fallback(self, url: str, reason: str) -> Dict[str, Any]:
        """Genera una respuesta neutral de respaldo ante errores de ingesta."""
        fallback_title = f"Preset Fallback ({url[:40]})" if url else "Preset Fallback Neutral"
        text_raw = fallback_title

        visual_features = extract_visual_features(None)
        audio_features = extract_audio_features(None)
        text_features = extract_text_features(text_raw)

        features = self._combine_features(visual_features, audio_features, text_features, text_raw)

        metadata = {
            "title": fallback_title,
            "url": url,
            "fallback_applied": True,
            "fallback_reason": reason,
        }

        return {
            "features": features,
            "metadata": metadata,
        }

    @staticmethod
    def _combine_features(
        visual_features: Dict[str, float],
        audio_features: Dict[str, float],
        text_features: Dict[str, Any],
        text_raw: str,
    ) -> Dict[str, Any]:
        """Combina y ordena las 25 características continuas más el texto crudo."""
        combined: Dict[str, Any] = {}
        for col in ALL_CONTINUOUS_FEATURE_NAMES:
            if col in visual_features:
                combined[col] = float(visual_features[col])
            elif col in audio_features:
                combined[col] = float(audio_features[col])
            elif col in text_features:
                combined[col] = float(text_features[col])
            else:
                combined[col] = 0.0

        combined["text_raw"] = text_features.get("text_raw", text_raw)
        return combined


def ingest_multimodal_url(url: str, ffmpeg_path: Optional[str] = None) -> Dict[str, Any]:
    """Función de conveniencia para ingesta efímera de una URL pública."""
    ingestor = MultimodalIngestor(ffmpeg_path=ffmpeg_path)
    return ingestor.ingest_url(url)


def ingest_audio_url(url: str, output_dir: Path, ffmpeg_path: Optional[str] = None) -> Optional[Path]:
    """Descarga efimera de audio desde un enlace publico (YouTube, SoundCloud, enlace directo).

    Retorna la ruta al archivo descargado en output_dir o None en caso de fallo.
    """
    if not url or not isinstance(url, str) or not url.strip():
        return None

    import uuid

    clean_url = url.strip()
    resolved_ffmpeg = ffmpeg_path or find_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_prefix = f"audio_url_{uuid.uuid4().hex[:10]}"
    out_template = str(output_dir / f"{target_prefix}.%(ext)s")

    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 12,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    if resolved_ffmpeg:
        ydl_opts["ffmpeg_location"] = resolved_ffmpeg

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clean_url])

        candidatos = [
            f for f in output_dir.glob(f"{target_prefix}.*")
            if f.is_file() and f.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".webm"}
        ]
        if candidatos:
            return candidatos[0]
        return None
    except Exception as exc:
        logger.warning("Fallo al descargar audio desde enlace '%s': %s", clean_url, exc)
        return None


# Alias de conveniencia
ingest_url = ingest_multimodal_url
