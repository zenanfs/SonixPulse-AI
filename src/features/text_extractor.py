#!/usr/bin/env python3
"""
SonixPulse AI - Deterministic CPU Text Feature Extractor
========================================================
Extrae características heurísticas y estadísticas de legibilidad y sentimiento
a partir de textos en español (títulos, descripciones, copies, letras).
Implementa heurística silábica adaptada al español, fórmulas de legibilidad
INFLESZ y Fernández-Huerta, y análisis de polaridad afectiva VADER (NLTK).
Tolerante a fallos con valores de respaldo neutrales.
"""

import re
import logging
from typing import Dict, Any, List, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger("SonixPulse.TextExtractor")

# Inicialización segura y perezosa de SentimentIntensityAnalyzer de NLTK
_VADER_ANALYZER: Optional[SentimentIntensityAnalyzer] = None


def _get_vader_analyzer() -> Optional[SentimentIntensityAnalyzer]:
    """Obtiene o inicializa el analizador VADER asegurando que el léxico esté disponible."""
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        try:
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
        except LookupError:
            try:
                nltk.download("vader_lexicon", quiet=True)
                _VADER_ANALYZER = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.warning("No se pudo inicializar VADER SentimentIntensityAnalyzer: %s", e)
                return None
        except Exception as e:
            logger.warning("Error al cargar VADER: %s", e)
            return None
    return _VADER_ANALYZER


# Valores de respaldo neutrales (texto vacío o nulo)
DEFAULT_TEXT_FEATURES: Dict[str, Any] = {
    "text_inflesz_score": 60.0,
    "text_fernandez_huerta_score": 60.0,
    "text_vader_compound": 0.0,
    "text_word_count": 0.0,
    "text_raw": "",
}

FEATURE_NAMES_TEXT = [
    "text_inflesz_score",
    "text_fernandez_huerta_score",
    "text_vader_compound",
    "text_word_count",
]


def count_syllables_word_spanish(word: str) -> int:
    """Estima heurísticamente el número de sílabas de una palabra en español.

    Considera las reglas fonológicas del español:
    - Agrupación de diptongos (vocal fuerte + débil no acentuada, débil + fuerte, débil + débil).
    - Agrupación de triptongos (débil + fuerte + débil).
    - Hiato por vocales fuertes adyacentes o acento ortográfico en vocales débiles (í, ú).
    """
    w = word.lower()
    # Mantener exclusivamente caracteres alfabéticos en español
    w = re.sub(r"[^a-záéíóúüñ]", "", w)
    if not w:
        return 0

    # 1. Triptongos estándar en español: vocal débil + vocal fuerte (con o sin tilde) + vocal débil
    w = re.sub(r"[iuü][aeoáéó][iuü]", "V", w)

    # 2. Diptongos: vocal fuerte + débil átona | débil átona + fuerte | dos débiles átonas
    # Nota: í y ú con tilde actúan como fuertes (rompen el diptongo en hiato)
    w = re.sub(r"[aeoáéó][iuü]|[iuü][aeoáéó]|[iuü]{2}", "V", w)

    # 3. Contar todas las vocales restantes (incluyendo los núcleos 'V' ya agrupados)
    vowels = re.findall(r"[aeiouáéíóúüV]", w)
    return max(1, len(vowels))


def split_sentences_spanish(text: str) -> List[str]:
    """Divide un texto en frases utilizando delimitadores sintácticos estándar."""
    raw_sentences = re.split(r"[.!?;\n\r]+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def extract_text_features(text_input: Optional[str] = None) -> Dict[str, Any]:
    """Extrae 4 métricas continuas de legibilidad/sentimiento más el texto crudo normalizado.

    Métricas calculadas:
    1. text_inflesz_score: Escala de legibilidad INFLESZ para español:
       206.835 - 62.3 * (sílabas / palabras) - (palabras / frases)
    2. text_fernandez_huerta_score: Índice de legibilidad Fernández-Huerta:
       206.84 - 60.0 * (sílabas / palabras) - 1.02 * (palabras / frases)
    3. text_vader_compound: Sentimiento afectivo compuesto (-1.0 a +1.0) mediante VADER NLTK.
    4. text_word_count: Conteo total de palabras/tokens.
    5. text_raw: Texto crudo normalizado en un string limpio para vectorización TF-IDF posterior.

    Args:
        text_input: Cadena de texto a analizar (título, copy, descripción) o None.

    Returns:
        Diccionario plano con las 4 características continuas numéricas y 'text_raw'.
        Si la entrada es vacía o None, devuelve valores neutrales de respaldo.
    """
    if text_input is None:
        return DEFAULT_TEXT_FEATURES.copy()

    try:
        raw_str = str(text_input).strip()
        if not raw_str:
            return DEFAULT_TEXT_FEATURES.copy()

        # Limpieza básica de espacios para text_raw
        text_raw = " ".join(raw_str.split())

        # Extracción vectorizada de palabras
        words = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ0-9]+", text_raw)
        n_words = len(words)

        if n_words == 0:
            return {
                "text_inflesz_score": 60.0,
                "text_fernandez_huerta_score": 60.0,
                "text_vader_compound": 0.0,
                "text_word_count": 0.0,
                "text_raw": text_raw,
            }

        sentences = split_sentences_spanish(text_raw)
        n_sentences = max(1, len(sentences))

        # Conteo heurístico silábico en español
        total_syllables = sum(count_syllables_word_spanish(w) for w in words)

        syllables_per_word = total_syllables / n_words
        words_per_sentence = n_words / n_sentences

        # 1. Escala INFLESZ: 206.835 - 62.3*(sílabas/palabras) - (palabras/frases)
        inflesz = 206.835 - 62.3 * syllables_per_word - words_per_sentence

        # 2. Índice Fernández-Huerta: 206.84 - 60.0*(sílabas/palabras) - 1.02*(palabras/frases)
        fernandez_huerta = 206.84 - 60.0 * syllables_per_word - 1.02 * words_per_sentence

        # 3. Sentimiento VADER Compound
        vader_analyzer = _get_vader_analyzer()
        if vader_analyzer is not None:
            vader_compound = float(vader_analyzer.polarity_scores(text_raw).get("compound", 0.0))
        else:
            vader_compound = 0.0

        return {
            "text_inflesz_score": float(inflesz),
            "text_fernandez_huerta_score": float(fernandez_huerta),
            "text_vader_compound": float(vader_compound),
            "text_word_count": float(n_words),
            "text_raw": text_raw,
        }

    except Exception as exc:
        logger.error("Error imprevisto en extract_text_features: %s. Retornando valores neutrales.", exc)
        return DEFAULT_TEXT_FEATURES.copy()
