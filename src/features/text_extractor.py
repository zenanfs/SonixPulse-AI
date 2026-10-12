"""
SonixPulse AI - Deterministic CPU Text Feature Extractor
Modulo de extraccion de caracteristicas heuristicas y estadisticas de legibilidad
y sentimiento afectivo para textos en espanol sobre CPU.
"""

import logging
import re
from typing import Any, Dict, List, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger("SonixPulse.TextExtractor")

_VADER_ANALYZER: Optional[SentimentIntensityAnalyzer] = None


def _get_vader_analyzer() -> Optional[SentimentIntensityAnalyzer]:
    """Inicializa de forma perezosa el analizador VADER asegurando descarga del lexico."""
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        try:
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
        except LookupError:
            try:
                nltk.download("vader_lexicon", quiet=True)
                _VADER_ANALYZER = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.warning("Fallo al inicializar VADER lexico: %s", e)
                return None
        except Exception as e:
            logger.warning("Fallo al inicializar VADER SentimentIntensityAnalyzer: %s", e)
            return None
    return _VADER_ANALYZER


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
    """Estima el numero de silabas de una palabra en espanol mediante reglas fonologicas.

    Agrupa diptongos y triptongos estandar en espanol y respeta rupturas por hiatos.
    """
    w = word.lower()
    w = re.sub(r"[^a-záéíóúüñ]", "", w)
    if not w:
        return 0

    # Triptongos: vocal debil + vocal fuerte + vocal debil
    w = re.sub(r"[iuü][aeoáéó][iuü]", "V", w)

    # Diptongos: vocal fuerte + debil atona | debil atona + fuerte | dos debiles atonas
    # Las vocales debiles con tilde (i, u) actuan como fuertes y rompen la agrupacion
    w = re.sub(r"[aeoáéó][iuü]|[iuü][aeoáéó]|[iuü]{2}", "V", w)

    vowels = re.findall(r"[aeiouáéíóúüV]", w)
    return max(1, len(vowels))


def split_sentences_spanish(text: str) -> List[str]:
    """Divide un texto en frases utilizando delimitadores sintacticos estandar."""
    raw_sentences = re.split(r"[.!?;\n\r]+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def extract_text_features(text_input: Optional[str] = None) -> Dict[str, Any]:
    """Extrae 4 metricas continuas de legibilidad y sentimiento mas el texto crudo normalizado.

    Metricas calculadas:
    1. text_inflesz_score: Escala de legibilidad INFLESZ para espanol.
    2. text_fernandez_huerta_score: Indice de legibilidad Fernandez-Huerta.
    3. text_vader_compound: Sentimiento afectivo compuesto (-1.0 a +1.0) mediante VADER NLTK.
    4. text_word_count: Conteo total de palabras o tokens.
    5. text_raw: Texto crudo normalizado para vectorizacion posterior.

    Args:
        text_input: Cadena de texto a analizar o None.

    Returns:
        Diccionario plano con las 4 caracteristicas continuas y 'text_raw'.
    """
    if text_input is None:
        return DEFAULT_TEXT_FEATURES.copy()

    try:
        raw_str = str(text_input).strip()
        if not raw_str:
            return DEFAULT_TEXT_FEATURES.copy()

        text_raw = " ".join(raw_str.split())
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

        total_syllables = sum(count_syllables_word_spanish(w) for w in words)
        syllables_per_word = total_syllables / n_words
        words_per_sentence = n_words / n_sentences

        # 1. Escala INFLESZ: 206.835 - 62.3*(silabas/palabras) - (palabras/frases)
        inflesz = 206.835 - 62.3 * syllables_per_word - words_per_sentence

        # 2. Indice Fernandez-Huerta: 206.84 - 60.0*(silabas/palabras) - 1.02*(palabras/frases)
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
