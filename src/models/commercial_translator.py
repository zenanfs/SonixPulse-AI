"""
SonixPulse AI - Modulo de Traduccion Comercial y Radar de 5 Ejes
Logica de mapeo de variables tecnicas a lenguaje de marketing
y calculo de indices normalizados para la toma de decisiones.
"""

from typing import Dict, Any, List, Optional
import numpy as np


COMMERCIAL_FEATURE_NAMES: Dict[str, str] = {
    "visual_contrast_rms": "Contraste Visual",
    "visual_brightness_mean": "Luminosidad de Portada",
    "visual_color_entropy": "Variedad Visual",
    "visual_saturation_mean": "Intensidad del Color",
    "visual_canny_edge_density": "Detalle y Nitidez",
    "audio_rms_mean": "Fuerza del Audio",
    "audio_zcr_mean": "Presencia de Frecuencias Altas",
    "audio_bpm_tempo": "Ritmo y Velocidad Musical",
    "audio_bpm": "Ritmo y Velocidad Musical",
    "text_inflesz_score": "Fluidez de Lectura",
    "text_fernandez_huerta_score": "Sencillez del Texto",
    "text_vader_compound": "Impacto Emocional",
    "text_vader_pos": "Afinidad Positiva",
    "text_word_count": "Longitud del Mensaje",
}


def get_commercial_name(feature_name: str) -> str:
    """
    Traduce el nombre tecnico de una variable a terminologia de negocio.
    """
    clean_name = feature_name.replace("num__", "")
    if clean_name.startswith("text__"):
        token = clean_name.replace("text__", "").strip()
        return f"Gancho en Texto: {token}"

    if clean_name.startswith("audio_mfcc_"):
        return "Complejidad Timbrica"

    return COMMERCIAL_FEATURE_NAMES.get(clean_name, clean_name.replace("_", " ").title())


def compute_commercial_radar(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Calcula los 5 ejes normalizados (0 a 100) del Radar Comercial:
    1. Atractivo Visual: ponderacion de contraste, saturacion, variedad y nitidez.
    2. Energia y Ritmo: combinacion de presion acustica y ventana optima 115-135 BPM.
    3. Claridad del Mensaje: promedio ponderado de legibilidad INFLESZ y Fernandez-Huerta.
    4. Conexion Emocional: magnitud de polaridad emocional combinada con intensidad cromatica.
    5. Retencion Estimada: indice sintetico de dinamismo sonoro, impacto visual y concision.
    """
    # 1. Atractivo Visual: normalizacion lineal con acotamiento superior
    contrast = float(features.get("visual_contrast_rms", 40.0))
    saturation = float(features.get("visual_saturation_mean", 90.0))
    entropy = float(features.get("visual_color_entropy", 6.0))
    edge = float(features.get("visual_canny_edge_density", 0.08))

    norm_contrast = float(np.clip(contrast / 65.0 * 100.0, 0.0, 100.0))
    norm_saturation = float(np.clip(saturation / 160.0 * 100.0, 0.0, 100.0))
    norm_entropy = float(np.clip(entropy / 7.5 * 100.0, 0.0, 100.0))
    norm_edge = float(np.clip(edge / 0.16 * 100.0, 0.0, 100.0))

    atractivo_visual = (
        0.30 * norm_contrast +
        0.30 * norm_saturation +
        0.20 * norm_entropy +
        0.20 * norm_edge
    )

    # 2. Energia y Ritmo: normalizacion de RMS y penalizacion cuadratico-lineal fuera de 115-135 BPM
    rms = float(features.get("audio_rms_mean", 0.15))
    bpm = float(features.get("audio_bpm_tempo", features.get("audio_bpm", 120.0)))

    norm_rms = float(np.clip((rms - 0.02) / 0.22 * 100.0, 0.0, 100.0))
    if 115.0 <= bpm <= 135.0:
        # Meseta optima centrada en 125 BPM
        norm_bpm = 100.0 - abs(bpm - 125.0) * 1.5
    elif bpm < 115.0:
        norm_bpm = max(0.0, 100.0 - (115.0 - bpm) * 2.5)
    else:
        norm_bpm = max(0.0, 100.0 - (bpm - 135.0) * 2.5)

    energia_ritmo = 0.50 * norm_rms + 0.50 * norm_bpm

    # 3. Claridad del Mensaje: escalas estandarizadas de comprension lectora
    inflesz = float(features.get("text_inflesz_score", 70.0))
    fernandez = float(features.get("text_fernandez_huerta_score", 70.0))

    norm_inflesz = float(np.clip(inflesz, 0.0, 100.0))
    norm_fernandez = float(np.clip(fernandez, 0.0, 100.0))

    claridad_mensaje = 0.50 * norm_inflesz + 0.50 * norm_fernandez

    # 4. Conexion Emocional: valor absoluto de polaridad afectiva e intensidad de color
    vader_comp = float(features.get("text_vader_compound", 0.5))
    norm_vader = float(np.clip(abs(vader_comp) * 100.0, 0.0, 100.0))

    conexion_emocional = 0.60 * norm_vader + 0.40 * norm_saturation

    # 5. Retencion Estimada: sintesis de dinamismo acustico, presencia visual y concision textual
    zcr = float(features.get("audio_zcr_mean", 0.08))
    norm_zcr = float(np.clip(zcr / 0.15 * 100.0, 0.0, 100.0))
    dinamismo_acustico = 0.60 * norm_rms + 0.40 * norm_zcr

    word_count = float(features.get("text_word_count", 15.0))
    if 8.0 <= word_count <= 25.0:
        concision_texto = 100.0
    elif word_count < 8.0:
        concision_texto = max(20.0, word_count * 12.5)
    else:
        concision_texto = max(15.0, 100.0 - (word_count - 25.0) * 2.8)

    retencion_estimada = (
        0.40 * dinamismo_acustico +
        0.35 * atractivo_visual +
        0.25 * concision_texto
    )

    return {
        "atractivo_visual": round(float(np.clip(atractivo_visual, 0.0, 100.0)), 1),
        "energia_ritmo": round(float(np.clip(energia_ritmo, 0.0, 100.0)), 1),
        "claridad_mensaje": round(float(np.clip(claridad_mensaje, 0.0, 100.0)), 1),
        "conexion_emocional": round(float(np.clip(conexion_emocional, 0.0, 100.0)), 1),
        "retencion_estimada": round(float(np.clip(retencion_estimada, 0.0, 100.0)), 1),
    }


def generate_business_recommendation(
    feature_name: str,
    shap_value: float,
    current_value: Optional[float] = None
) -> str:
    """
    Traduce el impacto SHAP a una recomendacion practica de negocio,
    libre de terminologia redundante o formulas artificiales.
    """
    clean_name = feature_name.replace("num__", "")

    if clean_name.startswith("text__"):
        token = clean_name.replace("text__", "").strip()
        if shap_value > 0:
            return f"Mantener el termino '{token}' en el texto de portada por su traccion demostrada."
        return f"Sustituir la palabra '{token}' por terminos con mayor gancho y claridad directa."

    if clean_name.startswith("audio_mfcc_"):
        if shap_value > 0:
            return "Mantener el balance timbrico actual; la mezcla sonora resulta agradable al oido."
        return "Ecualizar frecuencias medias y graves para evitar una mezcla sonora saturada."

    if clean_name == "visual_contrast_rms":
        if shap_value > 0:
            return "El contraste marcado retiene la atencion del usuario en los primeros frames."
        return "Aumentar el contraste de la imagen para captar atencion en el feed."

    if clean_name == "visual_brightness_mean":
        if shap_value > 0:
            return "La exposicion luminosa asegura visibilidad clara en cualquier pantalla movil."
        return "Subir la luminosidad de portada para evitar descarte en entornos con brillo bajo."

    if clean_name == "visual_color_entropy":
        if shap_value > 0:
            return "La variedad cromatica aporta dinamismo visual a la escena."
        return "Incorporar mayor variedad de color en la escena para dinamizar el plano."

    if clean_name == "visual_saturation_mean":
        if shap_value > 0:
            return "La viveza de los colores resalta frente a otros contenidos competidores."
        return "Ajustar la saturacion de color para que los tonos llamen mas la atencion."

    if clean_name == "visual_canny_edge_density":
        if shap_value > 0:
            return "La definicion y detalle visual transmiten calidad de produccion."
        return "Mejorar el enfoque y la definicion de bordes para evitar una apariencia borrosa."

    if clean_name == "audio_rms_mean":
        if shap_value > 0:
            return "El volumen y presencia del sonido compiten de forma directa en el feed."
        return "Subir el volumen general del audio para que el sonido no pase desapercibido."

    if clean_name in ("audio_bpm_tempo", "audio_bpm"):
        if shap_value > 0:
            return "El tempo musical acompasa de forma natural la atencion del usuario."
        if current_value is not None and current_value < 115:
            return "Elevar el ritmo musical hacia 120 BPM para transmitir mayor dinamismo."
        return "Ajustar el ritmo musical hacia el rango dinamico de 115 a 130 BPM."

    if clean_name == "audio_zcr_mean":
        if shap_value > 0:
            return "Los transitorios percusivos marcan el paso de forma nitida."
        return "Abrir frecuencias agudas y transitorios para dar mayor brillo al ritmo."

    if clean_name == "text_inflesz_score":
        if shap_value > 0:
            return "La fluidez de lectura reduce el esfuerzo mental de la audiencia."
        return "Simplificar la redaccion para permitir una lectura rapida en feed."

    if clean_name == "text_fernandez_huerta_score":
        if shap_value > 0:
            return "Las oraciones sencillas aceleran la asimilacion del mensaje."
        return "Reducir la longitud de las frases para acelerar la comprension."

    if clean_name == "text_vader_compound":
        if shap_value > 0:
            return "El tono emotivo definido estimula comentarios y reproducciones."
        return "Definir una postura emocional mas clara para despertar reaccion en el espectador."

    if clean_name == "text_word_count":
        if shap_value > 0:
            return "La extension del texto resulta concisa y directa."
        return "Ajustar la longitud del texto a frases breves de entre 10 y 20 palabras."

    # Regla por defecto para otras variables
    commercial_title = get_commercial_name(clean_name)
    if shap_value > 0:
        return f"El valor en {commercial_title} favorece la retencion de la audiencia."
    return f"Optimizar el parametro {commercial_title} para reducir el abandono inicial."


def format_commercial_factors(
    contributions_list: List[Dict[str, Any]],
    features_dict: Optional[Dict[str, Any]] = None,
    top_n: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Separa y estructura los factores SHAP en factores de impulso (positivos)
    y factores de friccion (negativos) con nomenclatura comercial y recomendaciones directas.
    """
    feats = features_dict or {}

    positives = [item for item in contributions_list if item.get("shap_value", 0.0) > 0]
    positives.sort(key=lambda x: x.get("shap_value", 0.0), reverse=True)

    negatives = [item for item in contributions_list if item.get("shap_value", 0.0) < 0]
    negatives.sort(key=lambda x: x.get("shap_value", 0.0))

    factores_impulso = []
    for item in positives[:top_n]:
        feat = item["feature"]
        clean = feat.replace("num__", "").replace("text__", "")
        shap_val = float(item["shap_value"])
        curr_val = feats.get(clean)
        rec = generate_business_recommendation(feat, shap_val, curr_val)
        factores_impulso.append({
            "caracteristica": feat,
            "nombre_comercial": get_commercial_name(feat),
            "impacto": round(shap_val, 4),
            "recomendacion": rec,
        })

    factores_friccion = []
    for item in negatives[:top_n]:
        feat = item["feature"]
        clean = feat.replace("num__", "").replace("text__", "")
        shap_val = float(item["shap_value"])
        curr_val = feats.get(clean)
        rec = generate_business_recommendation(feat, shap_val, curr_val)
        factores_friccion.append({
            "caracteristica": feat,
            "nombre_comercial": get_commercial_name(feat),
            "impacto": round(shap_val, 4),
            "recomendacion": rec,
        })

    return {
        "factores_impulso": factores_impulso,
        "factores_friccion": factores_friccion,
    }


# Alias en espanol segun codigo-humano
calcular_radar_comercial = compute_commercial_radar
traducir_factores_shap = format_commercial_factors
obtener_nombre_comercial = get_commercial_name
