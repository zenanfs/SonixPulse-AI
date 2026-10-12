"""
Pruebas automatizadas de la API REST de SonixPulse AI
Verificacion de endpoints /api/health y /api/predict-virality con TestClient.
"""

import io
import pytest
from fastapi.testclient import TestClient
from src.api.main import app, TEMP_UPLOADS_DIR

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["estado"] == "operativo"
    assert data["modelo_cargado"] is True
    assert data["variables_multimodales"] == 25


def test_predict_with_preset_hit_ritmico():
    response = client.post(
        "/api/predict-virality",
        data={"preset": "hit_ritmico"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["exito"] is True
    assert 0.0 <= data["probabilidad"] <= 1.0
    assert 0.0 <= data["probabilidad_porcentaje"] <= 100.0
    assert isinstance(data["es_viral_p90"], bool)

    radar = data["radar_comercial"]
    for axis in ["atractivo_visual", "energia_ritmo", "claridad_mensaje", "conexion_emocional", "retencion_estimada"]:
        assert axis in radar
        assert 0.0 <= radar[axis] <= 100.0

    assert len(data["factores_impulso"]) > 0
    assert "nombre_comercial" in data["factores_impulso"][0]
    assert "recomendacion" in data["factores_impulso"][0]

    assert len(data["factores_friccion"]) > 0
    assert "nombre_comercial" in data["factores_friccion"][0]
    assert "recomendacion" in data["factores_friccion"][0]

    assert len(data["recomendaciones"]) > 0


def test_predict_with_preset_campana_comercial():
    response = client.post(
        "/api/predict-virality",
        data={"preset": "campana_comercial"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["exito"] is True
    assert "radar_comercial" in data


def test_predict_with_preset_podcast_conversacional():
    response = client.post(
        "/api/predict-virality",
        data={"preset": "podcast_conversacional"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["exito"] is True
    assert data["es_viral_p90"] is False


def test_predict_solo_texto():
    response = client.post(
        "/api/predict-virality",
        data={"texto": "Campana exclusiva de lanzamiento para redes sociales."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["exito"] is True
    assert "metadata" in data
    assert data["metadata"]["origen"] == "publicidad_personalizada"
    assert "texto" in data["metadata"]["componentes_analizados"]
    assert "audio" not in data["metadata"]["componentes_analizados"]


def test_predict_imagen_y_texto_sin_audio():
    dummy_img = (
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff"
        b"\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00"
        b"\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    )
    response = client.post(
        "/api/predict-virality",
        files={"imagen": ("banner.gif", dummy_img, "image/gif")},
        data={"texto": "Anuncio con banner visual y llamado a la accion directo."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["exito"] is True
    assert data["metadata"]["origen"] == "publicidad_personalizada"
    assert "imagen" in data["metadata"]["componentes_analizados"]
    assert "texto" in data["metadata"]["componentes_analizados"]
    assert "audio" not in data["metadata"]["componentes_analizados"]


def test_predict_deseleccion_preset_vacio_o_none():
    # Enviar preset vacio explicito
    res_vacio = client.post(
        "/api/predict-virality",
        data={"preset": "", "texto": "Copy publicitario sin preset seleccionado."}
    )
    assert res_vacio.status_code == 200
    data_vacio = res_vacio.json()
    assert data_vacio["exito"] is True
    assert data_vacio["metadata"]["origen"] == "publicidad_personalizada"

    # Enviar preset none explicito
    res_none = client.post(
        "/api/predict-virality",
        data={"preset": "none", "texto": "Modo personalizado activo sin preset."}
    )
    assert res_none.status_code == 200
    data_none = res_none.json()
    assert data_none["exito"] is True
    assert data_none["metadata"]["origen"] == "publicidad_personalizada"


def test_predict_ephemeral_file_cleanup():
    # 1x1 GIF en memoria
    dummy_img = (
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff"
        b"\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00"
        b"\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    )

    initial_files = set(TEMP_UPLOADS_DIR.glob("*"))

    response = client.post(
        "/api/predict-virality",
        files={"imagen": ("test.gif", dummy_img, "image/gif")},
        data={"texto": "Lanzamiento visual con diseno moderno."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["exito"] is True

    # Comprobacion de que no quedaron archivos huerfanos en temp_uploads/
    final_files = set(TEMP_UPLOADS_DIR.glob("*"))
    assert len(final_files - initial_files) == 0


def test_predict_with_audio_url_fallback():
    response = client.post(
        "/api/predict-virality",
        data={
            "audio_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "texto": "Pista con enlace de audio de prueba."
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["exito"] is True
    assert "radar_comercial" in data
