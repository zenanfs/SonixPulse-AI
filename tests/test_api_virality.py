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


def test_predict_with_custom_text():
    response = client.post(
        "/api/predict-virality",
        data={"texto": "Campana exclusiva de lanzamiento para redes sociales."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["exito"] is True
    assert "metadata" in data
    assert data["metadata"]["origen"] == "texto_directo"


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
