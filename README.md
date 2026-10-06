# 🎧 SonixPulse AI — B2B Biometric Acoustic Intelligence & Sonic Branding

> **Plataforma de Inteligencia Musical B2B para Agencias de Publicidad**  
> SonixPulse AI utiliza datos biométricos de la API de Spotify (energy, danceability, valence, tempo) y modelos de Machine Learning (K-Means Clustering) junto con NLP local (Ollama) para predecir vacíos de mercado acústicos (*White-Space Opportunities*) a 6 meses vista.

---

## 🛠️ Arquitectura del Repositorio

```
sonixpulse/
├── data/
│   ├── raw/                      # Extraídos vía spotipy API / Fallback Benchmark
│   │   └── top50_ecuador.csv
│   └── processed/                # Asignación de clusters K-Means y predicciones
│       ├── top50_clustered.csv
│       ├── trend_predictions.json
│       └── financial_roi_benchmark.json
├── src/
│   ├── ingestion/
│   │   └── data_ingestion.py     # Ingesta en lotes (batching) con spotipy
│   ├── models/
│   │   ├── predictive_model.py   # Clustering K-Means y saturación acústica
│   │   └── financial_roi_engine.py # Motor empírico de licencias y recordación
│   ├── nlp/
│   │   └── lyric_sentiment.py    # Integración NLP offline con Ollama (qwen/gemma)
│   └── api/
│       └── main.py               # Servidor FastAPI REST Analytics
├── web/
│   └── index.html                # Pitch Deck B2B + Dashboard Interactivo (Chart.js)
├── tests/
│   └── test_pipeline.py          # Pruebas automatizadas del pipeline
├── launch.sh                     # Script ejecutable de 1 clic (Modo Cliente)
├── Dockerfile                    # Contenedor Docker oficial
├── docker-compose.yml            # Orquestación de servicios API + Web
├── .env.example                  # Plantilla de credenciales y variables de entorno
├── requirements.txt              # Dependencias de Python
└── README.md                     # Guía de ejecución en Ubuntu
```

---

## 🚀 Guía de Inicialización en Ubuntu

### 1. Ejecución Instantánea en Modo Cliente (1 Clic)
```bash
cd /home/zfernandez/.gemini/antigravity/scratch/trendscout
chmod +x launch.sh
./launch.sh
```

### 2. Ejecución Manual de la Canalización de Datos (Pipeline)
```bash
# Ingesta en vivo
python3 src/ingestion/data_ingestion.py

# Modelo K-Means Predictivo
python3 src/models/predictive_model.py

# Motor Financiero de ROI
python3 src/models/financial_roi_engine.py

# Suite de pruebas unitarias
python3 -m unittest tests/test_pipeline.py
```

---

## 📊 Visualización del Frontend Pitch Deck & Dashboard

Puedes abrir directamente la interfaz en tu navegador predeterminado:

```bash
# Opción 1: Abrir el archivo generado en Descargas
xdg-open /home/zfernandez/Downloads/sonixpulse_pitch_deck.html

# Opción 2: Servirlo localmente con Python
python3 -m http.server 8080 --directory web
# Abrir en el navegador: http://localhost:8080
```
