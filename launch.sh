#!/bin/bash
# ==============================================================================
# SonixPulse AI — Live Client Simulator & Launch Script
# ==============================================================================

echo "======================================================================"
echo "    🚀 INICIANDO PLATAFORMA SONIXPULSE AI (MODO SIMULACIÓN CLIENTE B2B)"
echo "======================================================================"

PROJECT_DIR="/home/zfernandez/.gemini/antigravity/scratch/trendscout"
cd "$PROJECT_DIR" || exit

# 1. Activar entorno virtual
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 2. Ejecutar actualización telemétrica en vivo (Ingesta + K-Means + Motor Financiero)
echo "[1/3] Ejecutando ingesta y modelo predictivo en tiempo real..."
python3 src/ingestion/data_ingestion.py
python3 src/models/predictive_model.py
python3 src/models/financial_roi_engine.py

# 3. Abrir la interfaz web interactiva en el navegador
echo "[2/3] Abriendo el Dashboard B2B en el navegador..."
if command -v xdg-open > /dev/null; then
    xdg-open "$PROJECT_DIR/web/index.html" &
else
    echo "Abre manualmente en tu navegador: file://$PROJECT_DIR/web/index.html"
fi

# 4. Iniciar el servidor API REST en segundo plano (Puerto 8000)
echo "[3/3] Iniciando Servidor API REST SonixPulse AI en http://localhost:8000..."
python3 src/api/main.py
