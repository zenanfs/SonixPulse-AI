@echo off
echo ===================================================
echo  Iniciando SonixPulse AI - Consola Multimodal Beta
echo ===================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo Usando entorno virtual .venv...
    ".venv\Scripts\python.exe" src/api/main.py
) else (
    echo [ERROR] No se encontro el entorno .venv en este directorio.
    pause
)
