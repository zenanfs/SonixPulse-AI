# Imagen base oficial de Python 3.10 slim
FROM python:3.10-slim

# Evitar escritura de archivos .pyc en disco y buffer de salida stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt uvicorn fastapi

# Copiar el código fuente del proyecto
COPY . .

# Puerto expuesto para el servidor API REST TrendScout
EXPOSE 8000

# Comando de inicio del servidor Uvicorn FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
