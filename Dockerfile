# ==============================================================================
# RAG Data Lab Inspector — Dockerfile
# ==============================================================================
# Construye una imagen con Django + pipeline RAG completo.
# Incluye Tesseract OCR, modelos de IA pre-descargados, y Gunicorn.
#
# El entrypoint.sh decide qué servicio arrancar:
#   SERVICE_TYPE=web    → Django (Gunicorn) + migraciones
#   SERVICE_TYPE=celery → Celery Worker
#
# Construcción:  docker compose build
# Ejecución:     docker compose up -d
# ==============================================================================

FROM python:3.12-slim

# --- Variables de entorno para Python ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TQDM_DISABLE=1

# --- Instalar dependencias del sistema operativo ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tesseract OCR + idioma español (para el motor híbrido PyMuPDF+OCR)
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    # Librerías de compilación (necesarias para PyMuPDF, sentence-transformers)
    build-essential \
    gcc \
    # Librerías de imagen (necesarias para Pillow/PyMuPDF)
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    # Utilidades
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Directorio de trabajo ---
WORKDIR /app

# --- Copiar requirements PRIMERO (optimización de caché Docker) ---
COPY requirements.txt .

# --- Instalar dependencias de Python ---
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Pre-descargar modelos de IA (quedan "horneados" en la imagen) ---
# Modelo de Embeddings: all-MiniLM-L6-v2 (~80 MB)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Modelo FlashRank: ms-marco-MiniLM-L-12-v2 (~25 MB)
RUN python -c "\
from flashrank import Ranker; \
Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/app/storage/flashrank_cache')"

# Descargar recursos NLTK para procesamiento de texto
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Modelo spaCy: Español (para NLP/entidades)
RUN python -m spacy download es_core_news_sm

# Modelo LlamaIndex EntityExtractor: tomaarsen/span-marker-mbert-base-multinerd (~500 MB)
RUN python -c "\
from llama_index.extractors.entity import EntityExtractor; \
EntityExtractor(device='cpu')"

# --- Copiar todo el código del proyecto ---
COPY . .

# --- Copiar y preparar el script de inicio ---
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# --- Crear directorios necesarios ---
RUN mkdir -p media/pdfs storage/docstore storage/flashrank_cache staticfiles

# --- Puerto ---
EXPOSE 8000

# --- Comando de inicio ---
# entrypoint.sh decide si arrancar Django o Celery según SERVICE_TYPE
ENTRYPOINT ["/app/entrypoint.sh"]
