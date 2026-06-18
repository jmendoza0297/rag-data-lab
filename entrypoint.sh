#!/bin/bash
# ==============================================================================
# RAG Data Lab Inspector — Entrypoint
# ==============================================================================
# Script de inicio inteligente para Docker.
# Decide qué servicio arrancar según la variable SERVICE_TYPE:
#   - "web"    → Django (Gunicorn) + migraciones
#   - "celery" → Celery Worker
# ==============================================================================
set -e

echo "============================================"
echo "  🔬 RAG Data Lab Inspector"
echo "  Service: ${SERVICE_TYPE:-web}"
echo "============================================"

if [ "${SERVICE_TYPE}" = "web" ] || [ -z "${SERVICE_TYPE}" ]; then
    echo ""
    echo "🔄 Ejecutando migraciones de base de datos..."
    python manage.py migrate --noinput
    
    echo "📦 Recopilando archivos estáticos..."
    python manage.py collectstatic --noinput 2>/dev/null || true
    
    echo ""
    echo "🌐 Iniciando Gunicorn en puerto 8000..."
    echo "   Workers: ${GUNICORN_WORKERS:-2}"
    echo "   Timeout: ${GUNICORN_TIMEOUT:-300}s"
    echo ""
    
    exec gunicorn rag_project.wsgi:application \
        --bind 0.0.0.0:8000 \
        --workers "${GUNICORN_WORKERS:-2}" \
        --timeout "${GUNICORN_TIMEOUT:-300}" \
        --access-logfile - \
        --error-logfile -

elif [ "${SERVICE_TYPE}" = "celery" ]; then
    echo ""
    echo "⏳ Esperando a que Django haga las migraciones (5s)..."
    sleep 5
    
    echo "⚙️  Iniciando Celery Worker..."
    echo "   Concurrency: ${CELERY_CONCURRENCY:-2}"
    echo ""
    
    exec celery -A rag_project worker \
        --loglevel=info \
        --concurrency="${CELERY_CONCURRENCY:-2}"

else
    echo "❌ SERVICE_TYPE no reconocido: ${SERVICE_TYPE}"
    echo "   Opciones válidas: web, celery"
    exit 1
fi
