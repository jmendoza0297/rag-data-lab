# 🐳 Guía de Despliegue en Ubuntu — RAG Data Lab Inspector

## Índice
1. [Requisitos del Servidor Ubuntu](#1-requisitos-del-servidor-ubuntu)
2. [Preparar la Máquina Ubuntu](#2-preparar-la-máquina-ubuntu)
3. [Instalar Ollama (Nativo)](#3-instalar-ollama-nativo)
4. [Transferir el Proyecto](#4-transferir-el-proyecto)
5. [Configurar Variables de Entorno](#5-configurar-variables-de-entorno)
6. [Construir y Levantar](#6-construir-y-levantar)
7. [Verificación](#7-verificación)
8. [Comandos Útiles](#8-comandos-útiles)
9. [Migrar Datos desde Windows](#9-migrar-datos-desde-windows)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Requisitos del Servidor Ubuntu

| Requisito | Mínimo | Recomendado |
|-----------|--------|-------------|
| **Ubuntu** | 22.04 LTS | 24.04 LTS |
| **RAM** | 8 GB | 16 GB |
| **CPU** | 4 cores | 8 cores |
| **Disco** | 20 GB libres | 50 GB libres |
| **GPU** | No requerida | NVIDIA (para Ollama rápido) |

---

## 2. Preparar la Máquina Ubuntu

### 2.1 Actualizar el sistema
```bash
sudo apt update && sudo apt upgrade -y
```

### 2.2 Instalar Docker
```bash
# Instalar dependencias
sudo apt install -y ca-certificates curl gnupg lsb-release

# Agregar clave GPG oficial de Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Agregar repositorio
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Instalar Docker Engine + Compose
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Agregar tu usuario al grupo docker (para no usar sudo)
sudo usermod -aG docker $USER
newgrp docker
```

### 2.3 Verificar Docker
```bash
docker --version
docker compose version
docker run hello-world
```

---

## 3. Instalar Ollama (Nativo)

> **IMPORTANTE**: Ollama se instala NATIVO (fuera de Docker) para acceder directamente a la GPU.

### 3.1 Instalar Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3.2 (Opcional) Si tienes GPU NVIDIA
```bash
# Verificar que los drivers NVIDIA están instalados
nvidia-smi

# Si NO tienes drivers, instálalos:
sudo apt install -y nvidia-driver-535
sudo reboot
```

### 3.3 Descargar los modelos necesarios
```bash
# Modelo principal (el que usa tu app)
ollama pull qwen2:1.5b

# (Opcional) Otros modelos que quieras usar
# ollama pull llama3.2:3b
# ollama pull mistral:7b
```

### 3.4 Verificar que Ollama está corriendo
```bash
# Ollama se inicia automáticamente como servicio
systemctl status ollama

# Probar que responde
curl http://localhost:11434/
# Debería devolver: "Ollama is running"

# Listar modelos descargados
ollama list
```

### 3.5 Configurar Ollama para aceptar conexiones Docker
```bash
# Editar la configuración del servicio
sudo systemctl edit ollama

# Agregar estas líneas en el editor:
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0:11434"

# Reiniciar el servicio
sudo systemctl restart ollama

# Verificar que escucha en todas las interfaces
ss -tlnp | grep 11434
```

---

## 4. Transferir el Proyecto

### Opción A: Git (Recomendada)
```bash
# En el servidor Ubuntu
cd /opt
sudo mkdir rag-data-lab && sudo chown $USER:$USER rag-data-lab
cd rag-data-lab

# Clonar desde tu repositorio
git clone <URL_DE_TU_REPO> .
```

### Opción B: SCP desde Windows
```bash
# Desde PowerShell en Windows (ajustar IP y usuario)
scp -r C:\Users\PC-EURO\.gemini\antigravity-ide\scratch\rag_django_ollama\* usuario@IP_UBUNTU:/opt/rag-data-lab/
```

### Opción C: Copiar manualmente (USB, FileZilla, etc.)
Copiar toda la carpeta `rag_django_ollama` al servidor en `/opt/rag-data-lab/`

> **NOTA**: No copies las carpetas `venv/`, `redis/`, `__pycache__/`, ni `db.sqlite3` (se generarán automáticamente en Docker).

---

## 5. Configurar Variables de Entorno

```bash
cd /opt/rag-data-lab

# Editar el archivo .env
nano .env
```

Ajustar estos valores según tu entorno:

```env
# --- Django ---
DEBUG=False                              # ← Poner False en producción
SECRET_KEY=genera-una-clave-secreta-aqui  # ← Cambiar!
ALLOWED_HOSTS=tu-ip-ubuntu,tu-dominio.com,localhost

# --- Redis ---
REDIS_URL=redis://redis:6379/0

# --- Qdrant ---
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=rag_collection

# --- Ollama ---
OLLAMA_BASE_URL=http://host.docker.internal:11434

# --- Base de Datos ---
DATABASE_PATH=/app/storage/db.sqlite3

# --- Gunicorn ---
GUNICORN_WORKERS=4   # ← Ajustar según CPUs disponibles (regla: 2*CPU + 1)
GUNICORN_TIMEOUT=300

# --- Celery ---
CELERY_CONCURRENCY=4  # ← Ajustar según CPUs
```

### Generar una clave secreta segura
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(50))"
```

---

## 6. Construir y Levantar

```bash
cd /opt/rag-data-lab

# Crear directorios de datos persistentes
mkdir -p media/pdfs storage/docstore storage/flashrank_cache

# Construir las imágenes (la primera vez toma ~10 min)
docker compose build

# Levantar todos los servicios en segundo plano
docker compose up -d

# Ver los logs en tiempo real
docker compose logs -f
```

### ¿Qué pasa al levantar?
1. **Redis** → Arranca primero (el broker de mensajes)
2. **Qdrant** → Arranca segundo (base de datos vectorial)
3. **web** → Espera a Redis, ejecuta migraciones, arranca Gunicorn
4. **celery-worker** → Espera 5s, arranca el worker

---

## 7. Verificación

```bash
# 1. Verificar que todos los contenedores estén corriendo
docker compose ps
# Deberían verse 4 servicios: rag-web, rag-celery, rag-redis, rag-qdrant

# 2. Verificar Django
curl http://localhost:8090/
# Debería devolver la página HTML de la app

# 3. Verificar Qdrant
curl http://localhost:6333/
# Debería devolver info de Qdrant

# 4. Verificar Redis
docker compose exec redis redis-cli ping
# Debería devolver: PONG

# 5. Verificar Ollama (desde dentro del contenedor)
docker compose exec web curl http://host.docker.internal:11434/
# Debería devolver: "Ollama is running"

# 6. Ver logs de cada servicio
docker compose logs web
docker compose logs celery-worker
docker compose logs redis
docker compose logs qdrant
```

### Acceder a la app
Desde cualquier navegador en la red:
```
http://IP_DEL_SERVIDOR:8090
```

---

## 8. Comandos Útiles

### Operaciones diarias
```bash
# Reiniciar todos los servicios
docker compose restart

# Detener todo
docker compose down

# Detener y BORRAR volúmenes (⚠️ borra datos de Qdrant)
docker compose down -v

# Reconstruir después de cambios en código
docker compose up -d --build

# Ver logs en tiempo real de un servicio
docker compose logs -f web
docker compose logs -f celery-worker
```

### Mantenimiento
```bash
# Entrar al contenedor Django (para debug)
docker compose exec web bash

# Ejecutar migraciones manualmente
docker compose exec web python manage.py migrate

# Crear superusuario Django
docker compose exec web python manage.py createsuperuser

# Ver el uso de espacio en disco
docker system df

# Limpiar imágenes y caché no usados
docker system prune -af
```

---

## 9. Migrar Datos desde Windows

Si quieres llevar los documentos ya procesados de Windows a Ubuntu:

### 9.1 Copiar la base de datos SQLite
```bash
# Desde Windows (PowerShell)
scp C:\Users\PC-EURO\.gemini\antigravity-ide\scratch\rag_django_ollama\db.sqlite3 usuario@IP_UBUNTU:/opt/rag-data-lab/storage/db.sqlite3
```

### 9.2 Copiar los PDFs
```bash
# Desde Windows (PowerShell)
scp -r C:\Users\PC-EURO\.gemini\antigravity-ide\scratch\rag_django_ollama\media\pdfs\* usuario@IP_UBUNTU:/opt/rag-data-lab/media/pdfs/
```

### 9.3 Copiar el docstore local
```bash
scp -r C:\Users\PC-EURO\.gemini\antigravity-ide\scratch\rag_django_ollama\storage\docstore\* usuario@IP_UBUNTU:/opt/rag-data-lab/storage/docstore/
```

> **NOTA**: Los datos de Qdrant NO se migran con archivos. Necesitarás re-procesar y hacer "Push a Qdrant" de cada documento desde la interfaz web.

---

## 10. Troubleshooting

### ❌ "Ollama is not running" desde Docker
```bash
# Verificar que Ollama escucha en 0.0.0.0
ss -tlnp | grep 11434
# Debe mostrar: *:11434 o 0.0.0.0:11434

# Si solo muestra 127.0.0.1:11434, reconfigura:
sudo systemctl edit ollama
# Agregar: Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

### ❌ "Connection refused" a Qdrant o Redis
```bash
# Verificar que los contenedores estén corriendo
docker compose ps

# Revisar logs del servicio problemático
docker compose logs qdrant
docker compose logs redis
```

### ❌ La imagen tarda mucho en construirse
```bash
# Usar buildkit para mejor caché
DOCKER_BUILDKIT=1 docker compose build

# Si se queda en "Downloading model", es normal la primera vez
# all-MiniLM-L6-v2 = ~80MB, FlashRank = ~25MB, spaCy = ~15MB
```

### ❌ Error de permisos en volúmenes
```bash
# Asegurar que los directorios tengan los permisos correctos
sudo chown -R 1000:1000 /opt/rag-data-lab/media
sudo chown -R 1000:1000 /opt/rag-data-lab/storage
```

### ❌ El worker de Celery no procesa tareas
```bash
# Ver los logs del worker
docker compose logs -f celery-worker

# Verificar conexión a Redis
docker compose exec celery-worker python -c "import redis; r = redis.from_url('redis://redis:6379/0'); print(r.ping())"
```

### ❌ Quiero reiniciar solo un servicio
```bash
docker compose restart web
docker compose restart celery-worker
```
