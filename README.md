# 🧠 RAG Data Lab Inspector

**Motor de Análisis Vectorial con Django + Ollama + ChromaDB**

Sistema completo de RAG (Retrieval-Augmented Generation) con interfaz visual para inspeccionar cada etapa del pipeline: extracción, chunking, vectorización y consulta semántica.

---

## 🚀 Características

| Módulo | Descripción |
|---|---|
| **Motor de Extracción Multi-Motor** | PyMuPDF, pdfplumber, PyPDF2, OCR (Tesseract), y **Híbrido PyMuPDF+OCR** |
| **🐭 deadmau5 (Deep Clean)** | Post-procesador que elimina ruido, ligaduras, páginas huérfanas y repara palabras cortadas |
| **Pipeline Visual** | Visualización completa de texto limpio, chunks y vectores con botones de descarga |
| **Vector Lab** | Consulta directa a ChromaDB sin LLM para validar embeddings (distancia L2 + interpretación) |
| **Chat & Análisis (IA)** | Chat con Ollama (Llama 3, Qwen, etc.) con Re-Ranking neuronal (FlashRank) |
| **Wizard Guiado** | Flujo paso a paso: Subir PDF → Motor → Chunking → Procesar |

## 📋 Stack Tecnológico

- **Backend:** Django + Django REST Framework
- **Extracción:** PyMuPDF, pdfplumber, pypdf, Tesseract OCR
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace
- **Vector Store:** ChromaDB (persistente)
- **LLM:** Ollama (local, modelos como `qwen2:1.5b`, `llama3`)
- **Re-Ranking:** FlashRank (`ms-marco-MiniLM-L-12-v2`)
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`

## ⚡ Instalación Rápida

```bash
# 1. Clonar
git clone https://github.com/jmendoza0297/rag-data-lab.git
cd rag-data-lab

# 2. Entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Dependencias
pip install -r requirements.txt

# 4. Migraciones
python manage.py migrate

# 5. Ejecutar
python manage.py runserver 8000
```

Abrir `http://localhost:8000`

## 🔧 Requisitos Externos

- **Ollama** → [ollama.com](https://ollama.com) (para el chat con IA)
- **Tesseract OCR** → Necesario para el motor híbrido con PDFs escaneados
  ```
  winget install UB-Mannheim.TesseractOCR
  ```

## 📁 Estructura del Proyecto

```
rag_django_ollama/
├── api/
│   ├── services/
│   │   ├── extractor.py      # 5 motores de extracción + deadmau5
│   │   └── rag_logic.py      # Pipeline RAG completo
│   ├── templates/
│   │   └── index.html         # UI completa (wizard + pipeline + chat)
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── serializers.py
├── rag_project/                # Configuración Django
├── requirements.txt
└── manage.py
```

## 📊 Motores de Extracción

| Motor | Cuándo usar |
|---|---|
| ⚡ PyMuPDF | PDFs digitales con texto seleccionable |
| 🧠 **Híbrido + OCR** | **RECOMENDADO** — Detecta texto + OCR en páginas escaneadas |
| 📊 pdfplumber | PDFs con tablas y diseño complejo |
| 📄 PyPDF2 | Fallback simple |
| 🔍 OCR puro | Solo imágenes |

---

**Desarrollado con 🔬 precisión técnica para auditoría RAG.**
