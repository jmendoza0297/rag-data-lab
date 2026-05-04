# 🧠 Informe Técnico: RAG Data Lab Inspector

## 1. Resumen Ejecutivo
**RAG Data Lab Inspector** es una aplicación web avanzada de inspección y auditoría para sistemas RAG (Retrieval-Augmented Generation). Su objetivo principal es desmitificar la "caja negra" que suele ser la ingesta de documentos y la generación de vectores, permitiendo a desarrolladores e investigadores tener control granular sobre la extracción de texto, fragmentación (chunking), y consultas de similitud matemática antes de involucrar a un LLM.

La arquitectura se basa en **Django** como framework backend, **ChromaDB** como base de datos vectorial local, **LangChain** para la orquestación, y **Ollama** para la inferencia de Modelos de Lenguaje Locales (LLMs).

## 2. Arquitectura del Sistema

### 2.1 Tecnologías Core
*   **Backend:** Python 3, Django, Django REST Framework (DRF)
*   **Vector Store:** ChromaDB (Persistencia local en `storage/`)
*   **Orquestación RAG:** LangChain (Core, Community, HuggingFace)
*   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Vía HuggingFace, dimensiones: 384)
*   **Inferencia LLM:** Ollama (Local)
*   **Re-Ranking:** FlashRank (`ms-marco-MiniLM-L-12-v2`)

### 2.2 Flujo de Procesamiento (Pipeline)
El pipeline de procesamiento se ejecuta de manera síncrona/asíncrona y consta de 4 etapas críticas:

1.  **Ingesta & Extracción:** Lectura del PDF usando un motor seleccionado.
2.  **Limpieza (deadmau5):** Post-procesamiento opcional para purificar el texto.
3.  **Fragmentación (Chunking):** División del texto en segmentos manejables.
4.  **Vectorización:** Conversión de los fragmentos en representaciones matemáticas (embeddings) y almacenamiento en ChromaDB.

## 3. Componentes Destacados

### 3.1 Módulo de Extracción Multi-Motor (`api.services.extractor`)
Para mitigar la pérdida de información durante la lectura de PDFs, el sistema implementa 5 motores de extracción intercambiables dinámicamente:
*   **PyMuPDF:** Máxima velocidad, ideal para PDFs nativos digitales.
*   **pdfplumber:** Preservación estricta de layouts y tablas complejas.
*   **PyPDF2/pypdf:** Fallback ligero.
*   **OCR (Tesseract):** Extracción pura basada en imágenes.
*   **Híbrido (PyMuPDF + OCR):** Motor inteligente (por defecto) que extrae texto digital, y si detecta una página escaneada, aplica OCR on-the-fly.

### 3.2 Post-Procesador `deadmau5 (Deep Clean)`
Filtro de alta precisión que se ejecuta pre-chunking para reducir el ruido semántico:
*   Resolución de ligaduras Unicode (ej. `ﬁ` -> `fi`).
*   Eliminación de números de página huérfanos y cabeceras repetitivas.
*   Reparación de palabras truncadas por saltos de línea/guiones.
*   Colapso de espacios en blanco horizontales y saltos redundantes.

### 3.3 Vector Lab (Laboratorio Vectorial)
Interfaz que elude la inferencia del LLM para realizar consultas directas a **ChromaDB**. Calcula la **Distancia L2** (Similitud Euclidiana) entre el query y los chunks, devolviendo un diagnóstico categórico (Idéntico, Muy Alta, Media, Baja) para auditar la calidad del embedding.

### 3.4 Chat & Análisis con Re-Ranking
Interfaz conversacional que utiliza Ollama. Aplica un proceso de **Re-Ranking con FlashRank**:
1. El Vector Store recupera el Top-K inicial de chunks relevantes.
2. El modelo Cross-Encoder evalúa la relevancia real del query contra cada chunk.
3. Se reordenan los resultados y se descartan los irrelevantes antes de enviarlos al prompt del LLM, reduciendo alucinaciones y ahorrando tokens de contexto.

## 4. Guía de Uso (Wizard Guiado)

El uso de la plataforma está orquestado a través de un Wizard en el panel lateral (Sidebar):

### Paso 1: Subir Documento
El usuario selecciona un archivo PDF local. El sistema lo carga al servidor (`media/pdfs/`).

### Paso 2: Selección de Motor
Se escoge cómo se interpretará el PDF.
*   *Recomendación:* Usar **PyMuPDF Híbrido + OCR** si no se conoce la naturaleza del PDF.
*   *Toggle deadmau5:* Se recomienda mantenerlo activo para asegurar chunks más limpios.

### Paso 3: Configuración de Fragmentación
*   **Estrategia:** `RecursiveCharacterTextSplitter` (divide por párrafos, luego oraciones, luego palabras).
*   **Tamaño (Chars):** Tamaño base del fragmento (default: 1000). Afecta cuánta información de contexto se incluye en un solo vector.
*   **Solapamiento:** Porcentaje o cantidad de caracteres que se repiten entre un chunk y el siguiente para no perder el contexto entre cortes (default: 15%).

### Paso 4: Procesar
Al iniciar el análisis, el backend ejecuta la pipeline. La interfaz (Vector Lab Pipeline) mostrará visualmente el resultado de cada etapa: texto extraído puro, JSON de los chunks, y muestra de los vectores generados, permitiendo su descarga para auditoría externa.

### 4.1 Pestañas de Interacción
*   **Vector Lab Pipeline:** Muestra el "esqueleto" de los datos y permite hacer consultas crudas a la base de datos vectorial para ver qué tan bien los embeddings coinciden matemáticamente con un query.
*   **Chat & Análisis (IA):** Permite chatear con un modelo de Ollama (ej. Llama 3) para que responda basándose estrictamente en los documentos cargados, utilizando el Re-Ranker para asegurar alta precisión.

## 5. Requerimientos de Infraestructura
*   Entorno Python con las dependencias listadas en `requirements.txt`.
*   Servidor Ollama corriendo localmente o accesible en red (API puerto 11434 por defecto).
*   Binarios de Tesseract OCR instalados a nivel de Sistema Operativo y agregados al PATH (para los motores OCR e Híbrido).
*   Suficiente RAM para mantener el modelo de Embeddings, el modelo de Re-Ranking (FlashRank), y procesar documentos grandes en memoria.
