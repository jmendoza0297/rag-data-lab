import os
import time
import psutil

# ¡PARCHE CRÍTICO PARA WINDOWS!
# Desactiva las barras de progreso de HuggingFace que causan OSError [Errno 22] en sys.stderr
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

from django.conf import settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models as qdrant_models
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from .extractor import extract as pdf_extract
from langchain_community.chat_models import ChatOllama
from langchain_classic.chains import RetrievalQA

# FlashRank: Re-ranking profesional
from flashrank import Ranker, RerankRequest

# === LlamaIndex: IngestionPipeline + Extractors ===
from llama_index.core import Document as LIDocument
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import KeywordExtractor
from llama_index.extractors.entity import EntityExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama as LIOllama


class RAGManager:
    def __init__(self):
        self.docstore_path = os.path.join(settings.BASE_DIR, 'storage', 'docstore')
        os.makedirs(self.docstore_path, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # === Qdrant Vector Store (externo) ===
        qdrant_url = getattr(settings, 'QDRANT_URL', 'http://localhost:6333')
        qdrant_api_key = getattr(settings, 'QDRANT_API_KEY', None)
        qdrant_collection = getattr(settings, 'QDRANT_COLLECTION_NAME', 'rag_collection')

        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        # Crear colección si no existe
        try:
            self.qdrant_client.get_collection(qdrant_collection)
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=qdrant_collection,
                vectors_config={
                    "fast-all-minilm-l6-v2": qdrant_models.VectorParams(
                        size=384,  # all-MiniLM-L6-v2 produce 384 dimensiones
                        distance=qdrant_models.Distance.COSINE,
                    )
                },
            )

        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=qdrant_collection,
            embedding=self.embeddings,
            vector_name=self._get_vector_name_for_collection(qdrant_collection),
        )

        fs = LocalFileStore(self.docstore_path)
        self.store = create_kv_docstore(fs)

        # Retriever persistente Parent-Child
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=250),
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000),
        )

        # FlashRank: Motor de Re-Ranking neuronal (descarga modelo ~25MB la primera vez)
        try:
            self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=os.path.join(settings.BASE_DIR, 'storage', 'flashrank_cache'))
        except Exception:
            self.reranker = None

        # === LlamaIndex: Embeddings para el pipeline de ingesta ===
        self.li_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _get_vector_name_for_collection(self, collection_name):
        """
        Obtiene el nombre del vector configurado para la colección de forma dinámica.
        Si la colección tiene un vector nombrado, lo retorna.
        Si la colección tiene un vector sin nombre (Default), retorna "".
        """
        try:
            info = self.qdrant_client.get_collection(collection_name)
            if isinstance(info.config.params.vectors, dict):
                return list(info.config.params.vectors.keys())[0]
        except Exception:
            pass
        return ""

    def _build_ingestion_pipeline(self, chunk_size=1000, chunk_overlap=150, use_entity_extractor=True, use_keyword_extractor=False, ollama_model="qwen2:1.5b"):
        """
        Construye un IngestionPipeline de LlamaIndex con:
        1. SentenceSplitter (chunking inteligente por oraciones)
        2. EntityExtractor (NER con modelo transformer local — NO usa LLM)
        3. KeywordExtractor (extracción de keywords via Ollama LLM — opcional)
        4. HuggingFaceEmbedding (generación de embeddings)
        """
        transformations = []

        # Paso 1: Chunking semántico por oraciones
        transformations.append(
            SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

        # Paso 2: EntityExtractor — NER local con transformer (NO requiere LLM)
        #   Usa span-marker bajo el capó con un modelo preentrenado de NER.
        #   Detecta: PER, ORG, LOC, MISC automáticamente.
        if use_entity_extractor:
            try:
                transformations.append(
                    EntityExtractor(
                        prediction_threshold=0.3,  # Bajo para capturar más ORG, LOC, MISC
                        label_entities=True,   # Incluir etiquetas como PER, ORG, LOC
                        device="cpu",          # CPU para compatibilidad Windows
                    )
                )
            except Exception as e:
                print(f"[LlamaIndex] EntityExtractor no disponible: {e}")

        # Paso 3: KeywordExtractor — Usa Ollama LLM para extraer keywords inteligentes
        #   A diferencia de spaCy (frecuencia de sustantivos), esto entiende el CONTEXTO.
        if use_keyword_extractor:
            try:
                llm = LIOllama(model=ollama_model, base_url=settings.OLLAMA_BASE_URL, request_timeout=120.0)
                transformations.append(
                    KeywordExtractor(llm=llm, keywords=10)
                )
            except Exception as e:
                print(f"[LlamaIndex] KeywordExtractor no disponible: {e}")

        # Paso 4: Embedding
        transformations.append(self.li_embed_model)

        return IngestionPipeline(transformations=transformations)

    def ingest_document(self, file_path, doc_id, strategy="recursive", size=1000, overlap=150, extraction_motor="pymupdf", apply_deadmau5=True):
        """
        Pipeline completo de ingesta con LlamaIndex IngestionPipeline:
        1. Extracción con el motor seleccionado por el usuario (pymupdf/pdfplumber/pypdf2/ocr)
        2. Conversión a LlamaIndex Documents
        3. IngestionPipeline: SentenceSplitter → EntityExtractor → Embedding
        4. Almacenamiento en Qdrant (via LangChain ParentDocumentRetriever)
        5. Retorno de metadatos enriquecidos para la UI
        """
        # === PASO 1: EXTRACCIÓN CON MOTOR SELECCIONADO (sin cambios — extractor.py) ===
        resultado_extraccion = pdf_extract(file_path, motor=extraction_motor, apply_deadmau5=apply_deadmau5)
        paginas = resultado_extraccion.get("pages", [])
        texto_puro_completo = resultado_extraccion.get("texto_completo", "")
        motor_usado = resultado_extraccion.get("motor_usado", extraction_motor)
        advertencias = resultado_extraccion.get("advertencias", [])

        # Metadatos del texto extraído
        texto_puro_completo_str = texto_puro_completo if isinstance(texto_puro_completo, str) else ""
        total_chars = len(texto_puro_completo_str)
        total_words = len(texto_puro_completo_str.split())

        # === PASO 2: CONVERTIR A LLAMAINDEX DOCUMENTS ===
        #   Se inyecta la marca de página "--- PAGINA X ---" al inicio del texto
        #   para que los chunks del SentenceSplitter hereden la posición de página.
        #   Esto permite al enriquecimiento semántico (paso 3.5) usar las marcas como anclas.
        li_documents = []
        for p in paginas:
            page_marker = f"--- PAGINA {p['page']} ---\n"
            li_documents.append(LIDocument(
                text=page_marker + p["text"],
                metadata={
                    "page": p["page"],
                    "source": file_path,
                    "file_id": str(doc_id),
                    "motor": motor_usado,
                }
            ))

        # === PASO 3: INGESTION PIPELINE (LlamaIndex) ===
        #   SentenceSplitter → EntityExtractor → Embedding
        #   El EntityExtractor enriquece cada nodo con entidades detectadas (PER, ORG, LOC, MISC)
        #   El KeywordExtractor se desactiva por defecto (requiere Ollama corriendo)
        pipeline = self._build_ingestion_pipeline(
            chunk_size=size,
            chunk_overlap=overlap,
            use_entity_extractor=True,
            use_keyword_extractor=False,  # Activar solo si Ollama está disponible
        )

        # Ejecutar el pipeline — retorna nodos procesados con embeddings y metadatos
        pipeline_log = {"status": "OK", "errors": []}
        try:
            processed_nodes = pipeline.run(documents=li_documents)
        except Exception as e:
            pipeline_log = {"status": "FALLBACK", "errors": [str(e)]}
            # Fallback: pipeline sin extractores, solo chunking + embedding
            fallback_pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=size, chunk_overlap=overlap),
                    self.li_embed_model,
                ]
            )
            processed_nodes = fallback_pipeline.run(documents=li_documents)
            advertencias.append(f"EntityExtractor falló, se usó pipeline de respaldo: {str(e)}")

        # === PASO 3.5: ENRIQUECIMIENTO SEMÁNTICO COMPLETO POR CHUNK ===
        #   Genera metadatos estructurados por cada fragmento siguiendo el schema:
        #   {hierarchical_id, level, entidades_detectadas{persons, organizations},
        #    contexto_referencial, atributos_tecnicos, palabras_clave}
        import re

        # --- Resolución de Entidades (Entity Resolution) ---
        PREFIJOS_ACADEMICOS = re.compile(
            r"^(?:Ing|Dr|Dra|Mat|PhD|Msc|MSc|Mgtr|Mgst|Lcdo|Lcda|Arq|Ab|Econ|Sr|Sra|Prof)[\.\s]+",
            re.IGNORECASE
        )

        def _limpiar_persona(nombre):
            """Elimina prefijos académicos y normaliza espacios."""
            n = PREFIJOS_ACADEMICOS.sub("", nombre.strip())
            n = re.sub(r"\s+", " ", n).strip()
            return n if len(n) > 3 else None

        def _dedup_personas(personas_set):
            """Deduplica personas: 'Ing. Wladimir Paredes' y 'Wladimir Paredes Parada' → uno solo."""
            limpias = {}
            for p in personas_set:
                limpio = _limpiar_persona(p)
                if not limpio:
                    continue
                # Buscar si ya existe una versión similar (una contiene a la otra)
                encontrado = False
                for key in list(limpias.keys()):
                    if limpio.lower() in key.lower() or key.lower() in limpio.lower():
                        # Quedarse con el nombre más largo (más completo)
                        if len(limpio) > len(key):
                            del limpias[key]
                            limpias[limpio] = True
                        encontrado = True
                        break
                if not encontrado:
                    limpias[limpio] = True
            return sorted(limpias.keys())

        def _filtrar_org(nombre):
            """Filtra y limpia organizaciones: elimina sufijos basura del NER."""
            nombre = nombre.strip()
            # Limpiar sufijos basura que el NER arrastra de la oración
            nombre = re.sub(
                r"\s+(?:y\s+(?:un|una|el|la|los|las|su|sus)\s+\w+.*"
                r"|de\s+esta\s+institución.*"
                r"|La\s+formación.*"
                r"|en\s+el\s+entorno.*"
                r"|que\s+(?:se|la|el|los).*"
                r"|para\s+(?:la|el|los).*"
                r"|con\s+(?:la|el|los).*"
                r"|\s+de$)",
                "", nombre, flags=re.IGNORECASE
            ).strip()
            palabras = nombre.split()
            if len(palabras) > 6:
                return None
            # Descartar si empieza con minúscula (ej. "instituto de los...")
            if nombre and nombre[0].islower():
                return None
            if len(nombre) < 3:
                return None
            if re.match(r"^(?:La |El |Los |Las |Un |Una )", nombre, re.IGNORECASE):
                if len(palabras) > 5:
                    return None
            return nombre

        # Diccionario de normalización forzada para instituciones conocidas
        NORM_ORGS = {
            "aseguramiento de la calidad": "CACES",
            "consejo de aseguramiento": "CACES",
            "consejo de evaluación": "CEAACES",
            "educación superior": "CES",
            "consejo de educación superior": "CES",
            "secretaría de educación superior": "SENESCYT",
            "secretaría nacional de planificación": "SENPLADES",
        }

        def _limpiar_org(nombre):
            """Aplica el filtrado base y la normalización de NORM_ORGS."""
            limpio = _filtrar_org(nombre)
            if not limpio: return None
            lower = limpio.lower()
            for patron, sigla in NORM_ORGS.items():
                if patron in lower:
                    return sigla
            return limpio

        def _dedup_orgs(orgs_set):
            """Deduplica organizaciones con normalización forzada."""
            canon = {}
            for o in orgs_set:
                limpio = _limpiar_org(o)
                if not limpio:
                    continue
                key = limpio.upper().strip()
                if key not in canon or len(limpio) > len(canon[key]):
                    canon[key] = limpio
            return sorted(canon.values())

        # --- Regex compilados para extracción de metadatos por chunk ---

        # Patrones jerárquicos: 1.2, 5.2.1, Sección A, Art. 118, Capítulo IV
        PAT_HIER_NUM = re.compile(
            r"(?:^|\n)\s*(\d{1,2}(?:\.\d{1,3}){1,3})[\.\s\)]",
        )
        PAT_HIER_ALPHA = re.compile(
            r"(?:Secci[oó]n|Cap[ií]tulo|PARTE|Cl[aá]usula|Criterio|Indicador)\s+([A-Z0-9IVXLCDM]+(?:\.\d+)*)",
            re.IGNORECASE
        )

        # Organizaciones por sigla/nombre — complemento al NER que falla con ORG
        PAT_ORG_SIGLA = re.compile(
            r"\b(?:CACES|SENESCYT|CES|CEAACES|CONEA|LOES|CONESUP|"
            r"UNESCO|ONU|OEA|SENPLADES|SNIESE|SIIES|"
            r"IES|ISTT|ITSI|ISTJBA|"
            r"Ministerio\s+de\s+\w+|"
            r"Consejo\s+de\s+\w[\w\s]{3,30}|"
            r"Secretaría\s+\w[\w\s]{3,30}|"
            r"Instituto\s+\w[\w\s]{3,40}|"
            r"Universidad\s+\w[\w\s]{3,40})\b",
            re.IGNORECASE
        )

        # Contexto referencial: leyes, artículos, normas, reglamentos
        PAT_REF_LEGAL = re.compile(
            r"(?:LOES|RLOES|RFTT|RRA|CES|CACES|ISO|NTE\s*INEN|Constitución)"
            r"(?:\s+(?:Art(?:[ií]culo)?\.?\s*\d+(?:\.\d+)*))?",
            re.IGNORECASE
        )
        PAT_REF_ART = re.compile(
            r"Art(?:[ií]culo)?\.?\s*(\d+(?:\.\d+)*(?:\s*(?:numeral|literal|inciso)\s*\w+)?)",
            re.IGNORECASE
        )
        PAT_REF_NORMA = re.compile(
            r"(?:Reglamento|Decreto|Resolución|Acuerdo|Ley\s+Orgánica|Ley\s+de)"
            r"\s+[A-ZÁÉÍÓÚa-záéíóú\s]{5,50}",
            re.IGNORECASE
        )

        # Fórmulas: ecuaciones académicas con = y operadores (permite \s* para multi-línea)
        PAT_FORMULA = re.compile(
            r"\$[^$]{3,80}\$"                                                    # LaTeX inline
            r"|([A-Z]{2,})\s*=\s*\d+\s*[\*\/\+\-]\s*\\?(?:frac)?\(?\{?[A-Z\s]+\}?\)?[\/\*]?\{?[A-Z\s]*\}?\)?"  # Fórmulas académicas permisivas
            r"|[A-Z]{2,10}\s*=\s*\(?[A-Z]{2,10}\s*[\/\*]\s*[A-Z]{2,10}\)?"      # FPEI = PFPEI / TP
            r"|\\frac\{[^}]+\}\{[^}]+\}",                                        # \frac{num}{den}
            re.MULTILINE
        )

        PAT_PORCENTAJE = re.compile(r"\d+(?:[.,]\d+)?\s*%")
        # USD: OBLIGATORIAMENTE debe tener dígitos después
        PAT_MONTO_USD = re.compile(
            r"USD\.?\s*\$?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?"
            r"|\$\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\s*(?:USD|dólares)?",
            re.IGNORECASE
        )
        PAT_UNIDAD = re.compile(
            r"\d+(?:\.\d+)?\s*(?:horas|hrs|créditos|semestres|meses|años|m²|m2|kg|km|hab|docentes|estudiantes|profesores)",
            re.IGNORECASE
        )
        # Estándares: captura oraciones completas hasta punto o punto y coma
        PAT_ESTANDAR_MIN = re.compile(
            r"(?:mínimo|máximo|al\s+menos|no\s+(?:menor|mayor)\s+(?:a|de|que)|requisito|estándar)"
            r"\s*:?\s*[^.;!?\n]{5,200}[.;]?",
            re.IGNORECASE
        )

        def _corte_inteligente(texto, max_chars=180):
            """Corta texto en el último espacio antes de max_chars, sin romper palabras."""
            if len(texto) <= max_chars:
                return texto.strip()
            recortado = texto[:max_chars].rsplit(" ", 1)[0]
            return recortado.strip() + "..."

        # Keys que EntityExtractor inyecta en metadata (span-marker NER)
        ENTITY_KEYS = [
            "persons", "organizations", "locations", "animals", "biological",
            "celestial", "diseases", "events", "foods", "instruments",
            "media", "plants", "mythological", "times", "vehicles",
        ]

        # --- Procesar cada chunk y enriquecer metadatos ---
        all_persons = set()
        all_organizations = set()
        all_refs_legales = set()
        all_atributos = {"formulas": set(), "porcentajes": set(), "montos": set(), "unidades": set(), "estandares_min": set()}
        enriched_chunks_meta = []  # Metadatos enriquecidos por chunk para la UI

        # CONTEXTO PERSISTENTE: el último hierarchical_id detectado se hereda a chunks sin ID propio
        last_hier_id = None
        last_level = 0

        # Regex para detectar marca de página inyectada en paso 2
        PAT_PAGE_MARKER = re.compile(r"---\s*PAGINA\s*(\d+)\s*---", re.IGNORECASE)

        for node in processed_nodes:
            text = node.text or ""
            # Pre-limpiar texto para regex:
            # 1. Preservar separadores de párrafo (\n\n → marker temporal)
            # 2. Unir líneas dentro del mismo párrafo (\n → espacio)
            # 3. Restaurar separadores de párrafo
            text_clean = text.replace("\n\n", "⟪PARR⟫")
            text_clean = re.sub(r"\s*\n\s*", " ", text_clean)
            text_clean = text_clean.replace("⟪PARR⟫", ". ")
            text_clean = re.sub(r"\s{2,}", " ", text_clean).strip()

            # 0. Extraer pagina_ancla de la marca "--- PAGINA X ---"
            pagina_ancla = node.metadata.get("page", None)
            m_page = PAT_PAGE_MARKER.search(text)
            if m_page:
                pagina_ancla = int(m_page.group(1))

            # 1. hierarchical_id y level — con herencia de contexto
            #    EXCLUSIÓN DE ÍNDICE: si el chunk es parte del índice, forzar INDICE
            hier_id = None
            level = 0
            text_for_hier = PAT_PAGE_MARKER.sub("", text[:400]).strip()
            es_indice = (pagina_ancla and 7 <= pagina_ancla <= 9) or bool(re.search(
                r"[ÍI]ndice\s+(?:de\s+)?(?:Contenido|Tablas|Figuras|General)"
                r"|Tabla\s+de\s+Contenido"
                r"|CONTENIDO"
                r"|TABLE\s+OF\s+CONTENTS",
                text_for_hier, re.IGNORECASE
            )) or ".........." in text_for_hier
            if es_indice:
                hier_id = "INDICE"
                level = 0
                # NO actualizar last_hier_id — evitar propagar IDs del índice
            else:
                m_num = PAT_HIER_NUM.search(text_for_hier[:300])
                m_alpha = PAT_HIER_ALPHA.search(text_for_hier[:300])
                if m_num:
                    hier_id = m_num.group(1)
                    level = hier_id.count(".") + 1
                    last_hier_id = hier_id
                    last_level = level
                elif m_alpha:
                    hier_id = m_alpha.group(0).strip()
                    parts = re.findall(r"\d+", m_alpha.group(1))
                    level = len(parts) if parts else 1
                    last_hier_id = hier_id
                    last_level = level
                else:
                    # HERENCIA: si no tiene ID propio, hereda el último detectado
                    if last_hier_id:
                        hier_id = last_hier_id
                        level = last_level

            # 2. entidades_detectadas
            chunk_persons = []
            chunk_organizations = []

            # 2a. Del EntityExtractor NER (span-marker)
            for ek in ENTITY_KEYS:
                vals = node.metadata.get(ek)
                if vals:
                    items = vals if isinstance(vals, list) else [v.strip() for v in str(vals).split(",")]
                    items = [v.strip() for v in items if v.strip()]
                    if ek == "persons":
                        limpias = []
                        for p in items:
                            pl = _limpiar_persona(p)
                            if pl and pl not in limpias: limpias.append(pl)
                        chunk_persons.extend(limpias)
                        all_persons.update(limpias)
                    elif ek == "organizations":
                        limpias = []
                        for o in items:
                            ol = _limpiar_org(o)
                            if ol and ol not in limpias: limpias.append(ol)
                        chunk_organizations.extend(limpias)
                        all_organizations.update(limpias)

            # 2b. Detección de organizaciones por REGEX (complemento al NER)
            for m in PAT_ORG_SIGLA.finditer(text_clean):
                org_name = _limpiar_org(m.group(0).strip())
                if org_name and org_name not in chunk_organizations:
                    chunk_organizations.append(org_name)
                    all_organizations.add(org_name)

            # 3. contexto_referencial
            refs = set()
            for m in PAT_REF_LEGAL.finditer(text_clean):
                refs.add(m.group(0).strip())
            for m in PAT_REF_ART.finditer(text_clean):
                refs.add(f"Art. {m.group(1).strip()}")
            for m in PAT_REF_NORMA.finditer(text_clean):
                ref_text = m.group(0).strip()
                if len(ref_text) > 10:
                    refs.add(ref_text)
            all_refs_legales.update(refs)

            # 4. atributos_tecnicos (sobre texto limpio sin \n)
            chunk_atributos = {}
            formulas = list(set(m.group(0).strip() for m in PAT_FORMULA.finditer(text_clean)))
            if formulas:
                chunk_atributos["formulas"] = formulas[:5]
                all_atributos["formulas"].update(formulas)
            porcentajes = list(set(m.group(0) for m in PAT_PORCENTAJE.finditer(text_clean)))
            if porcentajes:
                chunk_atributos["porcentajes"] = porcentajes[:5]
                all_atributos["porcentajes"].update(porcentajes)
            montos = list(set(m.group(0).strip() for m in PAT_MONTO_USD.finditer(text_clean)))
            if montos:
                chunk_atributos["montos_usd"] = montos[:3]
                all_atributos["montos"].update(montos)
            unidades = list(set(m.group(0) for m in PAT_UNIDAD.finditer(text_clean)))
            if unidades:
                chunk_atributos["unidades_medida"] = unidades[:5]
                all_atributos["unidades"].update(unidades)
            estandares_raw = list(set(m.group(0).strip() for m in PAT_ESTANDAR_MIN.finditer(text_clean)))
            estandares = [_corte_inteligente(e) for e in estandares_raw]
            if estandares:
                chunk_atributos["estandares_minimos"] = estandares[:3]
                all_atributos["estandares_min"].update(estandares)

            # 5. Guardar metadata enriquecida en el nodo
            node.metadata["hierarchical_id"] = hier_id
            node.metadata["level"] = level
            node.metadata["pagina_ancla"] = pagina_ancla
            node.metadata["contexto_referencial"] = sorted(list(refs))
            node.metadata["atributos_tecnicos"] = chunk_atributos

            # Construir objeto de metadatos por chunk para muestra UI
            # Generar preview del texto (sin la marca de página)
            text_preview = PAT_PAGE_MARKER.sub("", text).strip()[:120]
            enriched_chunks_meta.append({
                "hierarchical_id": hier_id,
                "level": level,
                "pagina_ancla": pagina_ancla,
                "entidades_detectadas": {
                    "persons": chunk_persons[:5],
                    "organizations": chunk_organizations[:5],
                },
                "contexto_referencial": sorted(list(refs))[:5],
                "atributos_tecnicos": chunk_atributos,
                "texto_preview": text_preview,
            })

        # === PASO 3.5b: PALABRAS CLAVE TF-IDF (sin LLM) ===
        tfidf_keywords = []
        tfidf_bigrams = []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np

            chunk_texts = [node.text for node in processed_nodes if node.text.strip()]
            if chunk_texts:
                spanish_stops = [
                    "de", "la", "el", "en", "y", "los", "las", "del", "se", "que",
                    "un", "una", "por", "con", "no", "para", "al", "es", "lo", "su",
                    "más", "como", "ya", "son", "pero", "sus", "le", "ha", "este",
                    "entre", "cuando", "muy", "sin", "sobre", "ser", "también",
                    "otros", "fue", "cual", "desde", "hacer", "dos", "tiene", "esta",
                    "eso", "ante", "todo", "cada", "mismo", "otro", "así", "nos",
                    "parte", "después", "toda", "general", "bien", "puede", "aquí",
                    "donde", "siendo", "sido", "debe", "están", "será", "hay",
                    "todas", "estos", "según", "mediante", "forma", "caso",
                    "artículo", "numeral", "literal", "page", "página",
                ]
                # Unigrams
                try:
                    vec = TfidfVectorizer(
                        max_features=30, stop_words=spanish_stops,
                        token_pattern=r"(?u)\b[a-záéíóúñA-ZÁÉÍÓÚÑ]{3,}\b",
                        min_df=2, max_df=0.85,
                    )
                    matrix = vec.fit_transform(chunk_texts)
                    names = vec.get_feature_names_out()
                    scores = np.asarray(matrix.sum(axis=0)).flatten()
                    tfidf_keywords = [names[i] for i in scores.argsort()[::-1][:25] if scores[i] > 0]
                except Exception:
                    pass
                # Bigrams
                try:
                    vec_bi = TfidfVectorizer(
                        max_features=20, stop_words=spanish_stops,
                        ngram_range=(2, 2), min_df=2, max_df=0.85,
                        token_pattern=r"(?u)\b[a-záéíóúñA-ZÁÉÍÓÚÑ]{3,}\b",
                    )
                    bi_matrix = vec_bi.fit_transform(chunk_texts)
                    bi_names = vec_bi.get_feature_names_out()
                    bi_scores = np.asarray(bi_matrix.sum(axis=0)).flatten()
                    tfidf_bigrams = [bi_names[i] for i in bi_scores.argsort()[::-1][:15] if bi_scores[i] > 0]
                except Exception:
                    pass
        except ImportError:
            pass
        except Exception as e:
            pipeline_log["errors"].append(f"TF-IDF: {str(e)}")

        # === PASO 3.5c: CONSTRUIR RESUMEN AGREGADO PARA LA UI ===
        # Aplicar Entity Resolution a las listas agregadas
        personas_limpias = _dedup_personas(all_persons)
        orgs_limpias = _dedup_orgs(all_organizations)

        nlp_semantic_data = {
            "motor_extraccion_metadata": "LlamaIndex IngestionPipeline + Regex Enrichment",
            "entity_extractor": "span-marker NER (transformer local) + Entity Resolution",
            "modelo_ner": "tomaarsen/span-marker-mbert-base-multinerd",
            "entidades_detectadas": {
                "persons": personas_limpias[:15],
                "organizations": orgs_limpias[:15],
            },
            "contexto_referencial": sorted(list(all_refs_legales))[:20],
            "atributos_tecnicos": {
                "formulas": sorted(list(all_atributos["formulas"]))[:10],
                "porcentajes": sorted(list(all_atributos["porcentajes"]))[:10],
                "montos_usd": sorted(list(all_atributos["montos"]))[:10],
                "unidades_medida": sorted(list(all_atributos["unidades"]))[:10],
                "estandares_minimos": sorted(list(all_atributos["estandares_min"]))[:10],
            },
            "palabras_clave": tfidf_keywords[:25],
            "frases_clave_bigrams": tfidf_bigrams[:15],
            "chunks_con_jerarquia": sum(1 for c in enriched_chunks_meta if c["hierarchical_id"]),
            "chunks_con_refs_legales": sum(1 for c in enriched_chunks_meta if c["contexto_referencial"]),
            "chunks_con_atributos": sum(1 for c in enriched_chunks_meta if c["atributos_tecnicos"]),
            # Muestra: priorizar chunks CON contenido real (no portada/índice)
            "muestra_chunks_enriquecidos": [
                c for c in enriched_chunks_meta
                if c.get("hierarchical_id") or c.get("contexto_referencial") or c.get("atributos_tecnicos")
            ][:5] or enriched_chunks_meta[:5],
        }

        # === PASO 3.6: DETECCIÓN DE ESTRUCTURA JERÁRQUICA (regex general) ===
        jerarquia_sample = []
        try:
            lineas = texto_puro_completo_str.split("\n")
            capitulo_actual = ""
            seccion_actual = ""
            subseccion_actual = ""

            pat_capitulo = re.compile(
                r"^(?:CAP[ÍI]TULO|TITULO|TÍTULO|SECCI[ÓO]N|PARTE)\s+[IVXLCDM\d]+[.\s:]|"
                r"^Art[íi]culo\s+\d+[.\s:]|"
                r"^\d{1,2}[.\s]\s*[A-ZÁÉÍÓÚ]",
                re.IGNORECASE
            )
            pat_seccion = re.compile(r"^\d{1,2}\.\d{1,2}[.\s]", re.IGNORECASE)
            pat_subseccion = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{1,3}[.\s]", re.IGNORECASE)

            for linea in lineas:
                linea_limpia = linea.strip()
                if not linea_limpia or len(linea_limpia) < 3:
                    continue
                linea_match = re.sub(r'^[#*]+\s*', '', linea_limpia)
                linea_match = re.sub(r'\*{1,2}', '', linea_match).strip()

                if pat_subseccion.match(linea_match):
                    match = pat_subseccion.match(linea_match)
                    subseccion_actual = match.group(0).strip().rstrip(".")
                    jerarquia_sample.append({
                        "nivel": "subseccion",
                        "codigo": subseccion_actual,
                        "capitulo": capitulo_actual,
                        "seccion": seccion_actual,
                        "texto": linea_match[:200]
                    })
                elif pat_seccion.match(linea_match):
                    match = pat_seccion.match(linea_match)
                    seccion_actual = match.group(0).strip().rstrip(".")
                    subseccion_actual = ""
                    jerarquia_sample.append({
                        "nivel": "seccion",
                        "codigo": seccion_actual,
                        "capitulo": capitulo_actual,
                        "texto": linea_match[:200]
                    })
                elif pat_capitulo.match(linea_match):
                    match = pat_capitulo.match(linea_match)
                    capitulo_actual = match.group(0).strip().rstrip(".")
                    seccion_actual = ""
                    subseccion_actual = ""
                    jerarquia_sample.append({
                        "nivel": "capitulo",
                        "codigo": capitulo_actual,
                        "texto": linea_match[:200]
                    })
        except Exception as e:
            jerarquia_sample = [{"error": str(e)}]

        extracted_metadata = {
            "motor_extraccion": motor_usado,
            "total_paginas": resultado_extraccion.get("total_pages_real", len(paginas)),
            "caracteres_totales": total_chars,
            "palabras_totales": total_words,
            "tiempo_lectura_estimado": f"{max(1, total_words // 200)} min",
            "advertencias": advertencias,
            "estructura_semantica_nlp": nlp_semantic_data,
            "objetivo_jerarquico": jerarquia_sample,
            "pipeline_log": pipeline_log,
        }

        # === PASO 4: PREPARAR MUESTRA DE CHUNKS PARA LA UI ===
        chunks_sample = []
        for i, node in enumerate(processed_nodes):
            page = node.metadata.get("page", "?")
            chunks_sample.append({
                "id": i + 1,
                "page": page,
                "size": len(node.text),
                "text": node.text,
                "entities": node.metadata.get("entities", {}),
                "keywords": node.metadata.get("excerpt_keywords", ""),
            })

        # === PASO 5: GENERAR VECTOR DATA PARA LA UI ===
        vector_data = []
        try:
            for i, node in enumerate(processed_nodes):
                if node.embedding:
                    vector_data.append({
                        "chunk_id": i + 1,
                        "dimensions": len(node.embedding),
                        "sample": [round(x, 6) for x in node.embedding[:15]],
                        "text_preview": node.text[:100] + "..."
                    })
        except Exception as e:
            vector_data = [{"error": str(e)}]

        # === PASO 6: PREPARAR STATS (sin almacenar en Qdrant — eso lo hace push_to_qdrant) ===
        lc_docs = []
        for p in paginas:
            lc_docs.append(LCDocument(
                page_content=p["text"],
                metadata={
                    "page": p["page"],
                    "source": file_path,
                    "file_id": str(doc_id),
                    "motor": motor_usado,
                    "keywords": ", ".join(tfidf_keywords[:20]),
                    "entities": str({"persons": sorted(list(all_persons))[:5], "organizations": sorted(list(all_organizations))[:5]}),
                    "total_paginas": str(extracted_metadata.get("total_paginas", "")),
                }
            ))

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(100, round(size / 4)),
            chunk_overlap=max(20, round(overlap / 2))
        )

        # Solo calcular stats, NO almacenar en Qdrant
        docs_validos = [d for d in lc_docs if d.page_content.strip()]
        all_parent_chunks = parent_splitter.split_documents(docs_validos)
        all_child_chunks = child_splitter.split_documents(docs_validos)

        return {
            "metadata": extracted_metadata,
            "texto_puro": texto_puro_completo,
            "chunks": chunks_sample,
            "stats": {
                "total_parent_chunks": len(all_parent_chunks),
                "total_child_chunks": len(all_child_chunks),
                "llamaindex_nodes": len(processed_nodes),
                "chunk_size": size,
                "chunk_overlap": overlap,
                "strategy": strategy,
                "motor_extraccion": motor_usado,
                "pipeline": "LlamaIndex IngestionPipeline v2",
            },
            "vectors": vector_data
        }

    def push_to_qdrant(self, doc_id, file_path, paginas, metadata, size=1000, overlap=150, collection_name=None):
        """
        Envía los datos procesados al Qdrant remoto.
        Se llama EXPLÍCITAMENTE por el usuario desde el botón de la UI.
        Crea una colección única para este documento.
        """
        import time
        t_start = time.time()
        
        if not collection_name:
            # Generar nombre de colección único (formato col_YYYYMMDD_HHMMSS_id)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            collection_name = f"col_{timestamp}_{str(doc_id)[:8]}"

        # Crear colección única en Qdrant
        try:
            self.qdrant_client.get_collection(collection_name)
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "fast-all-minilm-l6-v2": qdrant_models.VectorParams(
                        size=384,  # all-MiniLM-L6-v2 produce 384 dimensiones
                        distance=qdrant_models.Distance.COSINE,
                    )
                },
            )

        # Instanciar vectorstore dinámicamente para esta colección
        dynamic_vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
            vector_name=self._get_vector_name_for_collection(collection_name),
        )

        motor_usado = metadata.get("motor_extraccion", "pymupdf")
        nlp_data = metadata.get("estructura_semantica_nlp", {})
        tfidf_keywords = nlp_data.get("palabras_clave", [])
        all_persons = nlp_data.get("entidades_detectadas", {}).get("persons", [])
        all_organizations = nlp_data.get("entidades_detectadas", {}).get("organizations", [])

        # Construir LangChain Documents con metadatos enriquecidos
        lc_docs = []
        for p in paginas:
            lc_docs.append(LCDocument(
                page_content=p["text"],
                metadata={
                    "page": p["page"],
                    "source": file_path,
                    "file_id": str(doc_id),
                    "motor": motor_usado,
                    "keywords": ", ".join(tfidf_keywords[:20]),
                    "entities": str({"persons": sorted(list(all_persons))[:5], "organizations": sorted(list(all_organizations))[:5]}),
                    "total_paginas": str(metadata.get("total_paginas", "")),
                }
            ))

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(100, round(size / 4)),
            chunk_overlap=max(20, round(overlap / 2))
        )

        temp_retriever = ParentDocumentRetriever(
            vectorstore=dynamic_vectorstore,
            docstore=self.store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        temp_retriever.add_documents(lc_docs, ids=None)

        docs_validos = [d for d in lc_docs if d.page_content.strip()]
        all_parent_chunks = parent_splitter.split_documents(docs_validos)
        all_child_chunks = child_splitter.split_documents(docs_validos)

        t_end = time.time()
        qdrant_url = getattr(settings, 'QDRANT_URL', 'http://localhost:6333')

        return {
            "status": "OK",
            "server": qdrant_url,
            "collection": collection_name,
            "parent_chunks_pushed": len(all_parent_chunks),
            "child_chunks_pushed": len(all_child_chunks),
            "documents_pushed": len(lc_docs),
            "time_seconds": round(t_end - t_start, 2),
        }

    def check_qdrant_status(self, doc_id=None):
        """
        Verifica el estado de la colección en Qdrant y si un documento específico tiene vectores.
        """
        qdrant_url = getattr(settings, 'QDRANT_URL', 'http://localhost:6333')
        
        # Intentar obtener colección específica del documento
        doc_collection = None
        try:
            if doc_id:
                from api.models import Documento
                doc = Documento.objects.filter(id=doc_id).first()
                if doc and doc.qdrant_collection_name:
                    doc_collection = doc.qdrant_collection_name
        except Exception:
            pass

        qdrant_collection = doc_collection or getattr(settings, 'QDRANT_COLLECTION_NAME', 'rag_collection')
        result = {
            "server": qdrant_url,
            "collection": qdrant_collection,
            "connected": False,
            "total_points": 0,
            "doc_points": 0,
        }
        try:
            collection_info = self.qdrant_client.get_collection(qdrant_collection)
            result["connected"] = True
            result["total_points"] = collection_info.points_count

            if doc_collection:
                result["doc_points"] = collection_info.points_count
            elif doc_id:
                count_result = self.qdrant_client.count(
                    collection_name=qdrant_collection,
                    count_filter=qdrant_models.Filter(
                        should=[
                            qdrant_models.FieldCondition(
                                key="metadata.file_id",
                                match=qdrant_models.MatchValue(value=str(doc_id)),
                            ),
                            qdrant_models.FieldCondition(
                                key="file_id",
                                match=qdrant_models.MatchValue(value=str(doc_id)),
                            )
                        ]
                    ),
                    exact=True,
                )
                result["doc_points"] = count_result.count
        except Exception as e:
            result["error"] = str(e)

        return result

    def delete_document(self, doc_id):
        """Elimina todos los vectores de un documento en Qdrant (borrando su colección única o sus puntos)."""
        try:
            from api.models import Documento
            doc = Documento.objects.filter(id=doc_id).first()
            if doc and doc.qdrant_collection_name:
                self.qdrant_client.delete_collection(doc.qdrant_collection_name)
            else:
                collection_name = getattr(settings, 'QDRANT_COLLECTION_NAME', 'rag_collection')
                self.qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(
                            should=[
                                qdrant_models.FieldCondition(
                                    key="metadata.file_id",
                                    match=qdrant_models.MatchValue(value=str(doc_id)),
                                ),
                                qdrant_models.FieldCondition(
                                    key="file_id",
                                    match=qdrant_models.MatchValue(value=str(doc_id)),
                                )
                            ]
                        )
                    ),
                )
        except Exception:
            pass

    # =========================================================================
    # UTILIDADES DE PRECISIÓN: Query Expansion + Multi-Query + RRF Fusion
    # =========================================================================

    def _normalize_query(self, query: str) -> str:
        """Normaliza la query: elimina espacios extra y pasa a minúsculas."""
        import re
        q = query.strip()
        q = re.sub(r'\s+', ' ', q)
        return q

    def _expand_query(self, query: str) -> str:
        """
        Expande la query con sinónimos del dominio académico/normativo español.
        Fundamental para mejorar la cobertura semántica de queries cortas o genéricas.
        """
        EXPANSION_MAP = {
            "modelo":         "modelo de evaluación modelo de calificación modelo de acreditación estructura del modelo",
            "calidad":        "calidad educativa aseguramiento de la calidad sistema de calidad indicadores de calidad",
            "evaluacion":     "evaluación institucional proceso de evaluación criterios de evaluación modelo de evaluación",
            "evaluación":     "evaluación institucional proceso de evaluación criterios de evaluación modelo de evaluación",
            "acreditacion":   "acreditación institucional proceso de acreditación estándares de acreditación CACES CEAACES",
            "acreditación":   "acreditación institucional proceso de acreditación estándares de acreditación CACES CEAACES",
            "criterio":       "criterios de evaluación indicadores criterios de calidad estándares parámetros",
            "criterios":      "criterios de evaluación indicadores criterios de calidad estándares parámetros",
            "indicador":      "indicadores de evaluación métricas de calidad parámetros de medición índices",
            "indicadores":    "indicadores de evaluación métricas de calidad parámetros de medición índices",
            "docente":        "docentes profesores cuerpo docente planta docente personal académico",
            "docentes":       "docentes profesores cuerpo docente planta docente personal académico",
            "investigacion":  "investigación científica producción científica proyectos de investigación publicaciones",
            "investigación":  "investigación científica producción científica proyectos de investigación publicaciones",
            "infraestructura":"infraestructura física instalaciones laboratorios aulas recursos físicos equipamiento",
            "gestion":        "gestión académica administración institucional procesos administrativos organización",
            "gestión":        "gestión académica administración institucional procesos administrativos organización",
            "carrera":        "carrera profesional programa académico oferta académica titulación",
            "estudiante":     "estudiantes alumnos matrícula población estudiantil educandos",
            "estudiantes":    "estudiantes alumnos matrícula población estudiantil educandos",
            "titulo":         "título profesional titulación graduación egresados diploma",
            "título":         "título profesional titulación graduación egresados diploma",
            "vinculacion":    "vinculación con la sociedad extensión universitaria proyectos sociales",
            "vinculación":    "vinculación con la sociedad extensión universitaria proyectos sociales",
            "planificacion":  "planificación institucional plan estratégico planificación académica",
            "planificación":  "planificación institucional plan estratégico planificación académica",
            "reglamento":     "reglamento normativa regulación disposición legal artículo",
            "requisito":      "requisito estándar mínimo condición exigencia obligación",
            "requisitos":     "requisitos estándares mínimos condiciones exigencias obligaciones",
            "porcentaje":     "porcentaje tasa índice proporción relación ratio",
            "caces":          "CACES consejo de aseguramiento de la calidad evaluación acreditación",
            "ies":            "IES institución de educación superior universidad institución",
            "loes":           "LOES ley orgánica de educación superior reglamento artículo",
            "ces":            "CES consejo de educación superior regulación normativa",
            "metodologia":    "metodología método proceso metodológico procedimiento técnica",
            "metodología":    "metodología método proceso metodológico procedimiento técnica",
            "valoracion":     "valoración calificación puntaje puntuación nota",
            "valoración":     "valoración calificación puntaje puntuación nota",
            "subcriterio":    "subcriterio subindicador componente elemento criterio",
            "subcriterios":   "subcriterios subindicadores componentes elementos criterios",
        }

        words = query.lower().strip().split()
        expansions = []
        for word in words:
            # Coincidencia exacta
            if word in EXPANSION_MAP:
                expansions.append(EXPANSION_MAP[word])
            else:
                # Coincidencia parcial (raíz de 5 chars)
                root = word[:5]
                for key, expansion in EXPANSION_MAP.items():
                    if key.startswith(root) or root.startswith(key[:5]):
                        expansions.append(expansion)
                        break

        if expansions:
            expanded = query + " " + " ".join(expansions)
        else:
            expanded = query

        # Limitar a 50 palabras para no degradar el embedding
        return " ".join(expanded.split()[:50])

    def _reciprocal_rank_fusion(self, results_list, k=60):
        """
        Combina múltiples listas de resultados usando Reciprocal Rank Fusion (RRF).
        Mejora significativamente la precisión al fusionar búsquedas con variaciones de la query.
        k=60 es el valor estándar del paper original de RRF (Cormack et al., 2009).
        """
        rrf_scores = {}
        doc_map = {}

        for results in results_list:
            for rank, (doc, cosine_score) in enumerate(results):
                # Clave única basada en los primeros 120 chars del texto
                doc_key = doc.page_content[:120].strip()
                if doc_key not in rrf_scores:
                    rrf_scores[doc_key] = 0.0
                    doc_map[doc_key] = (doc, cosine_score)
                rrf_scores[doc_key] += 1.0 / (k + rank + 1)
                # Conservar el mayor score coseno original
                if cosine_score > doc_map[doc_key][1]:
                    doc_map[doc_key] = (doc, cosine_score)

        # Ordenar por score RRF descendente
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Reconstruir resultados con score mezclado (RRF 70% + coseno 30%)
        max_rrf = 1.0 / (k + 1)
        final_results = []
        for key in sorted_keys:
            doc, cosine_score = doc_map[key]
            rrf_normalized = min(1.0, rrf_scores[key] / max_rrf)
            blended_score = 0.7 * rrf_normalized + 0.3 * cosine_score
            final_results.append((doc, blended_score))

        return final_results

    def _interpret_score(self, score: float) -> str:
        """
        Interpreta el score de similitud coseno con umbrales calibrados para
        texto técnico en español. Los umbrales estándar (0.70, 0.90) son
        irreales para documentos académicos en español con queries cortas.
        """
        if score >= 0.85:   return "Identidad/Casi Idéntico"
        elif score >= 0.65: return "Alta Similitud"
        elif score >= 0.45: return "Similitud Media"
        elif score >= 0.28: return "Similitud Baja"
        else:               return "Disímiles"

    def _bm25_search(self, query: str, candidates: list, k: int):
        """
        Búsqueda BM25 (léxica/sparse) sobre una lista de candidatos.
        BM25 captura coincidencias exactas de palabras clave que los vectores
        densos a veces pierden cuando el contexto semántico difiere.

        Args:
            query: Query del usuario (normalizada)
            candidates: Lista de (doc, score) obtenida de la búsqueda densa
            k: Número de resultados a devolver

        Returns:
            Lista de (doc, bm25_score_normalizado) ordenada por BM25 desc.
        """
        from rank_bm25 import BM25Okapi
        import re

        if not candidates:
            return []

        def _tokenize(text: str) -> list:
            """Tokeniza texto en español: minúsculas, elimina puntuación, split."""
            text = text.lower()
            text = re.sub(r'[^\w\sáéíóúñü]', ' ', text)
            tokens = text.split()
            # Stopwords básicas en español
            STOP = {
                'de','la','el','en','y','los','las','del','se','que','un','una',
                'por','con','no','para','al','es','lo','su','más','como','ya',
                'son','pero','sus','le','ha','este','entre','cuando','muy','sin',
                'sobre','ser','también','fue','cual','desde','tiene','esta','ante',
                'todo','cada','mismo','otro','así','nos','parte','toda','bien',
                'puede','donde','siendo','sido','debe','están','será','hay','todas',
                'estos','según','mediante','forma','caso','página','pagina',
            }
            return [t for t in tokens if t not in STOP and len(t) > 1]

        # Construir corpus BM25 con los textos de los candidatos
        corpus = [doc.page_content for doc, _ in candidates]
        tokenized_corpus = [_tokenize(text) for text in corpus]
        tokenized_query = _tokenize(query)

        if not tokenized_query or not any(tokenized_corpus):
            return []

        bm25 = BM25Okapi(tokenized_corpus)
        raw_scores = bm25.get_scores(tokenized_query)

        # Normalizar scores BM25 a [0, 1]
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        normalized_scores = [s / max_score for s in raw_scores]

        # Construir lista de (doc, score_normalizado)
        scored = [(candidates[i][0], normalized_scores[i]) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # =========================================================================

    def vector_search(self, query, doc_id=None, k=5, collection_name=None):
        """
        Consulta DIRECTA a Qdrant con Multi-Query Retrieval + RRF Fusion.
        Incluye: normalización, expansión de query, variaciones semánticas
        y fusión de resultados para máxima precisión.
        """
        try:
            if not collection_name and doc_id:
                try:
                    from api.models import Documento
                    doc = Documento.objects.filter(id=doc_id).first()
                    if doc and doc.qdrant_collection_name:
                        collection_name = doc.qdrant_collection_name
                except Exception:
                    pass

            target_collection = collection_name or getattr(settings, 'QDRANT_COLLECTION_NAME', 'rag_collection')

            # Instanciar vectorstore para la colección correspondiente
            current_vectorstore = self.vectorstore
            if collection_name:
                current_vectorstore = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=collection_name,
                    embedding=self.embeddings,
                    vector_name=self._get_vector_name_for_collection(collection_name),
                )

            qdrant_filter = None
            if not collection_name and doc_id:
                qdrant_filter = qdrant_models.Filter(
                    should=[
                        qdrant_models.FieldCondition(
                            key="metadata.file_id",
                            match=qdrant_models.MatchValue(value=str(doc_id)),
                        ),
                        qdrant_models.FieldCondition(
                            key="file_id",
                            match=qdrant_models.MatchValue(value=str(doc_id)),
                        )
                    ]
                )

            # ── PASO 1: Normalizar y expandir la query ──────────────────────
            normalized = self._normalize_query(query)
            expanded = self._expand_query(normalized)

            # ── PASO 2: Variaciones de query para cobertura semántica ────────
            query_variations = [query, expanded]
            if len(query.strip().split()) <= 3:
                query_variations += [
                    f"información sobre {query}",
                    f"¿Qué es {query}?",
                ]

            # ── PASO 3: CANAL DENSO — búsqueda vectorial en Qdrant ──────────
            # Recuperar un pool amplio de candidatos para que BM25 tenga
            # suficiente material para re-rankear con coincidencias léxicas.
            pool_size = max(k * 4, 20)
            dense_results_lists = []
            for q_variant in query_variations[:4]:
                try:
                    partial = current_vectorstore.similarity_search_with_score(
                        q_variant, k=pool_size, filter=qdrant_filter
                    )
                    if partial:
                        dense_results_lists.append(partial)
                except Exception:
                    pass

            # Pool único sin duplicados (mayor score gana si aparece varias veces)
            pool_map = {}
            for results in dense_results_lists:
                for doc, score in results:
                    key = doc.page_content[:120].strip()
                    if key not in pool_map or score > pool_map[key][1]:
                        pool_map[key] = (doc, score)
            candidate_pool = list(pool_map.values())

            # ── PASO 4: CANAL SPARSE — BM25 sobre el pool de candidatos ──────
            bm25_results = self._bm25_search(
                query=normalized,
                candidates=candidate_pool,
                k=pool_size,
            )

            # ── PASO 5: FUSIÓN HÍBRIDA con RRF ───────────────────────────────
            # Convertir el pool denso a lista ordenada por score desc
            dense_ranked = sorted(candidate_pool, key=lambda x: x[1], reverse=True)

            # RRF sobre ambos canales
            hybrid_fused = self._reciprocal_rank_fusion(
                [dense_ranked, bm25_results], k=60
            )[:k]

            # Construir mapa de scores originales para enriquecer la respuesta
            dense_score_map = {doc.page_content[:120].strip(): sc for doc, sc in dense_ranked}
            bm25_score_map  = {doc.page_content[:120].strip(): sc for doc, sc in bm25_results}

            # ── PASO 6: Formatear resultados con scores de ambos canales ─────
            output = []
            for doc, hybrid_score in hybrid_fused:
                key = doc.page_content[:120].strip()
                d_score = dense_score_map.get(key, 0.0)
                b_score = bm25_score_map.get(key, 0.0)
                similarity_pct = max(0, min(100, round(hybrid_score * 100, 2)))
                output.append({
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", "?"),
                    "file_id": doc.metadata.get("file_id", "?"),
                    # Score híbrido (RRF de ambos canales)
                    "score_cosine": round(hybrid_score, 4),
                    "similarity_pct": similarity_pct,
                    "interpretation": self._interpret_score(hybrid_score),
                    # Scores individuales por canal (transparencia)
                    "dense_score": round(d_score, 4),
                    "bm25_score": round(b_score, 4),
                })

            return {
                "results": output,
                "query": query,
                "query_expanded": expanded,
                "total": len(output),
                "collection_used": target_collection,
                "search_mode": "Multi-Query RRF",
            }
        except Exception as e:
            return {"error": str(e), "results": [], "query": query, "total": 0}

    def ask_question(self, question, model_name="qwen2:1.5b", use_rerank=False, use_metadata_filter=False, doc_id=None, collection_name=None):
        try:
            t_start = time.time()
            process = psutil.Process(os.getpid())

            if not collection_name and doc_id:
                try:
                    from api.models import Documento
                    doc = Documento.objects.filter(id=doc_id).first()
                    if doc and doc.qdrant_collection_name:
                        collection_name = doc.qdrant_collection_name
                except Exception:
                    pass

            current_vectorstore = self.vectorstore
            current_retriever = self.retriever

            if collection_name:
                current_vectorstore = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=collection_name,
                    embedding=self.embeddings,
                    vector_name=self._get_vector_name_for_collection(collection_name),
                )
                current_retriever = ParentDocumentRetriever(
                    vectorstore=current_vectorstore,
                    docstore=self.store,
                    child_splitter=RecursiveCharacterTextSplitter(chunk_size=250),
                    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000),
                )

            # 1. Filtrado de Metadatos
            search_kwargs = {"k": 5}
            qdrant_filter = None
            if not collection_name and use_metadata_filter and doc_id:
                qdrant_filter = qdrant_models.Filter(
                    should=[
                        qdrant_models.FieldCondition(
                            key="metadata.file_id",
                            match=qdrant_models.MatchValue(value=str(doc_id)),
                        ),
                        qdrant_models.FieldCondition(
                            key="file_id",
                            match=qdrant_models.MatchValue(value=str(doc_id)),
                        )
                    ]
                )
                search_kwargs["filter"] = qdrant_filter

            current_retriever.search_kwargs = search_kwargs

            # Búsqueda Vectorial
            t_retrieval_start = time.time()

            # 2. Similitud Semántica
            raw_docs_with_scores = current_vectorstore.similarity_search_with_score(
                question, k=5, filter=qdrant_filter
            )
            top_score = 0
            if raw_docs_with_scores:
                distancia = raw_docs_with_scores[0][1]
                top_score = max(0, min(100, round(distancia * 100, 2)))

            # Búsqueda real usando el Parent-Child Retriever
            retrieved_docs = current_retriever.invoke(question)

            # 3. Re-Ranking con FlashRank
            rerank_log = "Inactivo"
            if use_rerank and retrieved_docs and self.reranker:
                rerank_request = RerankRequest(
                    query=question,
                    passages=[{"id": str(i), "text": doc.page_content} for i, doc in enumerate(retrieved_docs)]
                )
                rerank_results = self.reranker.rerank(rerank_request)

                reranked_docs = []
                for result in rerank_results:
                    idx = int(result["id"])
                    if idx < len(retrieved_docs):
                        reranked_docs.append(retrieved_docs[idx])

                if reranked_docs:
                    retrieved_docs = reranked_docs
                    top_rerank_score = round(rerank_results[0]["score"], 4) if rerank_results else 0
                    rerank_log = f"FlashRank ms-marco | Top Score: {top_rerank_score}"
            elif use_rerank and not self.reranker:
                rerank_log = "FlashRank no disponible (error al cargar)"

            t_retrieval_end = time.time()
            retrieval_time = round(t_retrieval_end - t_retrieval_start, 3)

            # Generación con LLM
            llm = ChatOllama(model=model_name, base_url=settings.OLLAMA_BASE_URL, temperature=0)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=current_retriever,
                return_source_documents=True
            )

            response = qa_chain.invoke({"query": question})

            t_end = time.time()
            ram_end = process.memory_info().rss / (1024 * 1024)
            total_duration = round(t_end - t_start, 2)
            generation_time = round(total_duration - retrieval_time, 2)

            return {
                "answer": response["result"],
                "transformation_log": [
                    {"step": "FILTRO META", "data": "Inactivo (Colección dedicada)" if collection_name else (f"doc_id={doc_id}" if use_metadata_filter else "Inactivo"), "status": "OK"},
                    {"step": "SIMILITUD", "data": f"Score: {top_score}%", "status": "OK"},
                    {"step": "RE-RANKING", "data": rerank_log, "status": "OK"},
                    {"step": "RECUPERACIÓN", "data": f"{len(retrieved_docs)} chunks encontrados", "status": "DONE"},
                    {"step": "OLLAMA IA", "data": f"Modelo: {model_name} generó en {generation_time}s", "status": "ACTIVE"}
                ],
                "telemetry": {
                    "total_time": f"{total_duration}s",
                    "retrieval_time": f"{retrieval_time}s",
                    "ram_usage": f"{round(ram_end, 1)}MB",
                    "chunks": len(retrieved_docs),
                    "fidelity": f"{top_score}%",
                    "cpu_load": f"{psutil.cpu_percent()}%"
                }
            }
        except Exception as e:
            import traceback
            error_msg = str(e)
            if "ConnectionError" in error_msg or "Connection refused" in error_msg:
                error_msg = "No se pudo conectar con Ollama. ¿Está ejecutándose en tu PC?"
            elif "not found" in error_msg.lower():
                error_msg = f"El modelo '{model_name}' no está instalado en Ollama. Abre la terminal y ejecuta: ollama pull {model_name}"

            return {
                "answer": f"⚠️ **ERROR DEL MOTOR IA:**\n{error_msg}",
                "transformation_log": [{"step": "CRITICAL", "data": "Fallo en motor RAG/Ollama", "status": "ERROR"}],
                "telemetry": {"total_time": "0s", "retrieval_time": "0s", "ram_usage": "0MB", "chunks": 0, "fidelity": "0%", "cpu_load": "0%"}
            }

    def intent_route_and_search(self, question, model_name="qwen2:1.5b", doc_id=None, collection_name=None):
        """
        Ruteador de Intenciones Inteligente (Intent Router).
        Clasifica la consulta, determina la estrategia en Qdrant y ejecuta.
        """
        import re
        import json
        t_start = time.time()
        process = psutil.Process(os.getpid())

        try:
            # Obtener colección dinámica si no se pasa explícitamente pero hay un documento seleccionado
            if not collection_name and doc_id:
                try:
                    from api.models import Documento
                    doc = Documento.objects.filter(id=doc_id).first()
                    if doc and doc.qdrant_collection_name:
                        collection_name = doc.qdrant_collection_name
                except Exception:
                    pass

            target_collection = collection_name or getattr(settings, 'QDRANT_COLLECTION_NAME', 'rag_collection')

            # Instanciar vectorstore y retriever dinámicamente
            current_vectorstore = self.vectorstore
            current_retriever = self.retriever

            if collection_name:
                current_vectorstore = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=collection_name,
                    embedding=self.embeddings,
                    vector_name=self._get_vector_name_for_collection(collection_name),
                )
                current_retriever = ParentDocumentRetriever(
                    vectorstore=current_vectorstore,
                    docstore=self.store,
                    child_splitter=RecursiveCharacterTextSplitter(chunk_size=250),
                    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000),
                )

            # 1. Clasificación de la Intención vía Ollama
            system_prompt = (
                "Eres un Ruteador de Intenciones (Intent Router) experto en sistemas RAG.\n"
                "Tu trabajo es analizar la pregunta de un usuario y clasificar su intención en una de las siguientes categorías:\n"
                "1. factual: Para buscar datos específicos, números, fechas, nombres, artículos o leyes concretas.\n"
                "2. resumen: Para pedir resúmenes, síntesis globales, ideas principales o resúmenes de todo el documento.\n"
                "3. comparacion: Para contrastar conceptos, ventajas/desventajas o diferencias entre secciones.\n"
                "4. conceptual: Para definiciones, explicaciones de términos o conceptos ('¿qué es X?', 'explica Y').\n\n"
                "Debes responder ÚNICAMENTE en formato JSON con la siguiente estructura, sin texto adicional antes ni después:\n"
                "{\n"
                '  "intent": "factual" | "resumen" | "comparacion" | "conceptual",\n'
                '  "confidence": <número entero entre 0 y 100>,\n'
                '  "reasoning": "Explicación breve de por qué se eligió esta categoría en base a las palabras clave del usuario"\n'
                "}"
            )

            from langchain_core.messages import SystemMessage, HumanMessage
            llm_router = ChatOllama(model=model_name, base_url=settings.OLLAMA_BASE_URL, temperature=0)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Pregunta del usuario: '{question}'")
            ]
            
            router_response = llm_router.invoke(messages)
            classification_text = router_response.content

            # Parsear la respuesta JSON del ruteador
            json_match = re.search(r"\{.*?\}", classification_text, re.DOTALL)
            if json_match:
                try:
                    classification = json.loads(json_match.group(0))
                except Exception:
                    classification = {"intent": "factual", "confidence": 60, "reasoning": "Error parseando JSON, fallback a factual."}
            else:
                classification = {"intent": "factual", "confidence": 50, "reasoning": "No se detectó JSON estructurado, fallback a factual."}

            intent = classification.get("intent", "factual").lower()
            confidence = classification.get("confidence", 80)
            reasoning = classification.get("reasoning", "Detección basada en patrones semánticos.")

            # 2. Configurar la estrategia de búsqueda
            use_metadata_filter = True
            use_rerank = False
            k = 5
            strategy_name = "Búsqueda Estándar"
            strategy_params = {}
            strategy_steps = []

            # Si es colección dedicada, el filtro de metadatos no es requerido
            meta_filter_desc = "Inactivo (Colección dedicada)" if collection_name else "Activo (document_id)"

            if intent == "factual":
                strategy_name = "Búsqueda Factual de Alta Precisión"
                k = 4
                use_metadata_filter = True
                use_rerank = False
                strategy_params = {
                    "K (vecinos)": k,
                    "Filtro de metadatos": meta_filter_desc,
                    "FlashRank Re-ranking": "Inactivo (Prioriza velocidad y coincidencia exacta)"
                }
                strategy_steps = [
                    "Identificación de consulta factual (fechas, datos numéricos o nombres específicos).",
                    "Configuración de ventana estrecha (K=4) para enfocar similitud local.",
                    "Búsqueda en colección dedicada Qdrant." if collection_name else "Filtrado estricto por ID de documento para excluir ruido de otros PDF.",
                    "Búsqueda vectorial en Qdrant por distancia Coseno."
                ]
            elif intent == "resumen":
                strategy_name = "Búsqueda Ampliada para Resúmenes"
                k = 8
                use_metadata_filter = True
                use_rerank = True
                strategy_params = {
                    "K (vecinos)": k,
                    "Filtro de metadatos": meta_filter_desc,
                    "FlashRank Re-ranking": "Activo (Para consolidar los 8 chunks más representativos)"
                }
                strategy_steps = [
                    "Identificación de solicitud global o de resumen del documento.",
                    "Configuración de ventana extendida (K=8) para capturar la mayor cantidad de páginas posibles.",
                    "Activación de Re-ranking neuronal (FlashRank) para ordenar la síntesis y evitar fragmentos duplicados.",
                    "Generación enfocada en sintetizar el contexto total sin alucinaciones."
                ]
            elif intent == "comparacion":
                strategy_name = "Búsqueda Cruzada Neuronal"
                k = 6
                use_metadata_filter = True
                use_rerank = True
                strategy_params = {
                    "K (vecinos)": k,
                    "Filtro de metadatos": meta_filter_desc,
                    "FlashRank Re-ranking": "Activo (Prioridad máxima para ordenar por contraste semántico)"
                }
                strategy_steps = [
                    "Identificación de contraste, pros/contras o comparación de secciones.",
                    "Configuración de ventana media (K=6) para cruzar fragmentos distantes.",
                    "Uso obligatorio de FlashRank Reranker para ordenar de forma lógica los argumentos opuestos.",
                    "Generación comparativa estructurada por el LLM."
                ]
            elif intent == "conceptual":
                strategy_name = "Búsqueda de Definición y Conceptos"
                k = 3
                use_metadata_filter = True
                use_rerank = False
                strategy_params = {
                    "K (vecinos)": k,
                    "Filtro de metadatos": meta_filter_desc,
                    "FlashRank Re-ranking": "Inactivo (Prioridad a definiciones exactas)"
                }
                strategy_steps = [
                    "Identificación de consulta conceptual o definición de términos.",
                    "Configuración de ventana compacta (K=3) para aislar la definición principal del concepto.",
                    "Búsqueda vectorial directa en Qdrant.",
                    "Generación con baja temperatura para mantener la precisión conceptual de los manuales."
                ]
            else:
                intent = "factual"
                strategy_name = "Búsqueda Estándar"
                k = 5
                strategy_params = {"K (vecinos)": k, "Filtro de metadatos": meta_filter_desc}
                strategy_steps = ["Intención no clasificada. Usando ruteo estándar RAG."]

            # 3. Búsqueda vectorial real en Qdrant
            qdrant_filter = None
            if not collection_name and use_metadata_filter and doc_id:
                qdrant_filter = qdrant_models.Filter(
                    should=[
                        qdrant_models.FieldCondition(
                            key="metadata.file_id",
                            match=qdrant_models.MatchValue(value=str(doc_id)),
                        ),
                        qdrant_models.FieldCondition(
                            key="file_id",
                            match=qdrant_models.MatchValue(value=str(doc_id)),
                        )
                    ]
                )

            # Obtener chunks con scores para mostrar en la interfaz
            raw_docs_with_scores = current_vectorstore.similarity_search_with_score(
                question, k=k, filter=qdrant_filter
            )
            
            chunks_output = []
            top_score = 0
            for doc, score in raw_docs_with_scores:
                similarity_pct = max(0, min(100, round(score * 100, 2)))
                if top_score == 0:
                    top_score = similarity_pct

                chunks_output.append({
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", "?"),
                    "score_cosine": round(score, 4),
                    "similarity_pct": similarity_pct,
                    "interpretation": self._interpret_score(score),
                })

            # Búsqueda real usando el retriever
            current_retriever.search_kwargs = {"k": k}
            if qdrant_filter:
                current_retriever.search_kwargs["filter"] = qdrant_filter
            
            retrieved_docs = current_retriever.invoke(question)

            # Reranking
            rerank_log = "Inactivo"
            if use_rerank and retrieved_docs and self.reranker:
                rerank_request = RerankRequest(
                    query=question,
                    passages=[{"id": str(i), "text": doc.page_content} for i, doc in enumerate(retrieved_docs)]
                )
                rerank_results = self.reranker.rerank(rerank_request)

                reranked_docs = []
                for result in rerank_results:
                    idx = int(result["id"])
                    if idx < len(retrieved_docs):
                        reranked_docs.append(retrieved_docs[idx])

                if reranked_docs:
                    retrieved_docs = reranked_docs
                    top_rerank_score = round(rerank_results[0]["score"], 4) if rerank_results else 0
                    rerank_log = f"FlashRank ms-marco | Top Score: {top_rerank_score}"
            elif use_rerank and not self.reranker:
                rerank_log = "FlashRank no disponible (error al cargar)"

            # 4. Generación final con LLM
            llm_gen = ChatOllama(model=model_name, base_url=settings.OLLAMA_BASE_URL, temperature=0)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_gen,
                chain_type="stuff",
                retriever=current_retriever,
                return_source_documents=True
            )

            response = qa_chain.invoke({"query": question})
            answer = response["result"]

            t_end = time.time()
            ram_end = process.memory_info().rss / (1024 * 1024)
            total_duration = round(t_end - t_start, 2)

            return {
                "intent": intent,
                "confidence": confidence,
                "reasoning": reasoning,
                "methodology": {
                    "strategy": strategy_name,
                    "params": strategy_params,
                    "steps": strategy_steps,
                    "rerank_log": rerank_log
                },
                "answer": answer,
                "results": chunks_output,
                "telemetry": {
                    "total_time": f"{total_duration}s",
                    "ram_usage": f"{round(ram_end, 1)}MB",
                    "chunks": len(retrieved_docs),
                    "fidelity": f"{top_score}%",
                    "cpu_load": f"{psutil.cpu_percent()}%"
                },
                "collection_used": target_collection
            }

        except Exception as e:
            import traceback
            error_msg = str(e)
            if "ConnectionError" in error_msg or "Connection refused" in error_msg:
                error_msg = "No se pudo conectar con Ollama. ¿Está ejecutándose en tu PC?"
            elif "not found" in error_msg.lower():
                error_msg = f"El modelo '{model_name}' no está instalado en Ollama. Abre la terminal y ejecuta: ollama pull {model_name}"

            return {
                "intent": "error",
                "confidence": 0,
                "reasoning": f"Error: {error_msg}",
                "methodology": {
                    "strategy": "Error de ejecución",
                    "params": {},
                    "steps": ["Fallo al conectar con Ollama o procesar consulta."],
                    "rerank_log": "Inactivo"
                },
                "answer": f"⚠️ **ERROR DEL MOTOR IA EN INTENT ROUTER:**\n{error_msg}",
                "results": [],
                "telemetry": {
                    "total_time": "0s",
                    "ram_usage": "0MB",
                    "chunks": 0,
                    "fidelity": "0%",
                    "cpu_load": "0%"
                }
            }
