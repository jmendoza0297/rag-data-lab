import os
import time
import psutil

# ¡PARCHE CRÍTICO PARA WINDOWS!
# Desactiva las barras de progreso de HuggingFace que causan OSError [Errno 22] en sys.stderr
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

from django.conf import settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document as LCDocument
from .extractor import extract as pdf_extract
from langchain_community.chat_models import ChatOllama
from langchain_classic.chains import RetrievalQA

# FlashRank: Re-ranking profesional
from flashrank import Ranker, RerankRequest


class RAGManager:
    def __init__(self):
        self.chroma_path = os.path.join(settings.BASE_DIR, 'storage', 'chroma')
        self.docstore_path = os.path.join(settings.BASE_DIR, 'storage', 'docstore')
        os.makedirs(self.chroma_path, exist_ok=True)
        os.makedirs(self.docstore_path, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=self.chroma_path
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

    def ingest_document(self, file_path, doc_id, strategy="recursive", size=1000, overlap=150, extraction_motor="pymupdf", apply_deadmau5=True):
        """
        Pipeline completo de ingesta:
        1. Extracción con el motor seleccionado por el usuario (pymupdf/pdfplumber/pypdf2/ocr)
        2. Texto limpio y estructurado por página
        3. Fragmentación (Chunking) con RecursiveCharacterTextSplitter
        4. Generación de Embeddings y almacenamiento en ChromaDB
        """
        # === PASO 1: EXTRACCIÓN CON MOTOR SELECCIONADO ===
        resultado_extraccion = pdf_extract(file_path, motor=extraction_motor, apply_deadmau5=apply_deadmau5)
        paginas = resultado_extraccion.get("pages", [])
        texto_puro_completo = resultado_extraccion.get("texto_completo", "")
        motor_usado = resultado_extraccion.get("motor_usado", extraction_motor)
        advertencias = resultado_extraccion.get("advertencias", [])

        # Convertir las páginas a LangChain Documents para el pipeline
        docs = []
        for p in paginas:
            docs.append(LCDocument(
                page_content=p["text"],
                metadata={
                    "page": p["page"],
                    "source": file_path,
                    "file_id": str(doc_id),
                    "motor": motor_usado,
                }
            ))

        # Metadatos enriquecidos para la UI
        extracted_metadata = {
            "source": file_path,
            "total_pages": resultado_extraccion.get("total_pages_real", len(paginas)),
            "paginas_con_texto": len(paginas),
            "motor_extraccion": motor_usado,
            "advertencias": advertencias,
        }

        # === PASO 3: FRAGMENTACIÓN (CHUNKING) ===
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(100, round(size / 4)),
            chunk_overlap=max(20, round(overlap / 2))
        )

        # Generar TODOS los chunks para la muestra visual (UI)
        docs_validos = [d for d in docs if d.page_content.strip()]
        all_parent_chunks = parent_splitter.split_documents(docs_validos)
        all_child_chunks = child_splitter.split_documents(docs_validos)

        # Preparar muestra completa de chunks para la UI
        chunks_sample = []
        for i, chunk in enumerate(all_parent_chunks):
            page = chunk.metadata.get("page", "?")
            chunks_sample.append({
                "id": i + 1,
                "page": page,
                "size": len(chunk.page_content),
                "text": chunk.page_content
            })

        # === PASO 4: EMBEDDINGS Y ALMACENAMIENTO ===
        vector_data = []
        try:
            # Generar embeddings de TODOS los chunks para la UI
            for i, chunk in enumerate(all_parent_chunks):
                raw_vector = self.embeddings.embed_query(chunk.page_content)
                vector_data.append({
                    "chunk_id": i + 1,
                    "dimensions": len(raw_vector),
                    "sample": [round(x, 6) for x in raw_vector[:15]],
                    "text_preview": chunk.page_content[:100] + "..."
                })
        except Exception as e:
            vector_data = [{"error": str(e)}]

        # Almacenar en ChromaDB usando el ParentDocumentRetriever
        temp_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        temp_retriever.add_documents(docs, ids=None)

        return {
            "metadata": extracted_metadata,
            "texto_puro": texto_puro_completo,
            "chunks": chunks_sample,
            "stats": {
                "total_parent_chunks": len(all_parent_chunks),
                "total_child_chunks": len(all_child_chunks),
                "chunk_size": size,
                "chunk_overlap": overlap,
                "strategy": strategy,
                "motor_extraccion": motor_usado,
            },
            "vectors": vector_data
        }

    def delete_document(self, doc_id):
        try:
            self.vectorstore.delete(where={"file_id": str(doc_id)})
        except:
            pass

    def vector_search(self, query, doc_id=None, k=5):
        """
        Consulta DIRECTA a la base de datos vectorial (ChromaDB).
        NO usa LLM. Solo retorna los chunks más similares con su score.
        Esto permite al usuario verificar que los embeddings y chunks están correctos.
        """
        try:
            search_kwargs = {"k": k}
            if doc_id:
                search_kwargs["filter"] = {"file_id": str(doc_id)}

            results = self.vectorstore.similarity_search_with_score(query, **search_kwargs)

            output = []
            for doc, score in results:
                # Convertir distancia L2 a porcentaje de similitud
                similarity_pct = max(0, min(100, round((1.0 - (score / 2.0)) * 100, 2)))
                
                # Interpretación técnica según la tabla del usuario
                interpretation = "Desconocida"
                if score <= 0.2: interpretation = "Identidad/Casi Idéntico"
                elif score <= 0.6: interpretation = "Alta Similitud"
                elif score <= 1.1: interpretation = "Similitud Media"
                elif score <= 2.0: interpretation = "Similitud Baja"
                else: interpretation = "Disímiles"

                output.append({
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", "?"),
                    "file_id": doc.metadata.get("file_id", "?"),
                    "score_l2": round(score, 4),
                    "similarity_pct": similarity_pct,
                    "interpretation": interpretation
                })

            return {"results": output, "query": query, "total": len(output)}
        except Exception as e:
            return {"error": str(e), "results": [], "query": query, "total": 0}

    def ask_question(self, question, model_name="qwen2:1.5b", use_rerank=False, use_metadata_filter=False, doc_id=None):
        try:
            t_start = time.time()
            process = psutil.Process(os.getpid())

            # 1. Filtrado de Metadatos (Metadata Filtering)
            search_kwargs = {"k": 5}
            if use_metadata_filter and doc_id:
                search_kwargs["filter"] = {"file_id": str(doc_id)}

            self.retriever.search_kwargs = search_kwargs

            # Búsqueda Vectorial
            t_retrieval_start = time.time()

            # 2. Similitud Semántica (Similarity Score)
            raw_docs_with_scores = self.vectorstore.similarity_search_with_score(question, **search_kwargs)
            top_score = 0
            if raw_docs_with_scores:
                distancia = raw_docs_with_scores[0][1]
                top_score = max(0, min(100, round((1.0 - (distancia / 2.0)) * 100, 2)))

            # Búsqueda real usando el Parent-Child Retriever
            retrieved_docs = self.retriever.invoke(question)

            # 3. Re-Ranking con FlashRank (Neuronal, no keyword-based)
            rerank_log = "Inactivo"
            if use_rerank and retrieved_docs and self.reranker:
                rerank_request = RerankRequest(
                    query=question,
                    passages=[{"id": str(i), "text": doc.page_content} for i, doc in enumerate(retrieved_docs)]
                )
                rerank_results = self.reranker.rerank(rerank_request)

                # Reordenar los documentos según el score neuronal
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
            llm = ChatOllama(model=model_name, base_url="http://localhost:11434", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,
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
                    {"step": "FILTRO META", "data": f"doc_id={doc_id}" if use_metadata_filter else "Inactivo", "status": "OK"},
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
