import os
import threading
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Documento
from .serializers import DocumentoSerializer
from .services.rag_logic import RAGManager

# Instancia global para evitar recargas constantes (Optimización de RAM)
_rag_instance = None

def get_rag_manager():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGManager()
    return _rag_instance

def _process_document_sync(doc_id, strategy, size, overlap, extraction_motor="pymupdf", apply_deadmau5=True):
    """Procesa el documento en un hilo separado (sin necesidad de Redis/Celery)"""
    try:
        doc = Documento.objects.get(id=doc_id)
        doc.estado = 'PROCESANDO'
        doc.chunk_strategy = strategy
        doc.chunk_size = size
        doc.chunk_overlap = overlap
        doc.save()

        rag = get_rag_manager()
        file_path = doc.archivo.path

        extraction = rag.ingest_document(
            file_path,
            doc_id,
            strategy=strategy,
            size=int(size),
            overlap=int(overlap),
            extraction_motor=extraction_motor,
            apply_deadmau5=apply_deadmau5
        )

        # Guardar las muestras para el Inspector de Datos y Vector Lab
        doc.metadata_sample = {
            "info": extraction.get("metadata", {}),
            "texto_puro": extraction.get("texto_puro", ""),
            "vectors": extraction.get("vectors", []),
            "stats": extraction.get("stats", {}),
            "motor_extraccion": extraction.get("metadata", {}).get("motor_extraccion", extraction_motor),
            "advertencias": extraction.get("metadata", {}).get("advertencias", []),
        }
        doc.chunks_sample = extraction.get("chunks", [])
        
        doc.estado = 'COMPLETADO'
        doc.qdrant_synced = False
        doc.qdrant_sync_date = None
        doc.save()
    except Exception as e:
        import traceback
        doc = Documento.objects.get(id=doc_id)
        doc.estado = 'ERROR'
        doc.error_message = traceback.format_exc()
        doc.save()

class HomeView(APIView):
    def get(self, request):
        return render(request, 'index.html')

class DocumentUploadView(APIView):
    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No se envió ningún archivo"}, status=status.HTTP_400_BAD_REQUEST)
        
        doc = Documento.objects.create(archivo=file, nombre=file.name)
        return Response(DocumentoSerializer(doc).data, status=status.HTTP_201_CREATED)

class DocumentListView(APIView):
    def get(self, request):
        docs = Documento.objects.all().order_by('-fecha_subida')
        return Response(DocumentoSerializer(docs, many=True).data)

class DocumentStatusView(APIView):
    def get(self, request, uuid):
        try:
            doc = Documento.objects.get(id=uuid)
            return Response(DocumentoSerializer(doc).data)
        except Documento.DoesNotExist:
            return Response({"error": "No encontrado"}, status=status.HTTP_404_NOT_FOUND)

class DocumentProcessView(APIView):
    def post(self, request, uuid):
        try:
            doc = Documento.objects.get(id=uuid)
            if doc.estado == 'PROCESANDO':
                return Response({"error": "El documento ya está siendo procesado"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Capturar parámetros del Lab con seguridad
            try:
                strategy = request.data.get('strategy', 'recursive')
                size = int(request.data.get('size', 1000))
                overlap_percent = int(request.data.get('overlap', 15))
                overlap = int(size * (overlap_percent / 100))
                extraction_motor = request.data.get('extraction_motor', 'pymupdf')
                apply_deadmau5 = request.data.get('apply_deadmau5', True)
            except:
                strategy, size, overlap, extraction_motor, apply_deadmau5 = 'recursive', 1000, 150, 'pymupdf', True
            
            # Procesar directamente (Síncrono) para evitar bloqueos de hilos en Windows
            _process_document_sync(doc.id, strategy, size, overlap, extraction_motor, apply_deadmau5)
            
            return Response({"message": "Motor encendido - Procesamiento Completado."})
        except Exception as e:
            import traceback
            return Response({"error": traceback.format_exc()}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DocumentDeleteView(APIView):
    def delete(self, request, uuid):
        try:
            doc = Documento.objects.get(id=uuid)
            # Borrado rápido de archivo y DB
            if doc.archivo and os.path.exists(doc.archivo.path):
                os.remove(doc.archivo.path)
            
            # Intento de borrado de vectores solo si el motor ya está encendido
            global _rag_instance
            if _rag_instance:
                _rag_instance.delete_document(doc.id)
            
            doc.delete()
            return Response({"message": "Eliminado"})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

@method_decorator(csrf_exempt, name='dispatch')
class ChatView(APIView):
    def post(self, request, uuid):
        pregunta = request.data.get('pregunta')
        modelo = request.data.get('modelo', 'qwen2:1.5b') # Modelo por defecto
        
        # Opciones RAG Avanzadas
        use_rerank = request.data.get('use_rerank', False)
        use_metadata_filter = request.data.get('use_metadata_filter', False)
        
        if not pregunta:
            return Response({"error": "La pregunta es obligatoria"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            rag = get_rag_manager()
            result = rag.ask_question(
                question=pregunta, 
                model_name=modelo,
                use_rerank=use_rerank,
                use_metadata_filter=use_metadata_filter,
                doc_id=uuid  # Pasamos el ID del documento para el filtrado
            )
            return Response(result)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(csrf_exempt, name='dispatch')
class VectorSearchView(APIView):
    """Consulta DIRECTA a Qdrant sin LLM. Para validar embeddings y chunks."""
    def post(self, request, uuid):
        query = request.data.get('query', '')
        k = int(request.data.get('k', 5))
        collection_name = request.data.get('collection_name')
        
        if not query:
            return Response({"error": "La consulta es obligatoria"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            rag = get_rag_manager()
            result = rag.vector_search(query=query, doc_id=uuid, k=k, collection_name=collection_name)
            return Response(result)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(csrf_exempt, name='dispatch')
class QdrantPushView(APIView):
    """Envía explícitamente los datos procesados al Qdrant remoto."""
    def post(self, request, uuid):
        try:
            from django.utils import timezone

            doc = Documento.objects.get(id=uuid)
            if doc.estado != 'COMPLETADO':
                return Response({"error": "El documento debe estar COMPLETADO antes de enviar a Qdrant"}, status=status.HTTP_400_BAD_REQUEST)

            if not doc.metadata_sample:
                return Response({"error": "No hay datos procesados para enviar"}, status=status.HTTP_400_BAD_REQUEST)

            rag = get_rag_manager()

            # Reconstruir páginas desde el texto puro almacenado
            texto_puro = doc.metadata_sample.get("texto_puro", "")
            meta_info = doc.metadata_sample.get("info", {})
            stats = doc.metadata_sample.get("stats", {})
            file_path = doc.archivo.path if doc.archivo else ""

            # Reconstruir lista de páginas desde los chunks almacenados
            # Cada chunk tiene page y text
            paginas = []
            chunks = doc.chunks_sample or []
            pages_seen = {}
            for chunk in chunks:
                page_num = chunk.get("page", 1)
                if page_num not in pages_seen:
                    pages_seen[page_num] = {"page": page_num, "text": chunk.get("text", "")}
                else:
                    pages_seen[page_num]["text"] += "\n" + chunk.get("text", "")
            paginas = sorted(pages_seen.values(), key=lambda x: x["page"])

            if not paginas:
                # Fallback: usar texto puro como una sola página
                paginas = [{"page": 1, "text": texto_puro}]

            chunk_size = stats.get("chunk_size", 1000)
            chunk_overlap = stats.get("chunk_overlap", 150)
            custom_collection_name = request.data.get("collection_name")

            result = rag.push_to_qdrant(
                doc_id=uuid,
                file_path=file_path,
                paginas=paginas,
                metadata=meta_info,
                size=chunk_size,
                overlap=chunk_overlap,
                collection_name=custom_collection_name,
            )

            # Actualizar estado de sincronización
            doc.qdrant_synced = True
            doc.qdrant_sync_date = timezone.now()
            doc.qdrant_collection_name = result.get("collection")
            doc.save()

            return Response(result)
        except Documento.DoesNotExist:
            return Response({"error": "Documento no encontrado"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            import traceback
            return Response({"error": traceback.format_exc()}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(csrf_exempt, name='dispatch')
class QdrantStatusView(APIView):
    """Consulta el estado de la colección en Qdrant y del documento."""
    def get(self, request, uuid):
        try:
            doc = Documento.objects.get(id=uuid)
            rag = get_rag_manager()
            result = rag.check_qdrant_status(doc_id=uuid)
            result["qdrant_synced"] = doc.qdrant_synced
            result["qdrant_sync_date"] = doc.qdrant_sync_date.isoformat() if doc.qdrant_sync_date else None
            return Response(result)
        except Documento.DoesNotExist:
            return Response({"error": "Documento no encontrado"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(csrf_exempt, name='dispatch')
class IntentSearchView(APIView):
    """Búsqueda RAG utilizando el Ruteador de Intención Inteligente."""
    def post(self, request, uuid):
        pregunta = request.data.get('query')
        modelo = request.data.get('model', 'qwen2:1.5b')
        collection_name = request.data.get('collection_name')
        
        if not pregunta:
            return Response({"error": "La consulta es obligatoria"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            rag = get_rag_manager()
            result = rag.intent_route_and_search(
                question=pregunta,
                model_name=modelo,
                doc_id=uuid,
                collection_name=collection_name
            )
            return Response(result)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class QdrantCollectionsListView(APIView):
    """Obtiene la lista de todas las colecciones existentes directamente desde Qdrant"""
    def get(self, request):
        try:
            rag = get_rag_manager()
            collections_response = rag.qdrant_client.get_collections()
            collections = [col.name for col in collections_response.collections]
            return Response({"collections": collections})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

