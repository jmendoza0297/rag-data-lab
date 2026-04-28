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
            if doc.estado != 'PENDIENTE':
                return Response({"error": "Solo se pueden procesar documentos PENDIENTES"}, status=status.HTTP_400_BAD_REQUEST)
            
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
    """Consulta DIRECTA a ChromaDB sin LLM. Para validar embeddings y chunks."""
    def post(self, request, uuid):
        query = request.data.get('query', '')
        k = int(request.data.get('k', 5))
        
        if not query:
            return Response({"error": "La consulta es obligatoria"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            rag = get_rag_manager()
            result = rag.vector_search(query=query, doc_id=uuid, k=k)
            return Response(result)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
