from celery import shared_task
from .models import Documento
from .services.rag_logic import RAGManager
import logging

logger = logging.getLogger(__name__)

@shared_task
def process_document_task(doc_id, strategy="recursive", size=1000, overlap=150):
    try:
        doc = Documento.objects.get(id=doc_id)
        doc.estado = 'PROCESANDO'
        # Guardar la configuración usada
        doc.chunk_strategy = strategy
        doc.chunk_size = size
        doc.chunk_overlap = overlap
        doc.save()

        rag = RAGManager()
        rag.ingest_document(
            doc.archivo.path, 
            doc.id, 
            strategy=strategy, 
            size=int(size), 
            overlap=int(overlap)
        )
        
        doc.estado = 'COMPLETADO'
        doc.save()
        return f"Documento {doc_id} procesado con éxito ({strategy})"
    except Exception as e:
        logger.error(f"Error procesando documento {doc_id}: {str(e)}")
        if 'doc' in locals():
            doc.estado = 'ERROR'
            doc.error_message = str(e)
            doc.save()
        return str(e)
