from rest_framework import serializers
from .models import Documento

class DocumentoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Documento
        fields = ['id', 'archivo', 'nombre', 'fecha_subida', 'estado', 'error_message', 'metadata_sample', 'chunks_sample', 'qdrant_synced', 'qdrant_sync_date', 'qdrant_collection_name']
        read_only_fields = ['id', 'estado', 'error_message', 'nombre']
