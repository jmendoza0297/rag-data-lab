from rest_framework import serializers
from .models import Documento

class DocumentoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Documento
        fields = ['id', 'archivo', 'nombre', 'fecha_subida', 'estado', 'error_message', 'metadata_sample', 'chunks_sample']
        read_only_fields = ['id', 'estado', 'error_message', 'nombre']
