import uuid
from django.db import models

class Documento(models.Model):
    ESTADOS = [
        ('PENDIENTE', 'Pendiente'),
        ('PROCESANDO', 'Procesando'),
        ('COMPLETADO', 'Completado'),
        ('ERROR', 'Error'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    archivo = models.FileField(upload_to='pdfs/')
    nombre = models.CharField(max_length=255)
    fecha_subida = models.DateTimeField(auto_now_add=True)
    estado = models.CharField(max_length=20, choices=ESTADOS, default='PENDIENTE')
    error_message = models.TextField(null=True, blank=True)
    
    # Configuración de Fragmentación
    chunk_strategy = models.CharField(max_length=50, default='recursive')
    chunk_size = models.IntegerField(default=1000)
    chunk_overlap = models.IntegerField(default=150)
    
    # Inspector de Datos
    metadata_sample = models.JSONField(null=True, blank=True)
    chunks_sample = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"{self.nombre} ({self.estado})"
