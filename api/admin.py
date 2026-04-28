from django.contrib import admin
from .models import Documento

@admin.register(Documento)
class DocumentoAdmin(admin.ModelAdmin):
    list_display = ('nombre', 'estado', 'fecha_subida')
    list_filter = ('estado',)
    search_fields = ('nombre',)
    readonly_fields = ('id',)
