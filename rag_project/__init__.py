# Este archivo importa la app de Celery para que se cargue
# automáticamente cuando Django arranca.
from .celery import app as celery_app

__all__ = ('celery_app',)
