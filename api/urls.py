from django.urls import path
from .views import (
    HomeView, DocumentUploadView, DocumentListView, 
    DocumentStatusView, DocumentProcessView, DocumentDeleteView, 
    ChatView, VectorSearchView, QdrantPushView, QdrantStatusView,
    IntentSearchView, QdrantCollectionsListView
)

urlpatterns = [
    path('', HomeView.as_view(), name='home'),
    path('api/documents/', DocumentListView.as_view(), name='list'),
    path('api/documents/upload/', DocumentUploadView.as_view(), name='upload'),
    path('api/documents/<uuid:uuid>/status/', DocumentStatusView.as_view(), name='status'),
    path('api/documents/<uuid:uuid>/process/', DocumentProcessView.as_view(), name='process'),
    path('api/documents/<uuid:uuid>/delete/', DocumentDeleteView.as_view(), name='delete'),
    path('api/documents/<uuid:uuid>/chat/', ChatView.as_view(), name='chat'),
    path('api/documents/<uuid:uuid>/vector-search/', VectorSearchView.as_view(), name='vector_search'),
    path('api/documents/<uuid:uuid>/push-qdrant/', QdrantPushView.as_view(), name='push_qdrant'),
    path('api/documents/<uuid:uuid>/qdrant-status/', QdrantStatusView.as_view(), name='qdrant_status'),
    path('api/documents/<uuid:uuid>/intent-search/', IntentSearchView.as_view(), name='intent_search'),
    path('api/qdrant/collections/', QdrantCollectionsListView.as_view(), name='qdrant_collections'),
]

