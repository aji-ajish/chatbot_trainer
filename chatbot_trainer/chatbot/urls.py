from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
    path('files/', views.list_files, name='list_files'),
    path('files/<int:file_id>/', views.delete_file, name='delete_file'),
    path('process_query/', views.process_query, name='process_query'),
    path('chat_history/', views.chat_history, name='chat_history'),
]
