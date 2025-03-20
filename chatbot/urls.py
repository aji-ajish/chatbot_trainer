from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
    path('update_course_type/<int:file_id>/', views.update_course_type, name='update_course_type'),
    path('files/', views.list_files, name='list_files'),
    path('files/<int:file_id>/', views.delete_file, name='delete_file'),
    path('process_query/', views.process_query, name='process_query'),
    path('chat_history/', views.chat_history, name='chat_history'),
    path('list_keywords/', views.list_keywords, name='list_keywords'),
    path('edit_keyword/', views.edit_keyword, name='edit_keyword'),
    path('delete_keyword/<int:file_id>/', views.delete_keyword, name='delete_keyword'),

]
