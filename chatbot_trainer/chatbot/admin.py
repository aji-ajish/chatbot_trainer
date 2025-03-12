from django.contrib import admin
from .models import UploadedFile, ChatHistory

# Register your models here.
admin.site.register(UploadedFile)
admin.site.register(ChatHistory)