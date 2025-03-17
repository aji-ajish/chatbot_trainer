from rest_framework import serializers
from .models import UploadedFile , ChatHistory

class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedFile
        fields = ['id', 'course_id', 'filename', 'filetype', 'filepath', 'request_token', 'response_token', 'upload_date']

class ChatHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatHistory
        fields = '__all__'  # Or specify fields like ['id', 'user_id', 'course_id', 'message', 'response']
