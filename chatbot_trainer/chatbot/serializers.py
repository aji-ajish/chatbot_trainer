from rest_framework import serializers
from .models import UploadedFile

class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedFile
        fields = ['id', 'course_id', 'filename', 'filetype', 'filepath', 'request_token', 'response_token', 'upload_date']
