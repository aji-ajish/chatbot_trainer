from django.db import models

class UploadedFile(models.Model):
    course_id = models.IntegerField()
    filename = models.CharField(max_length=255)
    filetype = models.CharField(max_length=255)
    filepath = models.CharField(max_length=255)
    request_token = models.IntegerField()
    response_token = models.IntegerField()
    upload_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.filename
