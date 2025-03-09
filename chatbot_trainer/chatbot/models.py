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


class ChatHistory(models.Model):
    user_id = models.IntegerField()
    course_id = models.IntegerField()
    user_query = models.TextField()
    response = models.TextField()
    req_token = models.IntegerField()
    res_token = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"User {self.user_id} - Course {self.course_id} - {self.timestamp}"