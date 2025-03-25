from django.db import models

class UploadedFile(models.Model):
    course_id = models.IntegerField()
    filename = models.CharField(max_length=255)
    filetype = models.CharField(max_length=255)
    filepath = models.CharField(max_length=255)
    type = models.IntegerField(default=0)  # New field: 0 or 1  
    request_token = models.IntegerField()
    response_token = models.IntegerField()
    upload_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.filename


class FileKeywords(models.Model):
    course_id = models.IntegerField()
    file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name="keywords")
    keywords = models.TextField()  # Comma-separated keywords

    def __str__(self):
        return f"Keywords for {self.file.filename}"


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
