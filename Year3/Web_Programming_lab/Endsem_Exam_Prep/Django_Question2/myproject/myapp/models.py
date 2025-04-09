from django.db import models

class Book(models.Model):
    title=models.CharField(max_length=100)
    author=models.CharField(max_length=100)
    published_date=models.DateField()
    genre=models.CharField(max_length=100)
    available=models.BooleanField()

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

def __str__(self):
    return self.title



