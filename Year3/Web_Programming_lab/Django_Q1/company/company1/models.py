from django.db import models

class WORKS(models.Model):
    person_name = models.CharField(max_length=100)
    company_name = models.CharField(max_length=100)
    salary = models.IntegerField()

    def __str__(self):
        return f"{self.person_name} - {self.company_name}"

class LIVES(models.Model):
    person_name = models.CharField(max_length=100)
    street = models.CharField(max_length=100)
    city = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.person_name} lives in {self.city}"
