from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    return HttpResponse("Hello, Django from Web Programming Lab Week 5!")

# Create your views here.
