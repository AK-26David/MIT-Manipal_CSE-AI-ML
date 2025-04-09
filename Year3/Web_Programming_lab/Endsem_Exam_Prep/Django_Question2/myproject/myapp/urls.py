from django.urls import path
from . import views

urlpatterns = [
    path('', views.add_list_book, name='list_book'),
]
