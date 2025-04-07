from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('insert_works/', views.insert_works, name='insert_works'),
    path('retrieve_employees/', views.retrieve_employees, name='retrieve_employees'),
]
