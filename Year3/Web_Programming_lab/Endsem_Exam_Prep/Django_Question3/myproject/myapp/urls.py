from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('insert/', views.insert_works, name='insert_works'),
    path('search/', views.search_people, name='search_people'),
]
