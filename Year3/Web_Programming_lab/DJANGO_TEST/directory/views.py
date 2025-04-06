from django.shortcuts import render, redirect
from .forms import CategoryForm, PageForm
from .models import Category, Page

def home(request):
    categories = Category.objects.all()
    return render(request, 'directory/home.html', {'categories': categories})

def add_category(request):
    if request.method == 'POST':
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = CategoryForm()
    return render(request, 'directory/add_category.html', {'form': form})

def add_page(request):
    if request.method == 'POST':
        form = PageForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = PageForm()
    return render(request, 'directory/add_page.html', {'form': form})
