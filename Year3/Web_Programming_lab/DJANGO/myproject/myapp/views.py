from django.shortcuts import render, redirect
from .forms import ProductForm
from .models import Product

def product_create(request):
    if request.method == 'POST':
        form = ProductForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('product_index')
    else:
        form = ProductForm()
    return render(request, 'product_create.html', {'form': form})

def product_index(request):
    products = Product.objects.all()
    return render(request, 'product_index.html', {'products': products})
