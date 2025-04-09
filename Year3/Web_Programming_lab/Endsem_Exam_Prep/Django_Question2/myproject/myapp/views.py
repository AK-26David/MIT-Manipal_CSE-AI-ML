from django.shortcuts import render,redirect
from .forms import BookForm
from .models import Book

def add_list_book(request):
    if request.method=='POST':
        form=BookForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('list_book')
    else:
        form=BookForm()
    available_books=Book.objects.filter(available=True).order_by('-published_date')

    return render(request, 'list_book.html', {
        'form': form,
        'books': available_books
    })


