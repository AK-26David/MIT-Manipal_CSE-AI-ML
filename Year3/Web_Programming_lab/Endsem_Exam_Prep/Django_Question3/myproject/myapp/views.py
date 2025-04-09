from django.shortcuts import render
from .forms import WORKSForm, CompanySearchForm
from .models import WORKS, LIVES

def home(request):
    return render(request, 'myapp/home.html')

def insert_works(request):
    if request.method == 'POST':
        form = WORKSForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, 'myapp/success.html')
    else:
        form = WORKSForm()
    return render(request, 'myapp/insert_works.html', {'form': form})

def search_people(request):
    results = None
    if request.method == 'POST':
        form = CompanySearchForm(request.POST)
        if form.is_valid():
            company = form.cleaned_data['company_name']
            results = WORKS.objects.filter(company_name=company).values_list('person_name', flat=True)
            data = [(name, LIVES.objects.get(person_name=name).city) for name in results if LIVES.objects.filter(person_name=name).exists()]
            return render(request, 'myapp/search_results.html', {'form': form, 'data': data})
    else:
        form = CompanySearchForm()
    return render(request, 'myapp/search_form.html', {'form': form})
