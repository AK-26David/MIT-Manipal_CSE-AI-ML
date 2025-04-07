from django.shortcuts import render, redirect
from .forms import WORKSForm, CompanySearchForm
from .models import WORKS, LIVES

def index(request):
    return render(request, 'jobs/index.html')

def insert_works(request):
    if request.method == 'POST':
        form = WORKSForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('index')
    else:
        form = WORKSForm()
    return render(request, 'jobs/insert_works.html', {'form': form})

def retrieve_employees(request):
    employees = None
    if request.method == 'POST':
        form = CompanySearchForm(request.POST)
        if form.is_valid():
            company = form.cleaned_data['company_name']
            works = WORKS.objects.filter(company_name=company)
            employees = []
            for w in works:
                try:
                    lives = LIVES.objects.get(person_name=w.person_name)
                    employees.append((w.person_name, lives.city))
                except LIVES.DoesNotExist:
                    employees.append((w.person_name, "City Unknown"))
    else:
        form = CompanySearchForm()
    return render(request, 'jobs/retrieve_employees.html', {'form': form, 'employees': employees})
