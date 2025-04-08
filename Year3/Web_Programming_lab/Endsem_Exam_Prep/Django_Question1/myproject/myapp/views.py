from django.shortcuts import render,redirect
from .models import Works, Lives
from .forms import WorksForm , LivesForm

def home(request):
    insert_form=WorksForm()
    search_form=LivesForm()
    results=None

    if request.method == 'POST':
        if 'insert' in request.POST:
            insert_form = WorksForm(request.POST)
            if insert_form.is_valid():
                insert_form.save()
        elif 'search' in request.POST:
            search_form = LivesForm(request.POST)
            if search_form.is_valid():
                company = search_form.cleaned_data['company_name']
                results = Works.objects.filter(company_name=company).select_related()
                # Join manually with Lives
                results = [
                    {'name': r.person_name, 'city': Lives.objects.get(person_name=r.person_name).city}
                    for r in results if Lives.objects.filter(person_name=r.person_name).exists()
                ]

    return render(request, 'home.html', {
        'insert_form': insert_form,
        'search_form': search_form,
        'results': results,
    })

# Create your views here.
