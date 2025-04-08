from django import forms
from .models import Works, Lives

class WorksForm(forms.ModelForm):
    class Meta:
        model=Works
        fields=['person_name','company_name','salary']


class LivesForm(forms.ModelForm):
    class Meta:
        company_name=forms.CharField(label='Company name', max_length=100)

