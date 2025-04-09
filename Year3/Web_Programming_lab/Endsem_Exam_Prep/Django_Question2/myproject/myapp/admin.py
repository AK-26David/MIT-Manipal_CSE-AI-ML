from django.contrib import admin
from .models import Book

@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'published_date', 'genre', 'available', 'created_at', 'updated_at')
    readonly_fields = ('created_at', 'updated_at')  # Prevents manual changes