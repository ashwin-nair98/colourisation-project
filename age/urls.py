from django.urls import path
from age import views

urlpatterns = [
    path('age/', views.guess_age),
]