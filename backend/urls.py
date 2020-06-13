from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('age/', views.guess_age),
    path('color/', views.colorize),

]