from django.urls import path
from vqe import views
urlpatterns = [
#path('', views.home, name='home'),
#path('vqe/', views.vqe, name='vqe')]
path('vqe/', views.vqesim)]
