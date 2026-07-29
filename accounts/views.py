from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def login_view(request):
    return HttpResponse("Login Page")

def register_view(request):
    return HttpResponse("Register Page")

def logout_view(request):
    return HttpResponse("Logout")