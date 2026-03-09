from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path("admin/",   admin.site.urls),
    path("",         views.home,        name="home"),
    path("loan/",    views.loan_form,   name="loan_form"),
    path("result/",  views.loan_result, name="loan_result"),
]