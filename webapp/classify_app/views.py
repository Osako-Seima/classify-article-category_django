from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import UrlForm
from .models import NaiveBayes

import re

def index(request):
    template = loader.get_template('classify_app/index.html')
    context = {'form': UrlForm()}
    return HttpResponse(template.render(context, request))

def predict(request):
    if not request.method == 'POST':
        return
        redirect('classify_app:index')
    
    form = UrlForm(request.POST)
    
    if not form.is_valid():
        raise ValueError('Formが不正です')

    pattern = "https?://[\w/:%#\$&\?\(\)~\.=\+\-]+"
    link = re.findall(pattern, str(form))

    target = NaiveBayes()
    category = target.classify(link[0])

    template = loader.get_template('classify_app/result.html')
    
    context = {
        'url':form,
        'category':category
        }
    
    return HttpResponse(template.render(context, request))
    
