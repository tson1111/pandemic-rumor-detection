from django.shortcuts import render
from django.http import HttpResponse
import json

sample = ""
with open('./mainapp/sample.json') as f:
    sample = json.load(f)


def jsonToHtml(jsonObj):
    s = ""
    s += "<h1>"+str(jsonObj['text'])+"</h1>"
    for key, value in jsonObj.items():
        s += "<p>" + str(key)+": "+str(value)+"</p>"
    return s
    # json.dumps(jsonObj, indent=2, ensure_ascii=False)


def index(request):
    return HttpResponse(jsonToHtml(sample)+jsonToHtml(sample))
