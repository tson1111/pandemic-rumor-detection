from django.shortcuts import render
from django.http import HttpResponse
import json

sample1 = ""
sample2 = ""
with open('./mainapp/sample1.json') as f:
    sample1 = json.load(f)
with open('./mainapp/sample2.json') as f:
    sample2 = json.load(f)

def jsonToHtml(jsonObj):
    s = ""
    s += "<h1>"+str(jsonObj['text'])+"</h1>"
    for key, value in jsonObj.items():
        s += "<p>" + str(key)+": "+str(value)+"</p>"
    return s
    # json.dumps(jsonObj, indent=2, ensure_ascii=False)


def index(request):
    return HttpResponse(jsonToHtml(sample1)+jsonToHtml(sample2))
