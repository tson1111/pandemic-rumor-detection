from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
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


def templateTest(request):
    latest_question_list = ['a', 'b', 'c']
    template = loader.get_template('mainapp/index.html')
    context = {
        'latest_question_list': latest_question_list,
    }
    return HttpResponse(template.render(context, request))


def index(request):
    templateTest(request)

    # with open("template.html", "r") as src:
    #     content = Template(src.read())
    # html = content.render(Context())
    # return HttpResponse(html)

    # return HttpResponse(jsonToHtml(sample1)+jsonToHtml(sample2))
