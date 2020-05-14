from django.http import HttpResponse
from django.template import loader
import json


def templateTest(request):
    latest_question_list = ['a', 'b', 'c']
    template = loader.get_template('mainapp/templateTest.html')
    context = {
        'latest_question_list': latest_question_list,
    }
    return HttpResponse(template.render(context, request))


def index(request):
    with open('./mainapp/data/sample1.json', encoding='utf-8') as f:
        sample1 = json.load(f)
    contents = [sample1] * 15
    template = loader.get_template('mainapp/index.html')
    context = {
        'contents': contents,
    }
    return HttpResponse(template.render(context, request))

    # return templateTest(request)

    # with open("template.html", "r") as src:
    #     content = Template(src.read())
    # html = content.render(Context())
    # return HttpResponse(html)

    # return HttpResponse(jsonToHtml(sample1)+jsonToHtml(sample2))


def jsonToHtml(jsonObj):
    """
    unused old version
    """

    s = ""
    s += "<h1>"+str(jsonObj['text'])+"</h1>"
    for key, value in jsonObj.items():
        s += "<p>" + str(key)+": "+str(value)+"</p>"
    return s
    # json.dumps(jsonObj, indent=2, ensure_ascii=False)
