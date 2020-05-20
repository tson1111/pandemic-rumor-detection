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
    with open('./mainapp/data/result520.json', encoding='utf-8') as f:
        data = json.load(f)['data']
    for i in data:
        i['ner'] = ""
        for field in ['org', 'company', 'person', 'job']:
            if i[field] is not '':
                i['ner'] += i[field]+' '
    template = loader.get_template('mainapp/index.html')
    context = {
        'contents': data,
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
