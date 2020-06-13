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
            if i[field] != '':
                i['ner'] += i[field]+' '
    template = loader.get_template('mainapp/index.html')
    context = {'contents': data, }
    return HttpResponse(template.render(context, request))

# search


def predict(request):
    request.encoding = 'utf-8'
    if 'q' in request.GET and request.GET['q']:
        message = request.GET['q']
        #! call
        result = {'content': message}
        # result = api.predict(request.GET['q'])
    else:
        message = '提交内容为空。'
        result = {'content': message}

    template = loader.get_template('mainapp/predict.html')
    return HttpResponse(template.render({'content': result}, request))


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
