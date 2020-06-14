# coding: utf-8
import os
import random
from multiprocessing import cpu_count
import numpy as np
import shutil
import paddle
import paddle.fluid as fluid
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import json
import csv
import codecs
import fool    # FoolNLTK


# use_cuda为False,表示运算场所为CPU;use_cuda为True,表示运算场所为GPU 
use_cuda = False 
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)  
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 用训练好的模型进行预测并输出预测结果
# 创建执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
infer_exe.run(fluid.default_startup_program())

save_path = './work/infer_model/'

# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=infer_exe)


# 获取数据
def get_data(sentence):
    # 读取数据字典
    with open('./data/dict.txt', 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    dict_txt = dict(dict_txt)
    # 把字符串数据转换成列表数据
    keys = dict_txt.keys()
    data = []
    for s in sentence:
        # 判断是否存在未知字符
        if not s in keys:
            s = '<unk>'
        data.append(int(dict_txt[s]))
    return data

content = []
date = []
data = []
# 获取数据
with open('./data/small.csv', 'r') as csvfile:
    f = csv.reader(csvfile, delimiter=',', quotechar='\"')
    next(f, None)  # skip the headers
    for row in f:
        date.append(row[1])
        content.append(row[2])
        data.append(get_data(row[2]))
csvfile.close()

# 获取每句话的单词数量
base_shape = [[len(c) for c in data]]

# 生成预测数据
tensor_words = fluid.create_lod_tensor(data, base_shape, place)

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: tensor_words},
                 fetch_list=target_var)

# 输出结果
with open('./data/result.json', 'w', encoding='utf-8-sig') as file:
    # writer = csv.writer(file)
    for i in range(len(data)):
        Dict = {'content':content[i], "rumor":1, "location":"", 
        "time":date[i], "org":"", "company":"", "person":"", "job":""}
        lab = np.argsort(result)[0][i][-1] # 获取结果概率最大的label
        Dict['rumor'] = str(lab)
        words, ners = fool.analysis(content[i])
        for entity in ners[0]:
            if 'location' in entity:
                Dict['location'] = entity[3]
            # elif 'time' in entity:
            #     Dict['time'] = entity[3]
            elif 'org' in entity:
                Dict['org'] = entity[3]
            elif 'company' in entity:
                Dict['company'] = entity[3]
            elif 'person' in entity:
                Dict['person'] = entity[3]
            elif 'job' in entity:
                Dict['job'] = entity[3]
        # writer.writerow([json.dumps(Dict, ensure_ascii=False)])
        json.dump(Dict, file, ensure_ascii=False)
        file.write(',\n')
file.close()



    