import fool
import csv

raw_data = []
# 获取数据
with open('./data/small.csv', 'r') as csvfile:
    f = csv.reader(csvfile, delimiter=',', quotechar='\"')
    next(f, None)  # skip the headers
    for row in f:
        raw_data.append(row[2])
csvfile.close()

# print(fool.pos_cut(text))
words, ners = fool.analysis(raw_data)
# print(words)

# NERS结果类型：
    # location
    # company
    # time
    # org
    # person
    # job
    # ....
# print(ners)
for item in ners:
    if item == []:
        continue
    Dict = {'title':'', 'content':'', "rumor":1, "location":"", 
        "time":"", "org":"", "company":"", "person":"", "job":""}
    for entity in item:
        if 'location' in entity:
            print(entity)