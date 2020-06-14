import csv
import json
csv_reader = csv.reader(open('./data/3518793057_labled.csv', encoding='utf-8'))
output_path = "./data/Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-new/"
i = 0
for row in csv_reader:
    i += 1
    text = row[2]
    date = row[6]
    label = row[13]
    res = dict()
    res['text'] = text
    res['date'] = date
    res['label'] = int(label)
    json_str = json.dumps(res)
    with open(output_path + str(i)+"3518793057.json", "w") as f:
        f.write(json_str)

    