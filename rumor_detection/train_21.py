# 解压原始数据集，将Rumor_Dataset.zip解压至data目录下
import zipfile
import os
import random
from PIL import Image
from PIL import ImageEnhance
import json

model_save_dir = "./work/infer_model_21/"

#分别为谣言数据、非谣言数据、全部数据的文件路径
rumor_class_dirs = os.listdir("./data/Chinese_Rumor_Dataset-master/CED_Dataset/rumor-repost/")
non_rumor_class_dirs = os.listdir("./data/Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-repost/")
original_microblog = "./data/Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog/"
non_rumor_new_original = "./data/Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-new/"
non_rumor_new_dirs = os.listdir("./data/Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-new/")
rumor_new_original = "./data/Chinese_Rumor_Dataset-master/CED_Dataset/rumor-new/"
rumor_new_dirs = os.listdir("./data/Chinese_Rumor_Dataset-master/CED_Dataset/rumor-new/")

#谣言标签为0，非谣言标签为1
rumor_label="0"
non_rumor_label="1"

#分别统计谣言数据与非谣言数据的总数
rumor_num = 0
non_rumor_num = 0

all_rumor_list = []
all_non_rumor_list = []

#解析谣言数据
for rumor_class_dir in rumor_class_dirs: 
    if(rumor_class_dir != '.DS_Store'):
        #遍历谣言数据，并解析
        with open(original_microblog + rumor_class_dir, 'r') as f:
            rumor_content = f.read()
        rumor_dict = json.loads(rumor_content)
        all_rumor_list.append(rumor_label+"\t"+rumor_dict["text"]+"\n")
        rumor_num += 1
for rumor_new_dir in rumor_new_dirs: 
    if(rumor_new_dir != '.DS_Store'):
        with open(rumor_new_original + rumor_new_dir, 'r') as f3:
            rumor_content = f3.read()
        rumor_dict = json.loads(rumor_content)
        all_rumor_list.append(rumor_label+"\t"+rumor_dict["text"]+"\n")
        rumor_num += 1

#解析非谣言数据
for non_rumor_class_dir in non_rumor_class_dirs: 
    if(non_rumor_class_dir != '.DS_Store'):
        with open(original_microblog + non_rumor_class_dir, 'r') as f2:
            non_rumor_content = f2.read()
        non_rumor_dict = json.loads(non_rumor_content)
        all_non_rumor_list.append(non_rumor_label+"\t"+non_rumor_dict["text"]+"\n")
        non_rumor_num += 1
for non_rumor_new_dir in non_rumor_new_dirs: 
    if(non_rumor_new_dir != '.DS_Store'):
        with open(non_rumor_new_original + non_rumor_new_dir, 'r') as f3:
            non_rumor_content = f3.read()
        non_rumor_dict = json.loads(non_rumor_content)
        all_non_rumor_list.append(non_rumor_label+"\t"+non_rumor_dict["text"]+"\n")
        non_rumor_num += 1
print("谣言数据总量为：" + str(rumor_num))
print("非谣言数据总量为：" + str(non_rumor_num))

#全部数据进行乱序后写入all_data.txt
all_data_path=model_save_dir + "all_data.txt"
all_data_list = all_rumor_list + all_non_rumor_list
random.shuffle(all_data_list)
with open(all_data_path, 'w') as f: #在生成all_data.txt之前，首先将其清空
    f.seek(0)
    f.truncate() 
with open(all_data_path, 'a') as f:
    for data in all_data_list:
        f.write(data) 

# 导入必要的包
import os
from multiprocessing import cpu_count
import numpy as np
import shutil
import paddle
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt

# 生成数据字典
def create_dict(data_path, dict_path):
    dict_set = set()
    # 读取全部数据
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 把数据生成一个元组
    for line in lines:
        content = line.split('\t')[-1].replace('\n', '')
        for s in content:
            dict_set.add(s)
    # 把元组转换成字典，一个字对应一个数字
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    # 添加未知字符
    dict_txt = dict(dict_list)
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)
    # 把这些字典保存到本地中
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))
    print("数据字典生成完成！")
      
# 获取字典的长度
def get_dict_len(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])
    return len(line.keys())

# 创建序列化表示的数据,并按照一定比例划分训练数据与验证数据
def create_data_list(model_save_dir):
    #在生成数据之前，首先将eval_list.txt和train_list.txt清空
    with open(os.path.join(model_save_dir, 'eval_list.txt'), 'w', encoding='utf-8') as f_eval:
        f_eval.seek(0)
        f_eval.truncate()
    with open(os.path.join(model_save_dir, 'train_list.txt'), 'w', encoding='utf-8') as f_train:
        f_train.seek(0)
        f_train.truncate()
    with open(os.path.join(model_save_dir, 'dict.txt'), 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    with open(os.path.join(model_save_dir, 'all_data.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()
    i = 0
    with open(os.path.join(model_save_dir, 'eval_list.txt'), 'a', encoding='utf-8') as f_eval,open(os.path.join(model_save_dir, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
        for line in lines:
            words = line.split('\t')[-1].replace('\n', '')
            label = line.split('\t')[0]
            labs = ""
            if i % 8 == 0:
                for s in words:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + label + '\n'
                f_eval.write(labs)
            else:
                for s in words:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + label + '\n'
                f_train.write(labs)
            i += 1
    print("数据列表生成完成！")

dict_path = model_save_dir + "dict.txt"
with open(dict_path, 'w') as f:
    f.seek(0)
    f.truncate() 
create_dict(all_data_path, dict_path)
create_data_list(model_save_dir)

def data_mapper(sample):
    data, label = sample
    data = [int(data) for data in data.split(',')]
    return data, int(label)

#定义数据读取器
def data_reader(data_path):
    def reader():
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data, label = line.split('\t')
                yield data, label
    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)

# 获取训练数据读取器和测试数据读取器
BATCH_SIZE = 128
train_list_path = model_save_dir+'train_list.txt'
eval_list_path = model_save_dir+'eval_list.txt'
train_reader = paddle.batch(
		reader=data_reader(train_list_path), 
		batch_size=BATCH_SIZE)
eval_reader = paddle.batch(
		reader=data_reader(eval_list_path), 
		batch_size=BATCH_SIZE)

###############  搭建神经网络  ##################
def lstm_net(ipt, input_dim):
    # 以数据的IDs作为输入
    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)
    # 第一个全连接层
    fc1 = fluid.layers.fc(input=emb, size=128)
    # 进行一个长短期记忆操作
    lstm1, _ = fluid.layers.dynamic_lstm(input=fc1, #返回：隐藏状态（hidden state），LSTM的神经元状态
                                         size=128) #size=4*hidden_size
    # 第一个最大序列池操作
    fc2 = fluid.layers.sequence_pool(input=fc1, pool_type='max')
    # 第二个最大序列池操作
    lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type='max')
    # 以softmax作为全连接的输出层，大小为2,也就是正负面
    out = fluid.layers.fc(input=[fc2, lstm2], size=2, act='softmax')
    return out

# 定义输入数据， lod_level不为0指定输入数据为序列数据
words = fluid.data(name='words', shape=[None,1], dtype='int64', lod_level=1)
label = fluid.data(name='label', shape=[None,1], dtype='int64')
# 获取数据字典长度
dict_dim = get_dict_len(dict_path)
# 获取分类器
model = lstm_net(words, dict_dim)
# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)
# 获取预测程序
test_program = fluid.default_main_program().clone(for_test=True)
# 定义优化方法
optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.001)
opt = optimizer.minimize(avg_cost)

######################  训练与评估  #######################
use_cuda = False   # False表示运算场所为CPU, True表示运算场所为GPU 
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
# 定义数据映射器
feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

# 展示训练数据
all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]
all_eval_iter=0
all_eval_iters=[]
all_eval_costs=[]
all_eval_accs=[]
def draw_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    plt.show()

EPOCH_NUM = 20
for pass_id in range(EPOCH_NUM):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost, acc])
        all_train_iter=all_train_iter+BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Acc:%0.5f' % (pass_id, batch_id, train_cost[0], train_acc[0]))
    # 进行验证
    eval_costs = []
    eval_accs = []
    for batch_id, data in enumerate(eval_reader()):
        eval_cost, eval_acc = exe.run(program=test_program,
                                              feed=feeder.feed(data),
                                              fetch_list=[avg_cost, acc])
        eval_costs.append(eval_cost[0])
        eval_accs.append(eval_acc[0])
        all_eval_iter=all_eval_iter+BATCH_SIZE
        all_eval_iters.append(all_eval_iter)
        all_eval_costs.append(eval_cost[0])                                       
        all_eval_accs.append(eval_acc[0])      
    # 计算平均预测损失在和准确率
    eval_cost = (sum(eval_costs) / len(eval_costs))
    eval_acc = (sum(eval_accs) / len(eval_accs))
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, eval_cost, eval_acc))

# 保存模型
if not os.path.exists(model_save_dir): 
    os.makedirs(model_save_dir) 
fluid.io.save_inference_model(model_save_dir, 
                            feeded_var_names=[words.name], 
                            target_vars=[model], 
                            executor=exe)
print('训练模型保存完成！') 

draw_process("train",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc")
draw_process("eval",all_eval_iters,all_eval_costs,all_eval_accs,"evaling cost","evaling acc")