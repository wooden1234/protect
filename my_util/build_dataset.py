import os
import random

trainval_percent = 1  # 训练验证数据集的百分比
train_percent = 0.8  # 训练集的百分比
filepath = r"/dataset-sample-self/ch4321gf/data_1/images"
total_img = os.listdir(filepath)

num = len(total_img)  # 列表的长度
list = range(num)

tv = int(num * trainval_percent)  # 训练验证集的图片个数
tr = int(tv * train_percent)  # 训练集的图片个数	# sample(seq, n) 从序列seq中选择n个随机且独立的元素；

trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

# 创建文件trainval.txt,test.txt,train.txt,val.txt
ftrain = open(r'/dataset-sample-self/ch4321gf/data_1/train.txt', 'w')
fval = open(r'/dataset-sample-self/ch4321gf/data_1/val.txt', 'w')
ftrainval = open(r'/dataset-sample-self/ch4321gf/data_1/trainval.txt', 'w')

for i in list:
    name = total_img[i]
    ftrainval.write(name+'\n')
    if i in train:
        ftrain.write(name+'\n')
    else:
        fval.write(name+'\n')

ftrain.close()
fval.close()
ftrainval.close()
