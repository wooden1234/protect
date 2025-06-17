import os
import random
#构建数据集，将裁剪后的训练数据按照 8:2 的比例划分为训练集和验证集
def read_file( filename):
    image_label_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.rstrip()
            image_label_list.append(content)
    return image_label_list

if __name__ == '__main__':#构建数据集，将裁剪后的train按照8：2分为训练和验证
    prefix      = os.path.abspath('..')+'/dataset-sample-self/ch4321gf/data_5'
    flag        = 'train'
    filename = prefix + '/' + flag + '.txt'
    filenames=read_file(filename)
    random.shuffle(filenames)
    trainval_percent = 0.8   #训练验证集占整个数据集的比重（划分训练集和测试验证集）
    num = len(filenames)    #数据集数量
    train_num = int(num * trainval_percent)    #训练集数量
    ftrain = open(os.path.join(prefix, 'train1.txt'), 'w')
    fval = open(os.path.join(prefix, 'val.txt'), 'w')
    for i in range(num):
        if i<=train_num:
            ftrain.write(filenames[i]+ '\n')
        else:
            fval.write(filenames[i]+ '\n')
    ftrain.close()
    fval.close()

