import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import cv2

def to_categorical(y, num_classes=None, dtype='float32'):
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical

def read_image(filename, resize_height=None, resize_width=None, normalization=False):

    bgr_image = cv2.imread(filename)
    if bgr_image is None:
        print("Warning::{}", filename)
        return None
    if len(bgr_image.shape) == 2: 
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    rgb_image = cv2.resize(rgb_image, dsize=(resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image, dtype='float32')
    if normalization:
        rgb_image = rgb_image / 255.0

    return rgb_image

class Rdata(Dataset):
    def __init__(self, filename, image_dir, num_classes, resize_height=224, resize_width=224):
        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.image_label_list = self.read_file(filename)
        random.shuffle(self.image_label_list)
        self.num_classes = num_classes
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.resize_height = resize_height
        self.resize_width = resize_width

        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        self.toTensor = transforms.ToTensor()
 
        '''class torchvision.transforms.Normalize(mean, std)
        此转换类作用于torch. * Tensor,给定均值(R, G, B) 和标准差(R, G, B)，
        用公式channel = (channel - mean) / std进行规范化。
        '''
        # self.normalize=transforms.Normalize()
 
    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_name = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, 'image-chips',image_name)
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=True)
        img = self.data_preproccess(img)
        print(img.type())
        label_path = os.path.join(self.image_dir, 'label-chips',image_name)
        label = self.load_data(label_path, self.resize_height, self.resize_width, normalization=False)[:,:,0]
        label = to_categorical(label,  6)
        label = self.data_preproccess(label)
        return img, label
 
    def __len__(self):
        return self.len
 
    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip()   
                image_label_list.append(content)
        return image_label_list
 
    def load_data(self, path, resize_height, resize_width, normalization):

        image = read_image(path, resize_height, resize_width, normalization)
        return image
 
    def data_preproccess(self, data):
        data = self.toTensor(data)
        return data

train_filename="dataset-sample/train.txt"
# test_filename="../dataset/test.txt"
image_dir='dataset-sample/'


epoch_num=2  #总样本循环次数
batch_size=1  #训练时的一组数据的大小
train_data_nums=4
max_iterate=int((train_data_nums+batch_size-1)/batch_size*epoch_num) #总迭代次数

train_data = Rdata(filename=train_filename, image_dir=image_dir, num_classes=7)
# test_data = TorchDataset(filename=test_filename, image_dir=image_dir,repeat=1)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)

# [1]使用epoch方法迭代，TorchDataset的参数repeat=1
for epoch in range(epoch_num):
    for batch_image, batch_label in train_loader:
        #print(batch_image.shape,batch_label.shape)
        image=batch_image[0,:]
        image=image.numpy()#image=np.array(image)
        image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        #image_processing.cv_show_image("image",image)
        #print("batch_image.shape:{},batch_label:{}".format(batch_image.shape,batch_label.shape))

