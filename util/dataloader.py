# %%
import os, sys

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
import cv2
import helper

class Rdata(Dataset):
    def __init__(self, filename, image_dir, num_classes, resize_height=512, resize_width=512):
        '''
        :param filename: 数据文件TXT：
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 图像尺寸
        :param resize_width  图像尺寸
        '''
        self.image_label_list = self.read_file(filename)
        # 随机打乱列表
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
        #print("i={},index={}".format(i, index))
        image_name = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, 'image-chips', image_name)
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=True)
        img = self.data_preproccess(img)
        # img = img.permute(0,1,2)
        label_path = os.path.join(self.image_dir, 'label-chips', image_name)
        label = self.load_data(label_path, self.resize_height, self.resize_width, normalization=False)
        # label = helper.to_categorical(label, self.num_classes)
        label = self.data_preproccess(label)
        return {'image': img, 'mask': label,'image_name':image_name}

    def __len__(self):
        return self.len

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip()
                image_label_list.append(content)
        return image_label_list

    def load_data(self, path, resize_height, resize_width, normalization):
        image = helper.read_image(path, resize_height, resize_width, normalization)
        return image

    def data_preproccess(self, data):
        data = self.toTensor(data)
        return data

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp
if __name__ == '__main__':
    path = 'ch4321gf'  #数据通道，文件
    dir = os.path.abspath('..')
    image_dir = dir+'/dataset-sample-self/' + path+'/data'
    train_filename = image_dir + "/train.txt"
    train_image_dir=image_dir+'/train'
    test_filename = image_dir + "/test.txt"
    test_image_dir = image_dir + '/test'
    tpye_model = "unet"  # 设置模型的类型（unet,fcn,deeplabv3）
    dir_checkpoint = dir+'/checkpoints/' +'/'+ tpye_model + '/'
    log_output = dir+'/dataset-sample-self/'+path+'/logs/'+tpye_model+'_output.txt'
    helper.dir_exists(dir_checkpoint)
    epochs = 100
    n_classes = 2
    batch_size = 4
    dataset = Rdata(filename=train_filename, image_dir=train_image_dir, num_classes=2)
    val_percent=0.3
    train_set,test_set =helper.dataset_random_split(dataset,val_percent)
    image_datasets = {'train': train_set,'test': test_set }
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'test': DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
    }
    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }
    for batch in dataloaders['test']:
        imgs = batch['image']
        true_mask = batch['mask']
        image_name=batch['image_name'][0]
        print(image_name)

