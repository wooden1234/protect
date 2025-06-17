#裁剪
import cv2
import os
import numpy as np
from config import train_ids, val_ids, test_ids, LABELMAP, INV_LABELMAP
import tifffile as tiff
size = 512#裁剪图片大小
stride = 409#步长
def color2class(orthochip, img):#将标签图映射到类别,即将颜色标签值赋予类别信息
    ret = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    ret = np.dstack([ret, ret, ret])#ret初始化为全零矩阵，用于存储映射后的结果
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)#提取图像中的颜色类别

    seen_colors = set( [tuple(color) for color in colors] )#存储已经处理的颜色集合
    IGNORE_COLOR = LABELMAP[0]#用于定义需要忽略的标签
    #IGNORE_COLOR=None
    #if IGNORE_COLOR in seen_colors:
    #    return None, None
    # print('INV_LABELMA',LABELMAP)
    # print('INV_LABELMA',INV_LABELMAP)
    # print('color',colors)
    # print('seen_colors',seen_colors)
    for color in colors:
        #print(color)
        locs = np.where( (img[:, :, 0] == color[0]) & (img[:, :, 1] == color[1]) & (img[:, :, 2] == color[2]) )
        ret[ locs[0], locs[1], : ] = INV_LABELMAP[ tuple(color) ] #tuple(color)可以获取颜色对应的标签
    return orthochip, ret   #循环遍历不同的颜色，查找每种颜色在图像中的像素位置

def image2tile(prefix, scene, dataset, orthofile,channel,  labelfile, windowx=size, windowy=size, stridex=stride, stridey=stride):
#根据指定的裁剪尺寸和步长，对输入的正射和标签图像进行裁剪并保存为小块图片，同时调用 color2class 进行类别映射
#prefix：路径前缀
#scene：场景名称
#dataset：数据集名称
#orthofile：正射文件路径
#channel：通道信息
#labelfile：标签文件路径
#windowx、windowy：裁剪窗口的尺寸
#stridex、stridey：裁剪窗口的步长
    if(channel=='ch321gf'or channel=='ch321gh'or channel=='ch432'or channel== 'ch321'):#3通道正射文件读取方式
        ortho = cv2.imread(orthofile)   #OpenCV库中用于读取图像文件的函数
        img_type='.tif'
    if(channel=='ch4321gf'):#4通道正射文件读取方式
        ortho = tiff.imread(orthofile)  #使用 tifffile 库中的函数来读取 TIFF 图像
        #ortho = tiff.imread(orthofile).transpose([1, 2, 0]) #图像转置处理，将通道维度从第0维移动到最后，以符合CV2库的通道顺序
        img_type = '.tif'
    label = cv2.imread(labelfile)   #标签文件读取
    #label = tiff.imread(labelfile)
    print('ortho.shape[0]：', ortho.shape[0])
    print('ortho.shape[1]：', ortho.shape[1])
    print('ortho.shape[2]：', ortho.shape[2])
    print('label.shape[0]：', label.shape[0])
    print('label.shape[1]：', label.shape[1])
    print('label.shape[2]：', label.shape[2])    #打印正射文件和标签文件的行列信息和通道信息
    # 不使用样本中的高程-但有时候会有用
    #eleva = cv2.imread(elevafile, -1)
    assert(ortho.shape[0] == label.shape[0])    #检查原图和标签行数是否一致
    print("要切块图像的大小为：{0}*{1}.".format(label.shape[1], label.shape[0]))
    assert(ortho.shape[1] == label.shape[1])    #检查原图和标签列数是否一致
    shape = ortho.shape
    xsize = shape[1]
    ysize = shape[0]
    #if(scene.strip().split('_')[0]=='test'):    #切割test的时候重置stridex
        #stridex=windowy
    counter = 0
    flag_x=0
    flag_y=0
    for i in range(0, shape[1] ):   #遍历图像的列，x轴方向
        xi = i * stridex
        flag_y = 0
        if (flag_x == 1):
            break
        if (xi >= shape[1] - windowx):
            flag_x = 1
            xi = shape[1] - windowy
        for j in range(0, shape[0]):
            yi = j * stridex
            if (flag_y == 1):
                break
            if (yi >= shape[0] - windowy):
                yi = shape[0] - windowy
                flag_y = 1
            global orth
            global  lab
            orthochip = ortho[yi:yi+windowy, xi:xi+windowx, :]
            labelchip = label[yi:yi+windowy, xi:xi+windowx, :]
            lab=labelchip
            orth=orthochip
            orthochip, classchip = color2class(orthochip, labelchip)
            if classchip is None:
                continue
            filename=scene + '_' + str(yi) + '_' + str(xi) + '_' + str(counter).zfill(6) + img_type
            with open(f"{prefix}/{dataset}", mode='a') as fd:
                fd.write(filename+'\n')
            orthochip_filename = os.path.join(prefix,scene.strip().split('_')[0], 'image-chips',filename)
            labelchip_filename = os.path.join(prefix,scene.strip().split('_')[0],'label-chips', filename)
            labelorth_filename = os.path.join(prefix,scene.strip().split('_')[0], 'label-orth',filename)
            if (channel == 'ch4321gf'):
                tiff.imsave(orthochip_filename, orthochip)
            if (channel=='ch321gf'or channel=='ch321gh'or channel=='ch432'or channel=='ch321'):
                cv2.imwrite(orthochip_filename, orthochip)
            cv2.imwrite(labelchip_filename, classchip[:,:,0]) #标签图只保存一个通道
            cv2.imwrite(labelorth_filename, lab)
            counter += 1
    print("图片切块完成，共切:{}块,路径为:{}".format(counter,orthochip_filename))
    log=scene +'/'+str(xsize)+'/' +str(ysize)+'/' +str(counter).zfill(6)
    log_filename = os.path.join(prefix, scene.strip().split('_')[0] )
    log_filename=log_filename+'_log.txt'
    with open(log_filename, mode='a') as fd:
        fd.write(log + '\n')

def get_split(scene):#确定图片裁剪后的存储路径
    #if scene in train_ids:
        #return "train.txt"
    #if scene in val_ids:
        #return 'val.txt'
    if scene in test_ids:
        return 'test.txt'
#删除当前目录下的所有文件
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def run(prefix,channel,flag):
#实现了图片裁剪的整体流程，遍历指定目录下的图片，进行裁剪并保存。同时清空之前保存图片的目录
    open(prefix + '/'+flag+'.txt', mode='w').close()
    open(prefix + '/' + flag + '_log.txt', mode='w').close()
    if not os.path.exists( os.path.join(prefix,flag,'image-chips') ):
        os.mkdir(os.path.join(prefix,flag,'image-chips'))
    del_file(os.path.join(prefix,flag,'image-chips'))#删除当前目录下的所有文件
    if not os.path.exists( os.path.join(prefix,flag,'image-chips') ):
        os.mkdir(os.path.join(prefix,flag, 'label-chips'))
    del_file(os.path.join(prefix, flag, 'label-chips'))#删除当前目录下的所有文件
    if not os.path.exists( os.path.join(prefix,flag,'image-chips') ):
        os.mkdir(os.path.join(prefix,flag, 'label-orth'))
    del_file(os.path.join(prefix, flag, 'label-orth'))#删除当前目录下的所有文件
    lines = [ line for line in open(f'{prefix}/index.csv') ]
    num_images = len(lines)-1
    print(f"converting {num_images} images to chips - this may take a few minutes but only needs to be done once.")
    for lineno, line in enumerate(lines):
        line = line.strip().split(' ')
        scene = line[0]
        dataset = get_split(scene)
        print(scene)
        print(flag)
        if ( scene.strip().split('_')[0]==flag):
            orthofile = os.path.join(prefix, 'image', scene + '.tif')
            labelfile = os.path.join(prefix, 'label', scene + '_label.tif')
            print(f"要切块的{flag}样本路径为：{orthofile}" )
            if os.path.exists(orthofile):
                image2tile(prefix, scene, dataset, orthofile, channel, labelfile)
            continue

if __name__ == '__main__':
    #----------------------------------------------------#
    # 切块需将图片名称添加到：1 index.csv
    #                2,util/config/train_ids,test_ids
    #----------------------------------------------------#
    prefix = 'D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1'#需要裁剪的文件名
    print(prefix)
    channel=prefix.strip().split('\\')[4]
    print(channel)
    run(prefix,channel,'train') #train,test
    #run(prefix, channel, 'test')
    print("图片已切割完成,可运行main程序进行模型训练")