"""  根据想要的重复率，滑动窗口进行裁剪
裁剪后的影像带有空间参考信息，可以在ArcGIS中进行显示
重复率可以提高影像的数量，增加训练的机会
"""

import os
from  osgeo import gdal
import numpy as np


#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


"""  滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
"""


def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()

    filenname = TifPath.split("\\")[-1][:-4]

    # 左上角x坐标， 水平分辨率，旋转参数， 左上角y坐标，旋转参数，竖直分辨率
    # (591071.0674822949, 1.0, 0.0, 4283641.852856705, 0.0, -1.0)
    geotrans = dataset_img.GetGeoTransform()

    right_x = geotrans[0] + geotrans[1] * width  # 新横坐标起始量
    right_y = geotrans[3] + geotrans[5] * height  # 新纵坐标起始量

    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    # new_name = len(os.listdir(SavePath)) + 1
    #
    new_name = 1

    geotrans_1 = geotrans
    #  裁剪图片,重复率为RepetitionRate
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            #  如果图像是单波段
            if (len(img.shape) == 2):
                cropped = img[
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  如果图像是多波段
            else:
                cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]

            #  写图像
            writeTiff(cropped, geotrans_1, proj, SavePath + "/%s_%d.tif" % (filenname, new_name))

            #  文件名 + 1
            new_name = new_name + 1

            # 新投影
            new_x_geo = geotrans_1[0] + geotrans_1[1] * pixel * RepetitionRate  # 新横坐标起始量
            new_y_geo = geotrans_1[3] + geotrans_1[5] * 0  # 新纵坐标起始量
            geotrans_1 = (new_x_geo, geotrans_1[1], geotrans_1[2], new_y_geo, geotrans_1[4], geotrans_1[5])

        new_x_geo = geotrans[0] + geotrans[1] * 0  # 新横坐标起始量
        new_y_geo = geotrans_1[3] + geotrans_1[5] * pixel * RepetitionRate  # 新纵坐标起始量
        geotrans_1 = (new_x_geo, geotrans_1[1], geotrans_1[2], new_y_geo, geotrans_1[4], geotrans_1[5])

    h_x_geo = right_x - geotrans[1] * pixel  # 新横坐标起始量
    h_y_geo = geotrans[3]
    geotrans_2 = (h_x_geo, geotrans[1], geotrans[2], h_y_geo, geotrans[4], geotrans[5])
    #  向前裁剪最后一列
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        else:
            cropped = img[:,
                      int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        #  写图像
        writeTiff(cropped, geotrans_2, proj, SavePath + "/%s_%d.tif" % (filenname, new_name))
        new_name = new_name + 1

        # 新投影
        new_x_geo = geotrans_2[0] + geotrans_2[1] * 0  # 新横坐标起始量
        new_y_geo = geotrans_2[3] + geotrans_2[5] * pixel * RepetitionRate  # 新纵坐标起始量
        geotrans_2 = (new_x_geo, geotrans[1], geotrans[2], new_y_geo, geotrans[4], geotrans[5])

    q_x_geo = geotrans[0]
    q_y_geo = right_y - geotrans[5] * pixel  # 新纵坐标起始量
    geotrans_3 = (q_x_geo, geotrans[1], geotrans[2], q_y_geo, geotrans[4], geotrans[5])
    #  向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 2):
            cropped = img[(height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:,
                      (height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        writeTiff(cropped, geotrans_3, proj, SavePath + "/%s_%d.tif" % (filenname, new_name))
        #  文件名 + 1
        new_name = new_name + 1

        new_x_geo = geotrans_3[0] + geotrans_3[1] * pixel * RepetitionRate  # 新横坐标起始量
        new_y_geo = geotrans_3[3] + geotrans_3[5] * 0  # 新纵坐标起始量
        geotrans_3 = (new_x_geo, geotrans[1], geotrans[2], new_y_geo, geotrans[4], geotrans[5])

    y_x_geo = right_x - geotrans[1] * pixel  # 新横坐标起始量
    y_y_geo = right_y - geotrans[5] * pixel  # 新纵坐标起始量
    geotrans_4 = (y_x_geo, geotrans[1], geotrans[2], y_y_geo, geotrans[4], geotrans[5])
    #  裁剪右下角
    if (len(img.shape) == 2):
        cropped = img[(height - CropSize): height,
                  (width - CropSize): width]
    else:
        cropped = img[:, (height - CropSize): height, (width - CropSize): width]
    writeTiff(cropped, geotrans_4, proj, SavePath + "/%s_%d.tif" % (filenname, new_name))
    new_name = new_name + 1


#  将影像1裁剪为重复率为0.1的256×256的数据集
'''
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
'''

pixel = 512

TifCrop(r"D:\python\unetNew\dataset-sample-self\ch4321gf\data_1\image\mihoutao.tif",
        r"D:\python\unetNew\dataset-sample-self\ch4321gf\data_1\images", pixel, 0.2)
# TifCrop(r"Datadata2labellabel.tif", r"datatrainlabel1", 256, 0.1)
