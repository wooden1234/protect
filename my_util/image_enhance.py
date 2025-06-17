from osgeo import gdal
import numpy as np
import os


def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
        return
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    im_blueBand = im_data[0, 0:im_height, 0:im_width]  # 获取蓝波段
    im_greenBand = im_data[1, 0:im_height, 0:im_width]  # 获取绿波段
    im_redBand = im_data[2, 0:im_height, 0:im_width]  # 获取红波段
    im_nirBand = im_data[3, 0:im_height, 0:im_width]  # 获取近红外波段
    return im_blueBand, im_greenBand, im_redBand, im_nirBand


def writeTiff(im_data, im_width, im_height, im_bands, path):
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
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr


def flip90_right(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr


def mirror_flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.flipud(new_arr)
    return new_arr


def mirror_flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    new_arr = np.flipud(new_arr)
    return new_arr


def mirror_flip90_right(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    new_arr = np.flipud(new_arr)
    return new_arr


filepath = r"D:\python\unetNew\dataset-sample-self\ch4321gf\data_1\images"
filename = os.listdir(filepath)

for file in filename:
    tifname = filepath + "/" + file
    newtifname = file.split(".")[0]
    print(newtifname)

    tiff1, tiff2, tiff3 ,tiff4= readTif(tifname)

    # 旋转90度 旋转90度后镜像翻转
    flip90_left1 = flip90_left(tiff1)
    flip90_left2 = flip90_left(tiff2)
    flip90_left3 = flip90_left(tiff3)
    flip90_left4 = flip90_left(tiff4)
    tiff90 = np.array((flip90_left1, flip90_left2, flip90_left3, flip90_left4))

    newtiffile90 = filepath + "/" + newtifname + "_90.tif"
    writeTiff(tiff90, 512, 512, 4, newtiffile90)
    print(newtiffile90)

    mirror_flip90_left1 = mirror_flip90_left(tiff1)
    mirror_flip90_left2 = mirror_flip90_left(tiff2)
    mirror_flip90_left3 = mirror_flip90_left(tiff3)
    mirror_flip90_left4 = mirror_flip90_left(tiff4)
    tiff901 = np.array((mirror_flip90_left1, mirror_flip90_left2, mirror_flip90_left3, mirror_flip90_left4))

    newtiffile901 = filepath + "/" + newtifname + "_901.tif"
    writeTiff(tiff901, 512, 512, 4, newtiffile901)
    print(newtiffile901)

    # 旋转180度 旋转180度后镜像翻转
    flip1801 = flip180(tiff1)
    flip1802 = flip180(tiff2)
    flip1803 = flip180(tiff3)
    flip1804 = flip180(tiff4)
    tiff180 = np.array((flip1801, flip1802, flip1803, flip1804))

    newtiffile180 = newtiffile901 = filepath + "/" + newtifname + "_180.tif"
    writeTiff(tiff180, 512, 512, 4, newtiffile180)

    mirror_flip1801 = mirror_flip180(tiff1)
    mirror_flip1802 = mirror_flip180(tiff2)
    mirror_flip1803 = mirror_flip180(tiff3)
    mirror_flip1804 = mirror_flip180(tiff4)
    tiff1801 = np.array((mirror_flip1801, mirror_flip1802, mirror_flip1803, mirror_flip1804))

    newtiffile1801 = filepath + "/" + newtifname + "_1801.tif"
    writeTiff(tiff1801, 512, 512, 4, newtiffile1801)

    # 旋转270度 旋转270度后镜像翻转
    flip90_right1 = flip90_right(tiff1)
    flip90_right2 = flip90_right(tiff2)
    flip90_right3 = flip90_right(tiff3)
    flip90_right4 = flip90_right(tiff4)
    tiff270 = np.array((flip90_right1, flip90_right2, flip90_right3, flip90_right4))

    newtiffile270 = filepath + "/" + newtifname + "_270.tif"
    writeTiff(tiff270, 512, 512, 4, newtiffile270)

    mirror_flip90_right1 = mirror_flip90_right(tiff1)
    mirror_flip90_right2 = mirror_flip90_right(tiff2)
    mirror_flip90_right3 = mirror_flip90_right(tiff3)
    mirror_flip90_right4 = mirror_flip90_right(tiff4)
    tiff2701 = np.array((mirror_flip90_right1, mirror_flip90_right2, mirror_flip90_right3, mirror_flip90_right4))

    newtiffile2701 = filepath + "/" + newtifname + "_2701.tif"
    writeTiff(tiff2701, 512, 512, 4, newtiffile2701)

    tiff01 = np.flipud(tiff1)
    tiff02 = np.flipud(tiff2)
    tiff03 = np.flipud(tiff3)
    tiff04 = np.flipud(tiff4)
    tiff001 = np.array((tiff01, tiff02, tiff03,tiff04))

    newtiffile001 = filepath + "/" + newtifname + "_1.tif"
    writeTiff(tiff001, 512, 512, 4, newtiffile001)
