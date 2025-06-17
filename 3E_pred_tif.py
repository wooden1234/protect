# coding=utf-8
from osgeo import gdal, ogr, osr

import numpy as np
import torch
from tqdm import tqdm

output_height = 512
output_width = 512
n_classes = 2

#model = torch.load(r"E:\model\3E_model_recall.pth", map_location='cuda:0')
# eval()框架会自动把BN和DropOut固定住，不会取平均，而且用训练好的值

device = 'cuda'

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

 # 读取tif数据集,并做归一化
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    proj = dataset.GetProjection()

    # 左上角x坐标， 水平分辨率，旋转参数， 左上角y坐标，旋转参数，竖直分辨率
    # (591071.0674822949, 1.0, 0.0, 4283641.852856705, 0.0, -1.0)
    geotrans = dataset.GetGeoTransform()
    """
    数据的归一化
    """
    band1 = dataset.GetRasterBand(1).ReadAsArray()
    band2 = dataset.GetRasterBand(2).ReadAsArray()
    band3 = dataset.GetRasterBand(3).ReadAsArray()
    band4 = dataset.GetRasterBand(4).ReadAsArray()
    dataset = np.array([band1, band2, band3, band4])
    dataset[np.isnan(dataset)] = 0
    dataset = dataset.astype(np.float32)

    return dataset, width, height, proj, geotrans


# def get_image(img_info):
#
#     tif = gdal.Open(img_info)
#     img = tif.ReadAsArray()
#     img = img.astype(np.float32)
#     # print("---------------------------------------------------------")
#     # print(img.shape)
#     img = img.transpose(1, 2, 0)  # 数组转置 3*224*224 => 224*224*3
#
#     return img


#  保存tif文件
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


#  预测tif文件,形成矩阵
def ArrayPred(Model_file, TifPath, CropSize, RepetitionRate,log_callback=None):
    model = torch.load(Model_file, map_location='cuda:0')

    model.eval()
    print("正在读取数据")
    img, width, height, proj, geotrans = readTif(TifPath)
    array = np.zeros((height, width))
    print("开始预测")

    #  裁剪图片,重复率为RepetitionRate
    for i in tqdm(range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))), position=0):
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

                x = np.array([cropped])
                x = torch.from_numpy(x).type(torch.cuda.FloatTensor)

                x = x.to(device)
                pr = model.forward(x)
                pr = torch.argmax(pr, dim=1).squeeze().cpu().numpy().round()
                if (j == 0 and i == 0):

                    array[int(i * CropSize * (1 - RepetitionRate)): int(
                        i * CropSize * (1 - RepetitionRate) + CropSize - CropSize * RepetitionRate / 2),
                    int(j * CropSize * (1 - RepetitionRate)): int(
                        j * CropSize * (1 - RepetitionRate) + CropSize - CropSize * RepetitionRate / 2)] = \
                        pr[0: int(CropSize * (1 - RepetitionRate / 2)),
                        0: int(CropSize * (1 - RepetitionRate / 2))]


                elif (j == 0):

                    array[int(i * CropSize * (1 - RepetitionRate) + CropSize * RepetitionRate / 2): int(
                        i * CropSize * (1 - RepetitionRate) + CropSize - CropSize * RepetitionRate / 2),
                    int(j * CropSize * (1 - RepetitionRate)): int(
                        j * CropSize * (1 - RepetitionRate) + CropSize - CropSize * RepetitionRate / 2)] = \
                        pr[int(CropSize * RepetitionRate / 2): int(CropSize * (1 - RepetitionRate / 2)),
                        0: int(CropSize * (1 - RepetitionRate / 2))]

                else:
                    array[int(i * CropSize * (1 - RepetitionRate) + CropSize * RepetitionRate / 2): int(
                        i * CropSize * (1 - RepetitionRate) + CropSize - CropSize * RepetitionRate / 2),
                    int(j * CropSize * (1 - RepetitionRate) + CropSize * RepetitionRate / 2): int(
                        j * CropSize * (1 - RepetitionRate) + CropSize - CropSize * RepetitionRate / 2)] = \
                        pr[int(CropSize * RepetitionRate / 2): int(CropSize - CropSize * RepetitionRate / 2),
                        int(CropSize * RepetitionRate / 2): int(CropSize - CropSize * RepetitionRate / 2)]

    #  裁剪最后一列
    for i in tqdm(range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))), position=0):
        if (len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        else:
            cropped = img[:,
                      int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
            x = np.array([cropped])
            x = torch.from_numpy(x).type(torch.cuda.FloatTensor)

            x = x.to(device)
            pr = model.forward(x)
            pr = torch.argmax(pr, dim=1).squeeze().cpu().numpy().round()

            if (i == 0):
                array[int(i * CropSize * (1 - RepetitionRate)): int(
                    i * CropSize * (1 - RepetitionRate) + CropSize - CropSize * RepetitionRate / 2),
                int(width - CropSize + CropSize * RepetitionRate / 2): width] = \
                    pr[0: int(CropSize - CropSize * RepetitionRate / 2),
                    int(CropSize * RepetitionRate / 2): int(CropSize)]
            else:
                array[int(i * CropSize * (1 - RepetitionRate) + CropSize * RepetitionRate / 2): int(
                    i * CropSize * (1 - RepetitionRate) + CropSize - CropSize * RepetitionRate / 2),
                int(width - CropSize + CropSize * RepetitionRate / 2): width] = \
                    pr[int(CropSize * RepetitionRate / 2): int(CropSize - CropSize * RepetitionRate / 2),
                    int(CropSize * RepetitionRate / 2): CropSize]

    #  裁剪最后一行
    for j in tqdm(range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))), position=0):
        if (len(img.shape) == 2):
            cropped = img[(height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:, (height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]

            x = np.array([cropped])
            x = torch.from_numpy(x).type(torch.cuda.FloatTensor)

            x = x.to(device)
            pr = model.forward(x)
            pr = torch.argmax(pr, dim=1).squeeze().cpu().numpy().round()

            if (j == 0):
                array[int(height - CropSize + CropSize * RepetitionRate / 2): height,
                int(j * CropSize * (1 - RepetitionRate)): int(
                    j * CropSize * (1 - RepetitionRate) + CropSize - CropSize * RepetitionRate / 2)] = \
                    pr[int(CropSize * RepetitionRate / 2): CropSize,
                    0:int(CropSize * (1 - RepetitionRate / 2))]

            else:
                array[int(height - CropSize + CropSize * RepetitionRate / 2): height,
                int(j * CropSize * (1 - RepetitionRate) + CropSize * RepetitionRate / 2): int(
                    j * CropSize * (1 - RepetitionRate) + CropSize - CropSize * RepetitionRate / 2)] = \
                    pr[int(CropSize * RepetitionRate / 2): CropSize,
                    int(CropSize * RepetitionRate / 2): int(CropSize * (1 - RepetitionRate / 2))]

            array[(height - CropSize): height,
            int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize] = pr

    #  裁剪右下角
    if (len(img.shape) == 2):
        cropped = img[(height - CropSize): height,
                  (width - CropSize): width]
    else:
        cropped = img[:, (height - CropSize): height, (width - CropSize): width]

        x = np.array([cropped])
        x = torch.from_numpy(x).type(torch.cuda.FloatTensor)

        x = x.to(device)
        pr = model.forward(x)
        pr = torch.argmax(pr, dim=1).squeeze().cpu().numpy().round()

        array[int(height - CropSize + CropSize * RepetitionRate / 2): height,
        int(width - CropSize + CropSize * RepetitionRate / 2): width] = \
            pr[int(CropSize * RepetitionRate / 2): CropSize,
            int(CropSize * RepetitionRate / 2): CropSize]

    return array, proj, geotrans

def readTif2(fileName):

    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")

    return dataset
def raster2shp(src, savepath):

    tgtLayer = savepath.split("\\")[-1].split(".")[0]
    band = src.GetRasterBand(1)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp = driver.CreateDataSource(savepath)
    # 拷贝空间索引
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = shp.CreateLayer(tgtLayer, srs=srs)

    # 创建dbf文件
    fd = ogr.FieldDefn("DN", ogr.OFTReal)
    layer.CreateField(fd)
    dst_field = 0
    # 从图片中自动提取特征
    gdal.Polygonize(band, band, layer, dst_field, [], None)

ModelFile = 'D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1\logs\MobileNet\checkpoints\MobileNet_ch4321gf_0.6_100_Medium_net_self.pth'
# FilePath = 'D:\python\\unetNew\preimage\image.tif'
FilePath = 'D:\python\\unetNew\ZH19\\test1_mihoutao.tif'
pathtif = 'D:\python\\unetNew\ZH19' + '-tif.tif'

array, pro, gt = ArrayPred(ModelFile, FilePath, 512, 0)
writeTiff(array, gt, pro, pathtif)
#
#
# path_img = r"G:\8.22\caijian"
# path_out = r"G:\8.22\pre"
#
# name = "\\1-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\2-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\3-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\4-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\5-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\6-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\7-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\8-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\9-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\10-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\11-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\12-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\13-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\14-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\15-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\16-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\17-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\18-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\19-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\20-dom"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
# path_img = r"F:\3e\8"
# path_out = r"F:\3e\8-pre"
#
# name = "\\result(1)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)

# path_img = r"F:\3e\院12\map"
# path_out = r"E:\3e\院12"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# path_img = r"F:\3e\院16\map"
# path_out = r"E:\3e\院16"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)

# path_img = r"F:\3e\院17\map"
# path_out = r"E:\3e\院17"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)

# path_img = r"F:\3e\院18\map"
# path_out = r"E:\3e\院18"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# path_img = r"F:\3e\院23\map"
# path_out = r"E:\3e\院23"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# path_img = r"F:\3e\院头19\map"
# path_out = r"E:\3e\院头19"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# path_img = r"F:\3e\院头20\map"
# path_out = r"E:\3e\院头20"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# path_img = r"F:\3e\院头20-1\map"
# path_out = r"E:\3e\院头20-1"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# path_img = r"F:\3e\院头21\院头21\map"
# path_out = r"E:\3e\院头21"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# path_img = r"F:\3e\院头22\map"
# path_out = r"E:\3e\院头22"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# path_img = r"F:\3e\院头24\map"
# path_out = r"E:\3e\院头24"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# path_img = r"F:\3e\院头25\院头25\map"
# path_out = r"E:\3e\院头25"
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)


# path_img = r"F:\3e\zh12caijian"
# path_out = r"F:\3e\zh12caijian-pre"

# name = "\\1-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\2-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\3-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\4-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\5-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\6-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\7-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\8-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\9-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\10-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\11-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\12-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\13-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
#
# name = "\\14-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\15-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)

# name = "\\16-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\17-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\18-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\19-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\20-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\21-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\22-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\23-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\24-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\25-result(1)zh12"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
# name = "\\result(2)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(3)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(4)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(5)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(6)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(7)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(8)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(9)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(10)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(11)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(12)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(13)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(14)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(15)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result(16)"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#
# name = "\\result8.3"
#
# FilePath = path_img + name + ".tif"
# array, pro, gt = ArrayPred(FilePath, 512, 0)
# pathtif = path_out + name + "-tif.tif"
# print(name + "预测完成，正在写入数据")
# writeTiff(array, gt, pro, pathtif)
#


'''FilePath = r"G:\3E无人机正射及酸枣分布\嘉应寺3E正射.tif"
array, pro, gt = ArrayPred(FilePath, 512, 0)
#
pathtif = r"G:\3E无人机正射及酸枣分布\嘉应寺3E正射_pre_tif.tif"
print("预测完成，正在写入数据")
tif = writeTiff(array, gt, pro, pathtif)'''


# # pathtif = ''
#
# tif = readTif2(pathtif)
# raster2shp(tif, pathtif)
# # pathshp =  + "/" + "filename" + ".shp"
# raster2shp(tif, savepath=r"H:\赞皇数据\贝林生态\赞皇02-M3E-防地60\pred-shp.shp")
# print("shp文件已输出")