from osgeo import gdal

# 打开多光谱影像文件
dataset = gdal.Open("D:\python\\unetNew\preimage\image.tif")

# 获取宽和高
width = dataset.RasterXSize  # 宽度（列数）
height = dataset.RasterYSize  # 高度（行数）
bands = dataset.RasterCount  # 波段数（多光谱通道数）

print(f"宽: {width}, 高: {height}, 波段数: {bands}")
