import matplotlib.pyplot as plt
import numpy as np

def plot_img_array(img_array, ncol=3):
    """
    根据给定的图片数组和列数，将多张图片以数组的形式展示出来。

    参数:
    img_array: 图片数组，包含多张图片。
    ncol: 每行显示的图片数量，默认为3。
    """
    # 计算图像数组应显示的行数
    nrow = len(img_array) // ncol

    # 创建子图，子图数量由图像数组决定，所有子图共享X轴和Y轴，设置图幅大小
    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    # 遍历图像数组，为每个图像找到正确的位置进行展示
    for i in range(len(img_array)):
        # 获取当前图像在子图中的位置
        plots[i // ncol, i % ncol]
        # 在指定的位置上展示图像
        plots[i // ncol, i % ncol].imshow(img_array[i])

from functools import reduce

def plot_side_by_side(img_arrays):
    """
    将多组图片并排展示，每组图片在一行中显示。

    参数:
    img_arrays: 多组图片数组，每组图片数组包含多张图片。
    """
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))
    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))

# LABELS = ['greenhouse','buildings', 'road', 'unkown', 'background']
LABELS = ['bac','mihoutao']
# LABELS = ['apple','corn', 'home', 'road']
# 类别到颜色（BGR）的映射
LABELMAP = {
     0 : (0,  0, 0),
     1 : (0,  0, 255),
     #0 : (0, 5, 255),
     #1 : (255, 0, 124),
}
# 颜色（BGR）到类别的映射
INV_LABELMAP = {
    (0, 0,0 )   :0,
    (0, 0, 255)   :1,
    #(0, 5, 255)   :0,
    #(255, 0, 124) :1,
}
LABELMAP_RGB = { k: (v[0], v[1], v[2]) for k, v in LABELMAP.items() }

INV_LABELMAP_RGB = { v: k for k, v in LABELMAP_RGB.items() }

# 训练集、验证集和测试集的ID列表
train_ids = [
      "train",
      #"train_321_11",
      #"train_321_22",
      #"train_321_55",
]
val_ids = [
    "val_321",
]
test_ids = [
    "test_mihoutao"
    #"test_321_11",
    #"test_321_33",
    #"test_321_66",
]
