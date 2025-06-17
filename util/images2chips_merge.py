import numpy as np
import cv2
import os
import glob
import tifffile as tiff
from PIL import Image
dir=os.path.abspath('.')
def merge_picture(filename, img_shape, merge_path, mergesave_path):
    """
    合并多张图片成一张大图。

    参数:
    filename: 包含所有小图片文件名的列表。
    img_shape: 最终合并后大图的形状（高度, 宽度）。
    merge_path: 小图片所在目录的路径。
    mergesave_path: 合并后大图保存的路径。

    返回值:
    无
    """
    # 构建第一张图片的完整路径
    img_path = merge_path + '/' + filename[0]
    print(img_path)
    # 读取第一张图片，获取其形状信息
    shape = cv2.imread(img_path, 1).shape  # 三通道的影像需把-1改成1
    # 计算合并后大图的列数和行数
    num_of_cols = int(img_shape[1] / 512) + 1  # 列数
    num_of_rows = int(img_shape[0] / 512) + 1  # 行数
    # 获取小图的宽度和高度
    cols = shape[1]  # 小图宽度，列
    rows = shape[0]  # 小图高度，行

    # 获取小图的通道数
    channels = shape[2]
    # 创建一个全黑的大图模板
    dst = np.zeros((img_shape[0], img_shape[1], channels), np.uint8)
    # 遍历所有小图片，将其合并到大图中
    for i in range(len(filename)):
        # 构建当前小图片的完整路径
        img_path = merge_path + '/' + filename[i]
        # 读取当前小图片
        img = cv2.imread(img_path, -1)
        # 提取列数和行数信息，用于确定小图片在大图中的位置
        cols_x = int(img_path.split("_")[-2])  # （列数）
        rows_y = int(img_path.split("_")[-3])
        # 提取小图片的区域
        roi = img[0:rows, 0:cols, :]
        # 将小图片复制到大图模板的相应位置
        dst[rows_y:rows_y + rows, cols_x:cols_x + cols, :] = roi
    # 裁剪大图模板，确保其大小与预期的合并后大图一致
    dst = dst[0:img_shape[0], 0:img_shape[1], :]
    # 保存合并后的图片
    cv2.imwrite(mergesave_path, dst)
    # 打印合并完成的消息和保存路径
    print(f"合并完成，保存路径：{mergesave_path}")



"""遍历文件夹下某格式图片"""

def read_file(filename):
    image_label_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.rstrip()
            image_label_list.append(content)
    return image_label_list

if __name__ == '__main__':

    """调用合并图片的代码"""
    model_name = 'unet_ch4321gf_0.6_100_sel.pth' #通过模型名称定位路径
    base_path = os.path.abspath('..') + '/dataset-sample-self/'+model_name.strip().split('_')[1]+'/data_1'
    print(base_path)
    #训练fcn，segnet，unet用上边这一条
    #base_path = os.path.abspath('..') + '/dataset-sample-self/' + model_name.strip().split('_')[2]
    #训练unet_se和unet_v1和deeplabv3_plus用上面这条
    merge_path = base_path + '//test//label-chips/'  # 要合并的小图片所在的文件夹
    #print(merge_path)
    mergesave_path = base_path + '//test//label/' + model_name   # 要合并的图片存放所在的文件夹
    # 读取测试集图像标签列表
    image_label_list = read_file(base_path + '/test.txt')
    # 读取训练日志文件中的图像场景列表
    image_scene_list1 = read_file(base_path + '/train_log.txt')
    # ['train_mihoutao/22398/11511/001540'] train_log.txt中的后缀为长宽
    # 11511
    # 22398
    # 1540
    # 定义合并保存目录
    mergesave_dir = base_path + '/test/label/' + model_name + '/'
    # 如果合并保存目录不存在，则创建该目录
    if not os.path.exists(mergesave_dir):
        os.makedirs(mergesave_dir)
    # for i in range(len(image_scene_list1)):
    #     tt=image_scene_list1[i]
    #     scene=tt.strip().split('/')[0]
    #     mergesave_path = mergesave_dir + scene+ '.tif'  # 要合并的图片存放所在的文件夹
    #     file_size=int(tt.strip().split('/')[3])
    #     if i==0:
    #         t=0
    #     else:
    #         t=int(image_scene_list1[i-1].strip().split('/')[3])
    #     filenames=image_label_list[t:t+file_size]
    #     filename=image_label_list[t+file_size-1]
    #     img_shape = (int(tt.strip().split('/')[2]), int(tt.strip().split('/')[1]))
    #     merge_picture(filenames, img_shape,merge_path,mergesave_path)
    start_index = 0

    for i in range(len(image_scene_list1)):
        tt = image_scene_list1[i]
        scene = tt.strip().split('/')[0]
        file_size = int(tt.strip().split('/')[3])

        # 当前场景的文件列表
        filenames = image_label_list[start_index:start_index + file_size]

        # 获取当前图像的尺寸
        img_shape = (int(tt.strip().split('/')[2]), int(tt.strip().split('/')[1]))
        print(img_shape[0])
        print(img_shape[1])
        print(file_size)
        # 拼接保存路径
        mergesave_path = mergesave_dir + scene + '.tif'

        # 合并
        merge_picture(filenames, img_shape, merge_path, mergesave_path)

        # 更新起始索引
        start_index += file_size
