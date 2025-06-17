import os
import glob
import cv2
import numpy as np
LABELMAP = {
    0 : (0, 139, 0),          #kiwi
    1 : (212,255, 127),       #corn
    2 : (255,  0, 0),
    3 : (0,  154,  238),
    4 : (205,  250, 255),
    5 : (238, 210, 238),
    6 : (0, 195,   0),
}
# 获取当前工作目录的绝对路径
dir = os.path.abspath('.')
# 拼接数据标签路径
data_label_path = dir + '/dataset-sample-self/ch4321gf/labels/'
# 获取所有以 .tif 结尾的文件路径列表
label_list = glob.glob(os.path.join(data_label_path, r'*.tif'))
# 初始化一个三维数组，数据类型为 uint8
ret = np.zeros((7, 3, 4), dtype='uint8')
# 初始化计数器 t
t = 0

# 遍历标签列表
for line in enumerate(label_list):
    # 初始化标志变量 flag
    flag = 0
    # 读取图像
    img = cv2.imread(line[1])
    # 遍历图像的每个像素
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 检查像素值是否匹配特定条件
            if img[i, j][0] == 255 and img[i, j][1] == 255:
                # 设置标志变量 flag
                flag = 1
                # 可以在此处对图像进行修改
            if img[i, j][1] == 205:
                flag = 1
                # 可以在此处对图像进行修改
    # 获取文件名长度
    line_len = len(line[1])
    # 根据标志变量 flag 的值决定是否保存图像
    if flag == 1:
        filename = line[1][:line_len - 4] + "_tt.tif"
        cv2.imwrite(filename, img)
    # 获取图像中所有唯一的颜色
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    # 打印文件路径和图像颜色信息
    print(line[1])
    print(colors[:, :])
    print('-----' * 10)






