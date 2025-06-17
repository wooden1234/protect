'''
from PIL import Image

# 打开需要裁剪的tif图像
im = Image.open("/home/zhuyuying/Project/tifp/train_321_111_label.tif")

# 获取图像的大小和分辨率
width, height = im.size
res = im.info['dpi']

# 定义裁剪范围
xmin, ymin = [222564.104000,3797968.694000]
xmax, ymax = [223965.104000,3799969.694000]


#xmin, ymin = [裁剪范围的左下角x坐标，裁剪范围的左下角y坐标]
#xmax, ymax = [裁剪范围的右上角x坐标，裁剪范围的右上角y坐标]
# 计算裁剪范围在图像中的像素坐标
x1 = int((xmin / 25.4) * res[0])
y1 = height - int((ymin / 25.4) * res[1])
x2 = int((xmax / 25.4) * res[0])
y2 = height - int((ymax / 25.4) * res[1])

# 裁剪图像
im_crop = im.crop((x1, y1, x2, y2))
# 保存输出图像
im_crop.save("/home/zhuyuying/Project/tifp/train_321_111_label_left1.tif")
'''
'''
import numpy as np
import tifffile as tiff
from PIL import Image
img = Image.open('/home/tuxiaoyu/project/codetd/dataset-sample-self/ch4321gf/images/test_321_11.tif')

# 获取图像宽度和高度
width, height = img.size

# 裁剪图像
cropped_img = img.crop((0, 0, 1542, 1030))  # 从左往右(左下x，左下y，右上x，右上y)
#左边三分之二0, 0, 1500, 2000
#右边三分之一1500,0,2000,2000
# 保存裁剪后的图像
cropped_img.save('/home/tuxiaoyu/project/codetd/dataset-sample-self/ch4321gf/images/test_321_111.tif')

'''
import cv2

# 读取.tif格式图片
image = cv2.imread('/home/tuxiaoyu/project/codetd/dataset-sample-self/ch4321gf/images/test_321_11.tif', cv2.IMREAD_UNCHANGED)

if image is not None:
    # 定义要裁剪的区域 (x, y, width, height)
    # 下面的示例是裁剪出图片的中间 100x100 区域
    # 你可以根据需要自定义裁剪的区域的左上角坐标和宽度、高度
    x = 0  # 左上角 x 坐标
    y = 0 # 左上角 y 坐标
    width = 1542  # 裁剪宽度
    height = 1030  # 裁剪高度

    # 裁剪图片
    cropped_image = image[y:y+height, x:x+width]

    # 保存裁剪后的图片
    cv2.imwrite('/home/tuxiaoyu/project/codetd/dataset-sample-self/ch4321gf/images/test_321_111.tif', cropped_image)
else:
    print("Error: Could not open the image.")