import os
import shutil

# 读取txt文件中的图片名称
with open('D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256/test.txt', 'r') as file:
    image_names = file.read().splitlines()
    print(image_names)

# 指定文件夹1、2、3以及对应的目标文件夹4、5、6
source_folders = ['D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256/val/image-chips/', 'D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256/val/label-chips/', 'D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256/val/label-orth/']#指定的源数据集，从这里抽取val和test
destination_folders = ['D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256/test/image-chips/', 'D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256/test/label-chips/', 'D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256/test/label-orth/']

for source_folder, dest_folder in zip(source_folders, destination_folders):
    os.makedirs(dest_folder, exist_ok=True)  # 创建目标文件夹

    # 计数器
    count = 0

    # 遍历文件夹中的文件
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file in image_names:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dest_folder, file)
                shutil.move(src_path, dst_path)  # 移动文件到目标文件夹
                count += 1

    print(f"移动了 {count} 张图片到 {dest_folder}")