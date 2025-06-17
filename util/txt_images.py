import os
import shutil

filepath = r"D:\python\unetNew\dataset-sample-self\ch4321gf\data_1\images"
# label_path = r"E:\TIF\test_8bit_32float\test\vis"
total_img = os.listdir(filepath)  # ['0700.tif', '0701.tif', '07011.tif', ...]
num = len(total_img)  # 列表的长度
filelist = []

for i in range(num):
    name = total_img[i] + '\n'
    filelist.append(name)

print(filelist)

with open(r'D:\python\unetNew\dataset-sample-self\ch4321gf\data_1\train.txt', 'r') as stream:
    lines = stream.readlines()
    print(lines)
    for line in lines:
        print(line)
        if line in filelist:
            line = line.replace("\n", "")
        #     print("没删除！")
        # else:
        #     line = line.replace("\n", "")
        # os.remove("E:/TIF/gf2_8bit_32float/2016152_8bit/tif/" + line + ".tif")
            old_name = filepath + "/" + line
            print(old_name)
            new_name = "D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1\\train\images/" + line
            print(new_name)
            shutil.copyfile(old_name, new_name)




