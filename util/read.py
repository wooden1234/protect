import os
# 文件夹路径
folder_path = "/data/wangqiang/unetNew/dataset-sample-self/ch4321gf/test/image-chips/"
# 获取文件夹中所有 .tif 图片文件的名称
tif_files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
# 将文件名写入 txt 文件
txt_path = "/data/wangqiang/unetNew/dataset-sample-self/ch4321gf/test.txt"
with open(txt_path, 'w') as file:
    for tif_file in tif_files:
        file.write(tif_file + '\n')
print("文件名已写入", txt_path)