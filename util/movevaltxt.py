interval = 56
chunk_size = 14#从整体数据集中抽取俩份验证集

# 读取原始文件
with open("D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256\\train.txt", 'r') as f:
    lines = f.readlines()

# 打开第二个文件，用于存储每隔56行取7行的内容
with open("D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256\\val.txt", 'w') as f2:
    for i in range(0, len(lines), interval + chunk_size):
        chunk = lines[i+interval:i+interval+chunk_size]
        f2.writelines(chunk)


# 读取train.txt文件和val.txt文件
with open("D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256\\train.txt", 'r') as train_file:
    train_lines = train_file.readlines()

with open("D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256\\val.txt", 'r') as val_file:
    val_lines = val_file.readlines()

# 找出train中包含的val中的内容
lines_to_remove = set(val_lines)

# 从train.txt中删除val中包含的内容
with open("D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256\\train.txt", 'w') as train_file:
    for line in train_lines:
        if line not in lines_to_remove:
            train_file.write(line)
