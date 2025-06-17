interval = 7
chunk_size = 7#再从俩份验证集中抽取一份为测试集

# 读取原始文件val.txt
with open("D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256\\val.txt", 'r') as f:
    lines = f.readlines()

# 打开第二个文件test.txt，用于存储每隔56行取7行的内容
with open("D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256\\test.txt", 'w') as f2:
    for i in range(0, len(lines), interval + chunk_size):
        chunk = lines[i+interval:i+interval+chunk_size]
        f2.writelines(chunk)


# 读取val.txt文件和test.txt文件
with open("D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256\\val.txt", 'r') as val_file:
    val_lines = val_file.readlines()

with open("D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256\\test.txt", 'r') as test_file:
    test_lines = test_file.readlines()

# 找出val中包含的test中的内容
lines_to_remove = set(test_lines)

# 从val.txt中删除test中包含的内容
with open("D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1_256\\val.txt", 'w') as val_file:
    for line in val_lines:
        if line not in lines_to_remove:
            val_file.write(line)
