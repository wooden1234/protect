import random
import os
import shutil
def split_train_val_test(interval=56, chunk_size=14, train_file_path='/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/train.txt', val_file_path='/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/val.txt'):
    with open(train_file_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)

    with open(val_file_path, 'w') as f2:
        for i in range(0, len(lines), interval + chunk_size):
            chunk = lines[i+interval:i+interval+chunk_size]
            f2.writelines(chunk)

    with open(train_file_path, 'r') as train_file:
        train_lines = train_file.readlines()

    with open(val_file_path, 'r') as test_file:
        test_lines = test_file.readlines()

    lines_to_remove = set(test_lines)

    with open(train_file_path, 'w') as train_file:
        for line in train_lines:
            if line not in lines_to_remove:
                train_file.write(line)
def split_val_test(interval=7, chunk_size=7, val_file_path='/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/val.txt', test_file_path='/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/test.txt'):
    with open(val_file_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)

    with open(test_file_path, 'w') as f2:
        for i in range(0, len(lines), interval + chunk_size):
            chunk = lines[i+interval:i+interval+chunk_size]
            f2.writelines(chunk)

    with open(val_file_path, 'r') as train_file:
        val_lines = train_file.readlines()

    with open(test_file_path, 'r') as test_file:
        test_lines = test_file.readlines()

    lines_to_remove = set(test_lines)

    with open(val_file_path, 'w') as val_file:
        for line in val_lines:
            if line not in lines_to_remove:
                val_file.write(line)


def move_images_based_on_test(image_list_file='/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/test.txt',
                              source_folders=[
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/train/image-chips/',
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/train/label-chips/',
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/train/label-orth/'],
                              destination_folders=[
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/test/image-chips/',
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/test/label-chips/',
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/test/label-orth/']):
    # 读取txt文件中的图片名称
    with open(image_list_file, 'r') as file:
        image_names = file.read().splitlines()

    for source_folder, dest_folder in zip(source_folders, destination_folders):
        os.makedirs(dest_folder, exist_ok=True)

        count = 0

        # 遍历文件夹中的文件
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file in image_names:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(dest_folder, file)
                    shutil.move(src_path, dst_path)  # 移动文件到目标文件夹
                    count += 1
        print(f"Moved {count} images to {dest_folder}")


def move_images_based_on_val(image_list_file='/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/val.txt',
                              source_folders=[
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/train/image-chips/',
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/train/label-chips/',
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/train/label-orth/'],
                              destination_folders=[
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/val/image-chips/',
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/val/label-chips/',
                                  '/home/tuxiaoyu/project/code1/dataset-sample-self/ch4321gf/val/label-orth/']):
    with open(image_list_file, 'r') as file:
        image_names = file.read().splitlines()

    for source_folder, dest_folder in zip(source_folders, destination_folders):
        os.makedirs(dest_folder, exist_ok=True)

        count = 0

        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file in image_names:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(dest_folder, file)
                    shutil.move(src_path, dst_path)
                    count += 1
        print(f"Moved {count} images to {dest_folder}")
if __name__ == '__main__':
    split_train_val_test()
    split_val_test()
    move_images_based_on_val()
    move_images_based_on_test()

