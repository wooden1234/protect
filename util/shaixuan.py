import cv2
import os
import numpy as np
# Class to color (BGR)
def contains_blue_pixel(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return (image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 255)#筛选掉没有（0，0，255）的标签

# Path to the directories containing images
directory_with_label_orth = "D:/python/unetNew/dataset-sample-self/ch4321gf/data_1_256/train/label-orth/"
directory_with_label_chips = "D:/python/unetNew/dataset-sample-self/ch4321gf/data_1_256/train/label-chips/"
directory_with_image_chips = "D:/python/unetNew/dataset-sample-self/ch4321gf/data_1_256/train/image-chips/"
train_txt_path = "D:/python/unetNew/dataset-sample-self/ch4321gf/data_1_256/train.txt"
# Get the list of image files in the label-orth directory
image_files = os.listdir(directory_with_label_orth)
# Read the train.txt file
with open(train_txt_path, 'r') as file:
    train_data = file.readlines()
# List to store the names of deleted images
deleted_images = []
# Count of deleted images
deleted_count = 0
# Iterate over each image file
for image_file in image_files:
    label_orth_path = os.path.join(directory_with_label_orth, image_file)
    label_chips_path = os.path.join(directory_with_label_chips, image_file.replace('label-orth', 'label-chips'))
    image_chips_path = os.path.join(directory_with_image_chips, image_file.replace('label-orth', 'image-chips'))

    if os.path.exists(label_chips_path) and os.path.exists(image_chips_path) and os.path.exists(label_orth_path):
        if not np.any(contains_blue_pixel(label_orth_path)):
            os.remove(label_chips_path)
            os.remove(image_chips_path)
            os.remove(label_orth_path)
            deleted_count += 1
            print(f"Deleted files: {label_chips_path}, {image_chips_path}, {label_orth_path}")
            deleted_images.append(image_file)
# Remove the deleted image names from train.txt
new_train_data = [line for line in train_data if not any(image_file in line for image_file in deleted_images)]

with open(train_txt_path, 'w') as file:
    file.writelines(new_train_data)
print("Deleted images:", deleted_images)
print("Total number of deleted images:\n", deleted_count)