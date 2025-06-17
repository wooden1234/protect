# -*- coding: utf-8 -*-
from PIL import Image
import os
import tifffile as tiff
import numpy as np

def enhance(img_path,save_path,log_filename,flag):
    #img = Image.open(img_path)
    '''
    LEFT_RIGHT  = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
    img_name1   = img_name.strip().split('.')[0]+ '_left' +'.tif'
    LEFT_RIGHT.save(save_path + img_name1)

    TOP_BOTTOM  = img.transpose(Image.FLIP_TOP_BOTTOM)  # 垂直翻转
    img_name2   = img_name.strip().split('.')[0]+ '_top' +'.tif'
    TOP_BOTTOM .save(save_path + img_name2)

    TOPLEFT_BOTTOM  = img.transpose(Image.FLIP_TOP_BOTTOM)  # 水平垂直翻转
    TOPLEFT_BOTTOM  = TOPLEFT_BOTTOM.transpose(Image.FLIP_LEFT_RIGHT)  # 水平垂直翻转
    img_name3       = img_name.strip().split('.')[0]+ '_topleft' +'.tif'
    TOPLEFT_BOTTOM.save(save_path + img_name3)

    rotated_90  = img.transpose(Image.ROTATE_90) # 旋转90度
    img_name4   = img_name.strip().split('.')[0]+ '_R90' +'.tif'
    rotated_90.save(save_path + img_name4)

    rotated_180 = img.transpose(Image.ROTATE_180)  # 旋转180度
    img_name5   = img_name.strip().split('.')[0] + '_R180' +'.tif'
    rotated_180.save(save_path + img_name5)


    rotated_270 = img.transpose(Image.ROTATE_270)   # 旋转270度
    img_name6 = img_name.strip().split('.')[0] + '_R270' + '.tif'
    rotated_270.save(save_path + img_name6)
    '''
    img = tiff.imread(img_path)
    print("img.shape", img.shape)


    LEFT_RIGHT = np.flip(img, axis=1)  # 水平翻转
    print("LEFT_RIGHT.shape", LEFT_RIGHT.shape)
    img_name1 = img_name.strip().split('.')[0] + '_left' + '.tif'
    tiff.imsave(save_path + img_name1, LEFT_RIGHT)

    '''TOP_BOTTOM = np.flip(img, axis=0)  # 垂直翻转
    img_name2 = img_name.strip().split('.')[0] + '_top' + '.tif'
    tiff.imsave(save_path + img_name2, TOP_BOTTOM)'''

    TOPLEFT_BOTTOM = np.flip(img, axis=1)  # 水平垂直翻转
    TOPLEFT_BOTTOM = np.flip(TOPLEFT_BOTTOM, axis=0)  # 水平垂直翻转
    img_name3 = img_name.strip().split('.')[0] + '_topleft' + '.tif'
    tiff.imsave(save_path + img_name3, TOPLEFT_BOTTOM)

    rotated_90 = np.rot90(img, k=1)  # 旋转90度
    img_name4 = img_name.strip().split('.')[0] + '_R90' + '.tif'
    tiff.imsave(save_path + img_name4, rotated_90)

    '''rotated_180 = np.rot90(img, k=2)  # 旋转180度
    img_name5 = img_name.strip().split('.')[0] + '_R180' + '.tif'
    # rotated_180.save(save_path + img_name5)
    tiff.imsave(save_path + img_name5, rotated_180)'''

    '''rotated_270 = np.rot90(img, k=3)  # 旋转270度
    img_name6 = img_name.strip().split('.')[0] + '_R270' + '.tif'
    tiff.imsave(save_path + img_name6, rotated_270)'''

    if(flag==True):
           with open(log_filename, mode='a') as fd:
               fd.write(  img_name1 + '\n'+img_name3 + '\n'
                         +img_name4 + '\n'
                          )

if __name__ == '__main__':
    prefix      = os.path.abspath('..')+'/dataset-sample-self/ch4321gf/data_1'
    print(os.path.abspath('..'))
    flag        = 'train'
    image_chips = os.path.join(prefix,flag,'image-chips/')
    label_chips = os.path.join(prefix,flag, 'label-chips/')
    label_orth  = os.path.join(prefix,flag, 'label-orth/')
    file_dir    = image_chips  # 原始图片路径
    log_filename= prefix + '/'+flag+'.txt'
    i=0
    open(log_filename, 'w').close()
    for img_name in os.listdir(file_dir):
        print(f'对{img_name}分别进行3种翻转和3种旋转')
        with open(log_filename, mode='a') as fd:
            fd.write(img_name + '\n')
        enhance(image_chips + img_name, image_chips,log_filename,True)
        enhance(label_chips + img_name, label_chips,log_filename,False)
        enhance(label_orth  + img_name, label_orth,log_filename, False)
        i+=1
    print(f"{i}张图片增强成{i*4}张图片")