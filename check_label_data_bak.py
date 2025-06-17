import os
import glob
import cv2
import numpy as np
LABELMAP = {
    0 : (0, 139, 0),
    1 : (212,255, 127),
    2 : (255,  0, 0),
    3 : (0,  154,  238),
    4 : (205,  250, 255),
    5 : (238, 210, 238),
    6 : (0, 195,   0),
}
dir=os.path.abspath('.')
data_label_path=dir+'/dataset-sample-self/ch321/labels/'
label_list= glob.glob(os.path.join(data_label_path, r'*.tif'))
ret = np.zeros((7, 3,4), dtype='uint8')
t=0
for line in enumerate(label_list):
    flag=0
    img = cv2.imread(line[1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j][0]==255 and img[i,j][1]==255 :
                flag=1
                img[i, j]=LABELMAP[4]
            if img[i,j][1]==205 :
                flag=1
                img[i, j]=LABELMAP[6]
    line_len=len(line[1])
    if(flag==1):
        filename=line[1][:line_len-4] +"_tt.tif"
        cv2.imwrite(filename, img )
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    print(line[1])
    print(colors[:, :])
    print('-----'*10)





