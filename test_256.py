# %%
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import helper
from util.dataloader import Rdata
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
#模型测试
def test(net, loader, device,labsave_path):
    """Evaluation model"""
    net.eval()
    n_test        = len(loader)  # the number of batch
    tot           = 0
    total_ious    = []
    labels, masks = [], []
    i=0
    with tqdm(total=n_test, desc='Test round', unit='batch', leave=False) as inner:
        for batch in loader:
            imgs       = batch['image']
            true_masks = batch['mask']
            image_name = batch['image_name']
            imgs       = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)


            with torch.no_grad():
                pred_mask  = net(imgs)


            pred_mask  = pred_mask.detach().cpu().numpy()
            true_masks = true_masks.detach().cpu().numpy()
            pred_mask  = np.argmax(pred_mask[:, :, :, :], axis=1)
            pred_lab = helper.category2mask(pred_mask.squeeze(0))
            cv2.imwrite(labsave_path + str(image_name[0]), pred_lab)
            true_masks = true_masks.squeeze(1)
            labels.append(pred_mask)
            masks.append(true_masks)
            total_ious.append(helper.iou(pred_mask, true_masks ,n_classes=net.n_classes))
            inner.update(imgs.shape[0])
    # Calculate average IoU
    total_ious = torch.Tensor(total_ious).transpose(0, 1)
    ious       = torch.Tensor(net.n_classes )
    for i, class_iou in enumerate(total_ious):
        ious[i] = class_iou[class_iou == class_iou].mean()  # Calculate mean, ignoring NaNs
    # 把真实标签和预测标签flatten成一维
    print("ious",ious)
    labels = np.array([sublabel for label in labels for sublabel in label])
    masks  = np.array([submask for mask in masks for submask in mask])
    labels = labels.flatten()
    masks  = masks.flatten()
    return ious.numpy(),labels,masks

if __name__ == '__main__':
    # ----------------------------------------------------#
    # 加载模型（fcn，unet，unet_se，deeplabv3_plus，segnet,unet_v1
    # ----------------------------------------------------#
    model_name = "wnet_ch4321gf_0.6_100_2_self.pth"
    tpye_model = model_name .strip().split('_')[0]
    print(tpye_model)
    # 训练fcn，segnet，unet用上边这一条
    #tpye_model = str(model_name.strip().split('_')[0] + '_' + model_name.strip().split('_')[1])
    # 训练unet_se和unet_v1和deeplabv3_plus用上面这条
    dir = os.path.abspath('.')
    print(dir)
    model_path= "/home/liuzhenyin/project/dataset-sample-self/ch4321gf/data_256/logs/wnet/checkpoints/wnet_ch4321gf_0.6_100_2_self.pth"
    print("加载模路径：")
    n_classes=2
    nInputChannels =helper.make_channel(model_name .strip().split('_')[1])
    # 训练fcn，segnet，unet用上边这一条
    #nInputChannels = helper.make_channel(model_name.strip().split('_')[2])
    # 训练unet_se和unet_v1和deeplabv3_plus用上面这条
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(model_path)
    model=torch.load(model_path,map_location="cpu")
    model.to(device)
    """加载测试样本"""
    dir = 'dataset-sample-self/' + model_name.split("_")[1]
    # 训练fcn，segnet，unet用上边这一条
    #dir = 'dataset-sample-self/' + model_name.split("_")[2]
    # 训练unet_se和unet_v1和deeplabv3_plus用上面这条
    test_image_dir = dir + '/data_256/test'
    test_filename = dir + "/data_256/test.txt"
    #labsave_path保存预测标签路径,savepath保存结果
    labsave_path = dir + '/data_256/test/label/'+model_name+'/'
    if not os.path.exists(labsave_path ):
        os.makedirs(labsave_path )
    savepath = dir + '/data_256/test/logs/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    #----------------------------------------------------#
    #   读测试样本
    #----------------------------------------------------#
    test_set    = Rdata(filename=test_filename,image_dir=test_image_dir, num_classes=n_classes)
    dataloaders = {
        'test':   DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    }
    # prediction
    """开始测试模型"""
    print("开始测试模型")
    ious, labels, masks = test(model, dataloaders['test'], device,labsave_path)
    acc = accuracy_score(masks, labels)
    ious_mean=ious.mean()
    helper.evaluate_kappa(masks, labels,savepath+model_name+'_Kappa_result.csv')
    with open(savepath+model_name+'_result', mode='a') as fd:
        fd.write(model_name + '/' + str(ious_mean) +'/' + str(acc) + '/'+str(ious)+'\n')
    print('\nMiou:' + str(ious_mean) + '/Acc:' + str(acc))
    #各类Iou
    #target_names = ['kiwi', 'corn', 'buiding', 'road', 'bare', 'river', 'forest', 'plant']
    #target_names = ['apple', 'corn', 'buiding', 'road', 'bare']
    #target_names=['background', 'greenhouse', 'buildings', 'road', 'unkown']
    target_names = ['background', 'mihoutao']
    #target_names=['background', 'greenhouse', 'buildings', 'road']
    print(target_names)
    print('Iou:' + str(ious) )
    print("\n测试样本的分类结果路径为：",labsave_path)
    print("接下来请运行util/chips_merge将分类结果小图片合并成大图片")
    print(f"在{savepath}路径下查看模型测试评价结果")



