# %%
import os
import numpy as np
from numpy.ma.core import squeeze
from torch.utils.data import Dataset, DataLoader
import helper
from util.dataloader import Rdata
import torch
import torch.nn as nn
import time
import datetime
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
#from boundary_loss import BoundaryLoss
def log_string(str):
    print(str)

#模型评估
def eval_net(net, loader, device,criterion):
    #criterion是计算损失函数
    """Evaluation model"""
    # 将模型设置为评估模式——————————冻结参数
    net.eval()

    # 获取验证数据集中的批次数量——————————验证集/batch_size
    n_test = len(loader)

    # 初始化变量以累积损失和IoU
    tot = 0
    total_ious = []

    # 初始化列表以存储预测标签和真实标签
    labels, masks = [], []

    # 设置进度条以显示验证过程的进度
    #unit='batch' 中，batch 是一个字符串，表示进度条中每步的单位名称。
    with tqdm(total=n_test, desc='Validation round', unit='batch', leave=False) as inner:
        # 遍历验证数据集中的每个批次
        for batch in loader:
            # 从批次中提取图像和真实掩码
            imgs = batch['image']
            true_masks = batch['mask']

            # 将图像和掩码移动到指定设备并转换为适当的类型
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            # 调整真实掩码的形状并进行值替换
            # true_masks = true_masks.squeeze(-1)
            # true_masks[true_masks == 1] = 0
            # true_masks[true_masks == 255] = 1
            # 禁用梯度计算以节省内存
            with torch.no_grad():
                pred_mask = net(imgs)
            # true_masks = torch.argmax(true_masks, dim=1)
            # 计算损失并累积到总损失中
            loss = criterion(pred_mask, true_masks)
            tot += loss.item()

            # 将预测掩码和真实掩码从GPU移动到CPU，并转换为numpy数组
            pred_mask = pred_mask.detach().cpu().numpy()
            true_masks = true_masks.detach().cpu().numpy()

            # 对预测掩码进行后处理，获取每个像素的类别
            pred_mask = np.argmax(pred_mask[:, :, :, :], axis=1)
            true_masks = true_masks.squeeze(1)

            # 存储预测标签和真实标签
            labels.append(pred_mask)
            masks.append(true_masks)

            # 计算并存储每个批次的IoU
            total_ious.append(helper.iou(pred_mask, true_masks, n_classes=net.n_classes))
            inner.update(n_test / 10)

    # 将模型恢复为训练模式
    net.train()

    # 将所有批次的IoU转换为张量并转置
    print("列表的长度:", len(total_ious))
    total_ious = torch.Tensor(total_ious).transpose(0, 1)

    # 初始化一个张量以存储每个类别的平均IoU
    ious = torch.Tensor(net.n_classes)

    # 计算每个类别的平均IoU，忽略NaN值
    for i, class_iou in enumerate(total_ious):
        ious[i] = class_iou[class_iou == class_iou].mean()

    # 将所有批次的预测标签和真实标签展平成一维数组
    labels = np.array([sublabel for label in labels for sublabel in label])
    masks = np.array([submask for mask in masks for submask in mask])
    labels = labels.flatten()
    masks = masks.flatten()

    # 返回累积的损失、每个类别的平均IoU、展平的预测标签和真实标签
    return tot, ious, labels, masks

def train_net(net,
              device,
              epochs,
              criterion,
              dir_checkpoint,
              save_path,
              lr=0.0001,
              save_cp=True):
    global_step = 0

    log_string(f'''Starting training:
        Epochs:          {epochs}
        Learning rate:   {lr}
        Training size:   {dataset_sizes['train']}
        val size:        {dataset_sizes['val']}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        n_classes:       {n_classes}
        model type:      {tpye_model}
        loss r :         {criterion.loss_r}
    ''')
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    curr_time = datetime.datetime.now()
    time_str  = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    log_output=save_path+'/'+str(epochs)+'_'+str(criterion.loss_r)+'_'+str(time_str)+'_output.txt'
    open(log_output, mode='w').close()
    train_loss,val_losses,val_ious,val_acc=[],[],[],[]
    log_string("train_loss,val_losses,val_IOU,val_ACC,mihoutao_iou")
    best_acc = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        start_time = time.time()
        with tqdm(total=dataset_sizes['train'], desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in dataloaders['train']:
                imgs        = batch['image']
               # print('imgs.size()',imgs.size())

                true_masks  = batch['mask']
                imgs        = imgs.to(device=device, dtype=torch.float32)
                true_masks  = true_masks.to(device=device, dtype=torch.long)
                # true_masks = true_masks.permute(0, 3, 1, 2)
               # true_masks = true_masks.squeeze(-1)
                # print(true_masks)
                # true_masks[true_masks == 1] = 0
                # true_masks[true_masks == 255] = 1
                # true_masks = true_masks.permute(0, 3, 1, 2)
                # imgs = imgs.permute(0, 3, 1, 2)  # 将形状从 (batch_size, height, width, channels) 转换为 (batch_size, channels, height, width)
                pred_masks  = net(imgs)
                # pred_masks = torch.argmax(pred_masks, dim=1)
                # true_masks = torch.argmax(true_masks, dim=1)
                loss = criterion(pred_masks, true_masks)
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1
        #验证模型
        val_loss,ious,labels,masks = eval_net(net, dataloaders['val'], device,criterion)
        greenhouse_iou  = ious.numpy()[1]
        ious_mean =ious.mean().numpy()
        acc = accuracy_score(masks, labels)
        # print("test_size={}".format(test_size))
        n_train  = len(dataloaders['train'])  # the number of batch
        n_val    = len(dataloaders['val'])  # the number of batch
        print('\nEpoch:' + str(epoch + 1) + '/train_loss:' + str(epoch_loss/n_train)+'/val_loss:'+str(val_loss/n_val)+
              '/Miou:'+str(ious_mean)+'/Acc:'+str(acc)+'/mihoutao_iou:'+str(greenhouse_iou))
        with open(log_output, mode='a') as fd:
            fd.write('Epoch/' + str(epoch + 1) + '/' + str(epoch_loss/n_train) + '/'+str(val_loss/n_val)
                     +'/'+str(ious_mean)+'/'+str(acc)+'/'+str(greenhouse_iou)+'\n')
        train_loss.append(epoch_loss/n_train)
        val_losses.append(val_loss/n_val)
        val_ious.append(ious_mean)
        val_acc.append(acc)
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                log_string('Created checkpoint directory')
            except OSError:
                pass
            if(acc >best_acc):
                best_acc=acc
                if (epoch>epochs*0.2):
                    # 将数值保存到文件
                    with open('bestepoch.txt', 'a') as file:
                        file.write(tpye_model+'bestepoch'+str(epoch))
                    kappa_savepath = save_path + '/' + str(epochs) +'_'+str(criterion.loss_r)+ '_' + str(time_str) + '_Kappa_result.csv'  # 测试数据保存路径
                    if os.path.exists(kappa_savepath):
                        open(kappa_savepath, mode='w', encoding='utf-8').close()
                    helper.evaluate_kappa(masks, labels, kappa_savepath)
                    torch.save(net,
                               dir_checkpoint + tpye_model + '_' + path+'_'+str(criterion.loss_r) + '_' + str(
                                   epochs) + '_2_self.pth')
                    log_string(f'Checkpoint {epoch + 1} saved !')
    return train_loss,val_losses,val_ious,val_acc

if __name__ == '__main__':
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    path = 'ch4321gf'  #数据通道，文件
    dir  = os.path.abspath('.')
    image_dir = 'dataset-sample-self/' + path+'/data_256'
    train_filename  =  image_dir + "/train.txt"
    train_image_dir =  image_dir+'/train'
    val_image_dir = image_dir + '/val'
    val_filename    =  image_dir + "/val.txt"
    #----------------------------------------------------#
    # 设置模型的类型（fcn，unet，unet_se，deeplabv3_plus，segnet,unet_v1）
    # 模型的路径和标签
    #----------------------------------------------------#
    tpye_model     = "wmamba"
    log_path       = dir+'/dataset-sample-self/'+path+'/data_256/logs/'+tpye_model+'/'
    save_path      = log_path+'/logs/'
    helper.dir_exists(save_path)
    dir_checkpoint = log_path + '/checkpoints/'
    helper.dir_exists(dir_checkpoint)
    epochs     = 100
    batch_size = 8
    n_classes  = 2
    #----------------------------------------------------#
    #   读取取训练样本和验证样本
    #----------------------------------------------------#
    train_set = Rdata(filename=train_filename, image_dir=train_image_dir,  num_classes=n_classes,)
    val_set   = Rdata(filename=val_filename,   image_dir=val_image_dir,  num_classes=n_classes)
    image_datasets = {'train': train_set,'val': val_set }
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0,drop_last=True),
        'val':   DataLoader(val_set,   batch_size=1,          shuffle=False, num_workers=0)
    }
    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }

    #------------------------------------------------------#
    #   创建模型
    #------------------------------------------------------#
    device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nInputChannels  = helper.make_channel(path)      #样本数据的通道数
    model           = helper.make_model(tpye_model,n_classes,nInputChannels)
    model           = model.to(device)
    #------------------------------------------------------#
    #   设置模型混合损失函数参数loss_r[0,1],训练模型：
    #------------------------------------------------------#
    criterion=helper.Loss(n_classes,loss_r=0.6)
    #criterion = BoundaryLoss()
    print("模型：",tpye_model)
    print("训练开始+++++++++++++++++++++++")
    train_loss,val_losses,val_IOU,val_ACC=train_net(model, device,epochs,criterion,dir_checkpoint,save_path)
    print("train_set:",train_loss)
    print("test_set:",val_losses)
    print("val_IOU:",val_IOU)
    print("val_ACC:",val_ACC)
    helper.loss_plot(train_loss,val_losses,save_path)
    helper.iou_plot(val_IOU, save_path)
    helper.acc_plot(val_ACC, save_path)
    print(f"训练完成，模型保存路径为：{dir_checkpoint}，可在{save_path}查看训练精度")
    print("接下来可运行test.py程序对模型进行测试")

