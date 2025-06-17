import os

import torch.nn as nn
#计算kappa矩阵
from sklearn.metrics import classification_report
import pandas as pd
import tifffile as tiff


def evaluate_kappa(masks, labels,save_path):
    #target_names = ['kiwi', 'corn', 'buiding', 'road', 'bare', 'river', 'forest', 'plant']
    #target_names = ['apple', 'corn', 'buiding', 'road', 'bare']
    #target_names = ['background', 'greenhouse', 'buildings', 'road', 'unkown']
    target_names = ['background', 'mihoutao']
    #target_names = ['background', 'greenhouse', 'buildings', 'road']
    print(classification_report(masks, labels, target_names=target_names,digits=3))
    report = classification_report(masks, labels, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(save_path, index=True)

#混合损失函数：参数mix_loss_r（0,1）
from util.losses import DiceLoss
class Loss(nn.Module):
    def __init__(self, n_classes, loss_r):
        super(Loss, self).__init__()
        self.loss_r = loss_r
        self.n_classes = n_classes
    def forward(self, pred_mask, true_masks):
        criterion = nn.CrossEntropyLoss()
        loss_dice = DiceLoss(self.n_classes)

        if  self.loss_r  == 1:
            loss = criterion(pred_mask, true_masks.squeeze(1))
        elif self.loss_r == 0:
            loss = loss_dice(pred_mask, true_masks.squeeze(1), softmax=True)
        else:
            loss_ce   = criterion(pred_mask, true_masks.squeeze(1))
            loss_dice = loss_dice(pred_mask, true_masks.squeeze(1), softmax=True)
            loss      = self.loss_r * loss_ce + (1 - self.loss_r) * loss_dice
        return loss

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

#读取图，并归一化
def read_image(filename, resize_height=None, resize_width=None, normalization=True):
    #pil_img = Image.open(filename)
    pil_img = tiff.imread(filename)
    #pil_img = pil_img.resize((resize_width, resize_height))
    #print("pil_img1.shape", pil_img.shape)
    img_nd = np.array(pil_img, dtype='float32')
    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)
    # HWC to CHW
    #img_trans = np.transpose(img_nd,(0,1,2))
    if normalization:
        img_nd = img_nd / 255
    return img_nd

# def read_image(filename, resize_height=None, resize_width=None, normalization=True):
#     img_nd = np.array(tiff.imread(filename), dtype='float32')
#
#     if img_nd.ndim == 2:
#         img_nd = np.expand_dims(img_nd, axis=-1)
#
#     if resize_height is not None and resize_width is not None:
#         img_nd = np.stack([
#             cv2.resize(img_nd[..., i], (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
#             for i in range(img_nd.shape[-1])
#         ], axis=-1)
#
#     if normalization:
#         if img_nd.max() > 1:
#             img_nd = img_nd / img_nd.max()
#
#     return img_nd


from util.config import *
def category2mask(label):
    mask = np.zeros(label.shape[:2] + (3, ), dtype='uint8')
    for category, mask_color in LABELMAP_RGB.items():
        locs = np.where(label == category)
        mask[locs] = mask_color
    return mask

def category2mask1(label):
    mask = np.zeros(label.shape[:2] + (3,), dtype='uint8')
    for category, mask_color in LABELMAP_RGB.items():
        locs = np.where(label == category)
        mask[locs] = mask_color
    return mask

def mask2category(img):
    ret = np.zeros(img.shape[:2] + (3,), dtype='uint8')
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    for color in colors:
        locs = np.where((img[:, :, 0] == color[0]) & (img[:, :, 1] == color[1]) & (img[:, :, 2] == color[2]))
        ret[locs[0], locs[1], :] = INV_LABELMAP[tuple(color)]
    return  ret

#确定通道数
def make_channel(path):
    if(path=='ch321geoeye'or path=='ch321gh'or path == 'ch432'or path == 'ch321'or path == 'class01'or path == 'class0123'or path == 'qb20140725'or path == 'geoeye20150329'):
        nInputChannels=4
    if(path=='ch4321gf'):
        nInputChannels=4
    return nInputChannels

#选择激活函数
def make_relu(my_ReLU):
    if my_ReLU=='ReLU':
        my_ReLU = nn.ReLU(inplace=True)
    elif my_ReLU == 'Hswish':
        my_ReLU = nn.Hardswish(inplace=True)
    elif my_ReLU == 'Swish':
        my_ReLU = nn.SiLU(inplace=True)
    else :
        print("输入模型错误：ReLU，Hswish，Swish")
    return my_ReLU

# %构建模型


from models.MobV4Unet import MobV4Unet
from models.CMTFMnet import CMTFMNet
from networks.unet_v1 import UNet_v1
from networks.unet import UNet
from networks.unet_se import UNet_SE
from networks.fcn import FCN8s,VGGNet
from networks.deeplabv3_plus import DeepLabv3_plus
from networks.segnet import SegNet
from networks.Enet import ENet
from networks.unetpp import UNet_2Plus
from networks.unetppp import UNet_3Plus
from networks.CMTFnet import CMTFNet
from networks.UNetFormer import UNetFormer
from networks.ABCnet import ABCNet
from networks.TransUnet import TransUNet
from networks.MAnet import MANet
from networks.DAnet import DANet
from networks.FAnet import FANet
from models.HorUnet import HorUnet
from models.hocsnet import HorsUnet
from models.CMTFhornet import CMTFHNet
from models.UnetCMTF import UnetCMTF
from models.CMTFmamba import CMTFmamba
from models.CMTFhcnet import CMTFHCNet
from models.CMTFover import CMTFover
from models.Wnet import Wnet
from models.Wmamba import Wmamba

def make_model(tpye_model,n_classes,nInputChannels):
    if tpye_model=='fcn':
        vgg16 = VGGNet(requires_grad=True)
        model = FCN8s(pretrained_net=vgg16, n_class=n_classes)
    elif tpye_model=='unet':
        model = UNet(nInputChannels,n_classes )
    elif tpye_model=='unet_v1':
        model = UNet_v1(nInputChannels,n_classes )
    elif tpye_model=='unet_se':
        model = UNet_SE(nInputChannels,n_classes )
    elif tpye_model == 'deeplabv3_plus':
        model = DeepLabv3_plus(nInputChannels, n_classes)
    elif tpye_model == 'hornet_unet':
        model = HorUnet(n_channels=nInputChannels, n_classes=n_classes)
    elif tpye_model == 'horcsunet':
        model = HorsUnet(n_channels=nInputChannels, n_classes=n_classes)
    elif tpye_model == 'segnet':
        model = SegNet(nInputChannels, n_classes)
    elif tpye_model == 'Enet':
        model = ENet(nInputChannels, n_classes,encoder_relu=False, decoder_relu=True)
    elif tpye_model == 'unetpp':
        model = UNet_2Plus(nInputChannels, n_classes,feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True)
    elif tpye_model == 'unetppp':
        model = UNet_3Plus(nInputChannels, n_classes,feature_scale=4, is_deconv=True, is_batchnorm=True)
    elif tpye_model == 'CMTFnet':
        model = CMTFNet( decode_channels=512,dropout=0.1, num_classes=n_classes)
    elif tpye_model == 'CMTFMnet':
        model = CMTFMNet( decode_channels=48,dropout=0.1, num_classes=n_classes)
    elif tpye_model == 'CMTFHnet':
        model = CMTFHNet( decode_channels=192,dropout=0.1, num_classes=n_classes)
    elif tpye_model == 'CMTFHCnet':
        model = CMTFHCNet( decode_channels=192,dropout=0.1, num_classes=n_classes)
    elif tpye_model == 'CMTFmamba':
        model = CMTFmamba( decode_channels=192,dropout=0.1, num_classes=n_classes)
    elif tpye_model == 'wnet':
        model = Wnet(nInputChannels, n_classes)
    elif tpye_model == 'wmamba':
        model = Wmamba(nInputChannels, n_classes)
    elif tpye_model == 'CMTFover':
        model = CMTFover( decode_channels=112,dropout=0.1, num_classes=n_classes)
    elif tpye_model == 'UnetCMTF':
        model = UnetCMTF( n_channels=nInputChannels,decode_channels=128,dropout=0.1, num_classes=n_classes)
    elif tpye_model == 'UNetFormer':
        model = UNetFormer( decode_channels=64,dropout=0.1,backbone_name='swsl_resnet18',pretrained=True,window_size=8,num_classes=n_classes)
    elif tpye_model == 'ABCnet':
        model = ABCNet( nInputChannels, n_classes)
    elif tpye_model == 'TransUnet':
        model = TransUNet(img_dim=512,
                          in_channels=nInputChannels,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=n_classes)
    elif tpye_model == 'MAnet':
        model = MANet( nInputChannels, n_classes)
    elif tpye_model == 'MobileNet':
        model = MobV4Unet(nInputChannels, n_classes)
    elif tpye_model == 'DAnet':
        model = DANet( n_classes, inter_channel=256)
    elif tpye_model == 'FAnet':
        model = FANet( n_classes,backbone='resnet18')
    else :
        print("输入模型错误：fcn，unet，unet_se，deeplabv3_plus，segnet,unet_v1")
    return model

#判断目录是否存在，并创建目录
def dir_exists(path):
    if not os.path.exists(path):
            os.makedirs(path)


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp

# %loss画图
import datetime
def loss_plot(train_loss,test_loss,save_path):
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    iters = range(len(train_loss))
    plt.figure()
    plt.plot(iters, train_loss, 'gs-', linewidth=2, label='train loss')
    plt.plot(iters, test_loss, 'ro--', linewidth=2, label='val loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(save_path, "epoch_loss_" + str(time_str) + ".png"))
    plt.cla()
    plt.close("all")

# %iou画图
def iou_plot(iou,save_path):
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    iters = range(len(iou))
    plt.figure()
    plt.plot(iters, iou, 'gs-', linewidth=2, label='iou')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('iou')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(save_path, "iou_" + str(time_str) + ".png"))
    plt.cla()
    plt.close("all")

# %ACC画图
def acc_plot(ACC,save_path):
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    iters = range(len(ACC))
    plt.figure()
    plt.plot(iters, ACC, 'gs-', linewidth=2, label='ACC')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(save_path, "ACC_" + str(time_str) + ".png"))
    plt.cla()
    plt.close("all")

# %计算iou
def iou(pred, target,n_classes=5 ):
  ious = []
  # Ignore IoU for background class
  #for cls in range(n_classes-1):
  for cls in range(n_classes):
    #pred_inds = pred == (cls+1)
    pred_inds = pred == (cls)
    #print('pred_inds',pred_inds)
    #target_inds = target == (cls+1)
    target_inds = target == (cls)
    #print('target_inds',target_inds)
    intersection = (pred_inds[target_inds]).sum().item()  # Cast to long to prevent overflows
    #print('intersection',intersection)
    union = pred_inds.sum().item() + target_inds.sum().item() - intersection
    #print('union',union)
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(intersection / max(union, 1))
  return ious

# %分割数据
from torch.utils.data import random_split
def dataset_random_split(dataset,val_percent):
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    return train, val

# %
from torchvision import transforms
def inference(model, imgpath, device,size=(224, 224)):
    ori_img = read_image(imgpath, resize_height=size[0], resize_width=size[1])
    T = transforms.ToTensor()
    img = T(ori_img)
    img = img.unsqueeze(0)
    img = img.to(device)
    pred = model(img).detach().cpu().numpy()
    lab = np.argmax(pred[0, :, :, :], axis=0)
    mask = category2mask(lab)
    return ori_img, mask
