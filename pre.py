import cv2
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import helper
import tifffile as tiff  # 更适合高光谱/多通道图像

def predict_and_calculate_ratio(net, tif_path, device, save_path, target_class=1):
    """
    对单张.tif图像进行预测，并计算目标类别的像素比例
    :param net: 已加载的模型
    :param tif_path: 输入.tif图像路径
    :param device: CUDA或CPU
    :param save_path: 预测标签保存路径
    :param target_class: 要统计的目标类别ID，默认是1（mihoutao）
    :return: 目标类别所占比例 (float)
    """
    net.eval()

    # 读取.tif图像
    image = tiff.imread(tif_path)  # shape: [H, W, C] or [C, H, W]
    if image.shape[0] <= 10:  # 假设是 [C, H, W]
        image = np.transpose(image, (1, 2, 0))

    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = net(image_tensor)
    pred_label = pred.argmax(dim=1).squeeze().cpu().numpy()  # shape: [H, W]

    # 保存预测标签（彩色图）
    pred_vis = helper.category2mask(pred_label)
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, os.path.basename(tif_path).replace('.tif', '_label.png'))
    cv2.imwrite(save_name, pred_vis)

    # 计算目标类别像素比例
    total_pixels = pred_label.size
    target_pixels = np.sum(pred_label == target_class)
    ratio = target_pixels / total_pixels

    print(f"Target class pixel count: {target_pixels}")
    print(f"Total pixel count: {total_pixels}")
    print(f"Target ratio: {ratio:.4f}")

    return ratio

if __name__ == '__main__':
    model_name = "unet_ch4321gf_0.6_100_self.pth"
    tpye_model = model_name.strip().split('_')[0]
    model_path = "D:\\python\\unetNew\\dataset-sample-self\\ch4321gf\\data_1\\logs\\unet\\checkpoints\\" + model_name

    n_classes = 2
    nInputChannels = helper.make_channel(model_name.strip().split('_')[1])
    model = helper.make_model(tpye_model, n_classes, nInputChannels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    tif_path = "D:\python\\unetNew\dataset-sample-self\ch4321gf\data_1\\train\image-chips\\train_mihoutao_0_1227_000084.tif"
    save_path = "dataset-sample-self/" + model_name + "/"

    ratio = predict_and_calculate_ratio(model, tif_path, device, save_path, target_class=1)
    print(f"预测中目标类别（mihoutao）占比: {ratio:.4f}")
