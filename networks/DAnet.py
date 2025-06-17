import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ResNet import *

class PositionAttentionModule(nn.Module):
    ''' self-attention '''

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(
            in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along spatial dimensions
        """

        N, C, H, W = x.shape
        query = self.query_conv(x).view(
            N, -1, H*W).permute(0, 2, 1)  # (N, H*W, C')
        key = self.key_conv(x).view(N, -1, H*W)  # (N, C', H*W)

        # caluculate correlation
        energy = torch.bmm(query, key)    # (N, H*W, H*W)
        # spatial normalize
        attention = self.softmax(energy)

        value = self.value_conv(x).view(N, -1, H*W)    # (N, C, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along a channel dimension
        """

        N, C, H, W = x.shape
        query = x.view(N, C, -1)    # (N, C, H*W)
        key = x.view(N, C, -1).permute(0, 2, 1)    # (N, H*W, C)

        # calculate correlation
        energy = torch.bmm(query, key)    # (N, C, C)
        energy = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        value = x.view(N, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out


class DANet(nn.Module):
    def __init__(self, n_classes, inter_channel,backbone = ResNet50):
        super().__init__()
        self.n_classes = n_classes
        #self.conv1 = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1)
        # set a base model
        '''if model == 'drn_d_22':
            print('Dilated ResNet D 22 wil be used as a base model')
            self.base = drn_d_22(pretrained=True)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])
        elif model == 'drn_d_38':
            print('Dilated ResNet D 38 wil be used as a base model')
            self.base = drn_d_38(pretrained=True)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])
        else:
            print('There is no option you choose as a base model.')
            print('Instead, Dilated ResNet D 22 wil be used as a base model')
            self.base = drn_d_22(pretrained=True)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])'''

        # convolution before attention modules
        self.conv2pam = nn.Sequential(
            nn.Conv2d( inter_channel,  inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d( inter_channel),
            nn.ReLU()
        )
        '''input_tensor = torch.randn(1, 4, 512, 512)
        # 通过网络模型
        output_tensor = self.conv2pam(input_tensor)
        print(output_tensor.shape)'''
        self.conv2cam = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU()
        )
        '''input_tensor = torch.randn(1, 3, 512, 512)
        # 通过网络模型
        output_tensor = self.conv2cam(input_tensor)
        print(output_tensor.shape)'''
        # attention modules
        self.pam = PositionAttentionModule(in_channels=inter_channel)
        self.cam = ChannelAttentionModule()

        # convolution after attention modules
        self.pam2conv = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU())
        self.cam2conv = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU())

        # output layers for each attention module and sum features.
        self.conv_pam_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, n_classes, 1)
        )
        self.conv_cam_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, n_classes, 1)
        )
        self.conv_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, n_classes, 1)
        )

        self.backbone = backbone()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #print("x ：", x.shape)
        #x = self.base(x)
        res1,res2,res3,res4=self.backbone(x)
        x=res1
        '''print("res1:",res1.shape)
        print("res2:", res2.shape)
        print("res3:", res3.shape)
        print("res4:", res4.shape)
        print("basex ：", x.shape)'''
        # outputs from attention modules
        pam_out = self.conv2pam(x)
        #print("pam_out ：", pam_out.shape)
        pam_out = self.pam(pam_out)
        #print("pam_out ：", pam_out.shape)
        pam_out = self.pam2conv(pam_out)
        #print("pam_out ：", pam_out.shape)

        cam_out = self.conv2cam(x)
       # print("cam_out ：", cam_out.shape)
        cam_out = self.cam(cam_out)
       # print("cam_out ：", cam_out.shape)
        cam_out = self.cam2conv(cam_out)
        #print("cam_out ：", cam_out.shape)

        # segmentation result
        outputs = []
        feats_sum = pam_out + cam_out
       # print("feats_sum ：", feats_sum.shape)
        # 使用双线性插值进行上采样
        upsampled_feats_sum = F.interpolate(feats_sum, size=(512, 512), mode='bilinear', align_corners=False)
       # print("fupsampled_feats_sum ：", upsampled_feats_sum.shape)
        outputs.append(self.conv_out(upsampled_feats_sum))

        outputs.append(self.conv_pam_out(pam_out))
        outputs.append(self.conv_cam_out(cam_out))
        '''print("outputs ：", outputs[0].shape)
        list_length = len(outputs)
        print("Length of the list:", list_length)'''
        return outputs[0]
if __name__=='__main__':
    net = DANet( 2,256)
# 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将网络移动到 GPU
    net.to(device)
#定义一个输入张量
    input_tensor = torch.randn(1,4, 512, 512).to(device)
    print("input_tensor ：", input_tensor.shape)
# 网络处理输入
    output_tensor = net(input_tensor)
# 输出网络输出
    print("GPU处理后的张量：",output_tensor.shape)
