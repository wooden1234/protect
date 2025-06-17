import numpy as np
import torch
import torch.nn as nn
from modules.backbone import mobilemamba
import math
import torch.nn.functional as F
import os
from einops import rearrange
from modules.attentions.CBAM import CBAM
class conv_block(nn.Module):#卷积块
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
class up_conv(nn.Module):#上采样
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),

            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.up(x)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class MAF(nn.Module):
    def __init__(self, dim, fc_ratio, dilation=[3, 5, 7], dropout=0., num_classes=6):
        super(MAF, self).__init__()

        self.conv0 = nn.Conv2d(dim, dim//fc_ratio, 1)
        self.bn0 = nn.BatchNorm2d(dim//fc_ratio)

        self.conv1_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3], groups=dim//fc_ratio)
        self.bn1_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv1_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn1_2 = nn.BatchNorm2d(dim)

        self.conv2_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2], groups=dim//fc_ratio)
        self.bn2_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv2_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn2_2 = nn.BatchNorm2d(dim)

        self.conv3_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1], groups=dim//fc_ratio)
        self.bn3_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv3_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn3_2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()

        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.bn4 = nn.BatchNorm2d(dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//fc_ratio, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(dim//fc_ratio, dim, 1, 1),
            nn.Sigmoid()
        )

        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

        self.head = nn.Sequential(SeparableConvBNReLU(dim, dim, kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  Conv(576, num_classes, kernel_size=1))

    def forward(self, x):
        u = x.clone()

        attn1_0 = self.relu(self.bn0(self.conv0(x)))
        attn1_1 = self.relu(self.bn1_1(self.conv1_1(attn1_0)))
        attn1_1 = self.relu(self.bn1_2(self.conv1_2(attn1_1)))
        attn1_2 = self.relu(self.bn2_1(self.conv2_1(attn1_0)))
        attn1_2 = self.relu(self.bn2_2(self.conv2_2(attn1_2)))
        attn1_3 = self.relu(self.bn3_1(self.conv3_1(attn1_0)))
        attn1_3 = self.relu(self.bn3_2(self.conv3_2(attn1_3)))

        c_attn = self.avg_pool(x)
        c_attn = self.fc(c_attn)
        c_attn = u * c_attn

        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)
        s_attn = u * s_attn

        attn = attn1_1 + attn1_2 + attn1_3
        attn = self.relu(self.bn4(self.conv4(attn)))
        attn = u * attn

        out = self.head(attn + c_attn + s_attn)

        return out

class ASPP_module(nn.Module):  # ASpp模块的组成
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        x = self.relu(x)
        x =F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class MutilScal(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MutilScal, self).__init__()
        self.conv0_1 = nn.Conv2d(dim, dim//fc_ratio, 1)
        self.bn0_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv0_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3], groups=dim //fc_ratio)
        self.bn0_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv0_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn0_3 = nn.BatchNorm2d(dim)

        self.conv1_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2], groups=dim // fc_ratio)
        self.bn1_2 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv1_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn1_3 = nn.BatchNorm2d(dim)

        self.conv2_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1], groups=dim//fc_ratio)
        self.bn2_2 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv2_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn2_3 = nn.BatchNorm2d(dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()

        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()

        attn0_1 = self.relu(self.bn0_1(self.conv0_1(x)))
        attn0_2 = self.relu(self.bn0_2(self.conv0_2(attn0_1)))
        attn0_3 = self.relu(self.bn0_3(self.conv0_3(attn0_2)))

        attn1_2 = self.relu(self.bn1_2(self.conv1_2(attn0_1)))
        attn1_3 = self.relu(self.bn1_3(self.conv1_3(attn1_2)))

        attn2_2 = self.relu(self.bn2_2(self.conv2_2(attn0_1)))
        attn2_3 = self.relu(self.bn2_3(self.conv2_3(attn2_2)))

        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.relu(self.bn3(self.conv3(attn)))
        attn = attn * u

        pool = self.Avg(attn)

        return pool

class Mutilscal_MHSA(nn.Module):
    def __init__(self, dim, num_heads, atten_drop = 0., proj_drop = 0., dilation = [3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(Mutilscal_MHSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.MSC = MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim//fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim//fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = Conv(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)

        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads,
                      d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads,
                         d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads,
                         d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return attn + c_attn

class E_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=5, act_layer=nn.ReLU6, drop=0.):
        super(E_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.conv1 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=ksize,
                                groups=hidden_features)
        self.conv2 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3,
                                groups=hidden_features)
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.fc2(x1 + x2)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x


class Wmamba(nn.Module):
    def __init__(self, n_channels=3, n_classes=7, scale_factor=1,os=16):
        super(Wmamba, self).__init__()
        # filters = np.array([32,32,64,96,128])
        # filters = np.array([32,48,80,160,256])
        filters = np.array([192,384,448])
        # filters = np.array([24, 48, 96, 192, 512])
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.backbone = mobilemamba.MobileMamba_B4(num_classes=n_classes)

        # ASPP,挑选参数
        if os == 16:
            dilations = [1, 3, 6, 9]
        elif os == 8:
            dilations = [1, 6, 12, 18]
        else:
            raise NotImplementedError
        # 四个不同带洞卷积的设置，获取不同感受野
        self.aspp1 = ASPP_module(448, 192, dilation=dilations[0])
        self.aspp2 = ASPP_module(448, 192, dilation=dilations[1])
        self.aspp3 = ASPP_module(448, 192, dilation=dilations[2])
        self.aspp4 = ASPP_module(448, 192, dilation=dilations[3])
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(448, 192, 1, stride=1, bias=False),
                                             nn.GroupNorm(1,192),
                                             nn.ReLU())

        self.CBAM = CBAM(in_chans=192)
        self.MHSA =Mutilscal_MHSA(dim=384,num_heads=16)
        self.EFFN = E_FFN(in_features=384,out_features=192)
        self.CB =ConvBN(960,filters[-3],1)
        self.CBR =ConvBNReLU(filters[-3],filters[-3],1)
        self.MAF =MAF(dim=576,fc_ratio=4,dropout=0.1,num_classes=n_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        # encoding path
        # Stage 1
        x2,x3,x4 = self.backbone(x)
        # decoding + concat path
        # Stage 5d
        x5 = self.aspp1(x4)
        x6 = self.aspp2(x4)
        x7 = self.aspp3(x4)
        x8 = self.aspp4(x4)
        x9 = self.global_avg_pool(x4)
        x9 = F.interpolate(x9, size=x8.size()[2:], mode='bilinear', align_corners=True)
        # 把四个ASPP模块以及全局池化层拼接起来
        x = torch.cat((x5, x6, x7, x8, x9), dim=1)
        x = self.CB(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        x3 = self.MHSA(x3)
        x3 = self.EFFN(x3)
        x3 =F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)

        x2 = self.CBR(x2)
        x2 = self.CBAM(x2)
        x2 =F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x,x2,x3), dim=1)

        x =self.MAF(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

if __name__=='__main__':
    model = Wmamba(4,2)
    #print(model)

    #from thop import profile

    input  = torch.randn(1, 4, 512, 512)
    #flops, params = profile(model, inputs=(input,))
    #print("flops:{:.3f}G".format(flops / 1e9))#打印计算量
    #print("params:{:.3f}M".format(params / 1e6))#打印模型的参数数量
    out = model(input)

    print(out.shape)

