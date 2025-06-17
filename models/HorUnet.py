import numpy as np
import torch
import torch.nn as nn
from modules.backbone import hornet


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

class up_conv_block_1(nn.Module):#卷积块
    def __init__(self, ch_in, ch_out):
        super(up_conv_block_1, self).__init__()
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in // 2, ch_out, kernel_size=1)
        )

    def forward(self, x):
        return self.segmentation_head(x)



class HorUnet(nn.Module):
    def __init__(self,n_channels=4, n_classes=7, scale_factor=1):
        super(HorUnet, self).__init__()
        filters = np.array([96, 192, 384, 768])
        self.n_channels = n_channels
        self.n_classes = n_classes
       # self.backbone = hornet.hornet_small_7x7(n_channels,n_classes,pretrained=False,in_22k=False)
        self.backbone = hornet.get_model('hornet_tiny_7x7', n_channels, n_classes)
        self.scale_factor = scale_factor

        self.Up       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 上采样
        self.Up_conv5 = conv_block(ch_in=filters[3]+filters[2], ch_out=filters[2])
        self.Up_conv4 = conv_block(ch_in=filters[2]+filters[1], ch_out=filters[1])
        self.Up_conv3 = conv_block(ch_in=filters[1]+filters[0], ch_out=filters[0])
        self.Conv_1x1 = up_conv_block_1(ch_in=filters[0], ch_out=self.n_classes)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        # Stage 1
        x,x1,x2,x3 = self.backbone(x)

        # decoding + concat path
        # Stage 5d
        d3 = self.Up(x)
        d3 = torch.cat((x1, d3), dim=1)
        d3 = self.Up_conv5(d3)

        # Stage 4d
        d2 = self.Up(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv4(d2)

        # Stage 3d
        d1 = self.Up(d2)
        d1 = torch.cat((x3, d1), dim=1)
        d1 = self.Up_conv3(d1)

        # Stage 2d
        d =self.Conv_1x1(d1)

        # Stage 1d
        d_out = self.sigmoid(d)#上采样，由sigmod激活函数输出概率分割图
        return d_out

if __name__=='__main__':
    model = HorUnet(4,2)
    #print(model)

    #from thop import profile

    input  = torch.randn(4, 4, 512, 512)
    #flops, params = profile(model, inputs=(input,))
    #print("flops:{:.3f}G".format(flops / 1e9))#打印计算量
    #print("params:{:.3f}M".format(params / 1e6))#打印模型的参数数量
    out = model(input)
    print(out.shape)
