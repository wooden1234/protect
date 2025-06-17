import numpy as np
import torch
import torch.nn as nn


##########################################################################
## SE注意力模块
class se_block(nn.Module):
    def __init__(self, channels):
        super(se_block, self).__init__()
        # 空间信息进行压缩
        ratio = 16
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 经过两次全连接层，学习不同通道的重要性
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 取出batch size和通道数

        # b,c,w,h->b,c,1,1->b,c 压缩与通道信息学习
        avg = self.avgpool(x).view(b, c)

        # b,c->b,c->b,c,1,1 激励操作
        y = self.fc(avg).view(b, c, 1, 1)
        return x * y.expand_as(x)


##########################################################################
## conv_block
class conv_block(nn.Module):
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

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
        )
    def forward(self, x):
        return self.up(x)

class UNet_SE(nn.Module):
    def __init__(self, n_channels=3, n_classes=7, scale_factor=1):
        super(UNet_SE, self).__init__()
        filters = np.array([64, 128, 256, 512, 1024])
        #filters = np.array([32, 64, 128, 256, 512])
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])

        self.Up       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 上采样
        self.Up_conv5 = conv_block(ch_in=filters[4]+filters[3], ch_out=filters[3])
        self.Up_conv4 = conv_block(ch_in=filters[3]+filters[2], ch_out=filters[2])
        self.Up_conv3 = conv_block(ch_in=filters[2]+filters[1], ch_out=filters[1])
        self.Up_conv2 = conv_block(ch_in=filters[1]+filters[0], ch_out=filters[0])
        self.Conv_1x1 = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.se1 = se_block(channels=64)
        self.se2 = se_block(channels=128)
        self.se3 = se_block(channels=256)
        self.se4 = se_block(channels=512)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x11 = self.se1(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x22 = self.se2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x33 = self.se3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x44 = self.se4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up(x5)
        d5 = torch.cat((x44, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up(d5)
        d4 = torch.cat((x33, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up(d4)
        d3 = torch.cat((x22, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up(d3)
        d2 = torch.cat((x11, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)
        return d1

if __name__ == '__main__':
    model = UNet_SE(3, 7)
    print(model)
    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)

    from thop import profile
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
