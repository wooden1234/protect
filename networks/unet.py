import numpy as np
import torch
import torch.nn as nn

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



class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=7, scale_factor=1):
        super(UNet, self).__init__()
        filters = np.array([64, 128, 256, 512, 1024])
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
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        # Stage 1
        x1 = self.Conv1(x)

        # Stage 2
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        # Stage 3
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        # Stage 4
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)

        # Stage 5
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)#5个卷积块，每个块之后一个最大池化

        # decoding + concat path
        # Stage 5d
        d5 = self.Up(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        # Stage 4d
        d4 = self.Up(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        # Stage 3d
        d3 = self.Up(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        # Stage 2d
        d2 = self.Up(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # Stage 1d
        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)#上采样，由sigmod激活函数输出概率分割图
        return d1

if __name__=='__main__':
    model = UNet(4,2)
    #print(model)

    #from thop import profile

    input  = torch.randn(1, 4, 512, 512)
    #flops, params = profile(model, inputs=(input,))
    #print("flops:{:.3f}G".format(flops / 1e9))#打印计算量
    #print("params:{:.3f}M".format(params / 1e6))#打印模型的参数数量
    out = model(input)
    print(out.shape)
