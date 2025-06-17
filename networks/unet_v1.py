import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet_v1(nn.Module):

    def __init__(self, nInputChannels,n_classes ):
        super().__init__()

        self.dconv_down1 = double_conv(nInputChannels, 64)  # 输入输出通道
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 上采样

        self.dconv_up3 = double_conv(256 + 512, 256)  #
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, 1)
        self.n_classes = n_classes
        self.n_classes = nInputChannels

    def forward(self, x):  # x是image
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)  # 通道数堆叠

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


if __name__=='__main__':
    model = UNet_v1(3,7)
    print(model)
    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
