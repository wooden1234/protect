import torchvision
import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

__all__ = ["ResNet18", "ResNet50", "ResNet34"]

class ResNet18(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=pretrained)
        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4



class ResNet34(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True):
        super(ResNet34, self).__init__()
        pretrained = torchvision.models.resnet34(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4


class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)

        # Add a new convolutional layer to handle four channels input
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data[:, :3] = pretrained.conv1.weight.data
        self.conv1.weight.data[:, 3] = pretrained.conv1.weight.data.mean(dim=1)

        for module_name in [
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x):
        b0 = self.relu(self.bn1(self.conv1(x)))
        b = self.maxpool(b0)
        b1 = self.layer1(b)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4


# Create an instance of the modified ResNet50 model
resnet50_4ch = ResNet50(pretrained=True)


class resnext50_32x4d(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(resnext50_32x4d, self).__init__()
        pretrained = torchvision.models.resnext50_32x4d(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        if get_ha:
            return b1, b2, b3, b4, pool

        return pool


class resnet152(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(resnet152, self).__init__()
        pretrained = torchvision.models.resnet152(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        if get_ha:
            return b1, b2, b3, b4, pool

        return pool

if __name__ == "__main__":
    from thop import profile
    x = torch.autograd.Variable(torch.randn(1, 4, 512, 512))
    net = ResNet50()
    print(net)
    out = net(x)
    print(out[0].shape,out[1].shape,out[2].shape,out[3].shape)
    # flops, params = profile(net, (x,))
    # print('flops: ', flops, 'params: ', params)