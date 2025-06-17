import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWiseConv2dImplicitGEMM(nn.Module):
    def __init__(self, dim, kernel, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel = kernel
        self.padding = kernel // 2  # 保持输入输出尺寸一致
        self.bias = bias

        # 深度可分离卷积的权重（形状: [in_channels, 1, kernel, kernel]）
        self.weight = nn.Parameter(torch.randn(dim, 1, kernel, kernel))
        if bias:
            self.bias_term = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('bias_term', None)

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight,
            bias=self.bias_term,
            stride=1,
            padding=self.padding,
            groups=self.dim  # 关键：groups=dim 实现深度可分离卷积
        )

class DepthWiseConv2dImplicitGEMM_1(nn.Module):
    def __init__(self, dim, kernel_size, bias=True):
        super().__init__()
        self.dim = dim
        self.bias = bias

        # 确保 kernel_size 是 (H, W)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise ValueError("kernel_size must be int or (H, W) tuple")

        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # 深度可分离卷积的权重 [dim, 1, kh, kw]
        self.weight = nn.Parameter(torch.randn(dim, 1, *kernel_size))
        if bias:
            self.bias_term = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('bias_term', None)

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight,
            bias=self.bias_term,
            stride=1,
            padding=self.padding,
            groups=self.dim  # 深度可分离卷积
        )