import torch
import torch.nn.functional as F
import torch.nn as nn

class SRU(nn.Module):
    """ 空间重构单元（支持膨胀卷积和动态分组）"""

    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_threshold: float = 0.5,
                 dilation: int = 1,  # 新增膨胀参数
                 dynamic_group: bool = False):
        super().__init__()
        self.dilation = dilation
        self.dynamic_group = dynamic_group

        # 分割成组并根据重要性重排
        self.group_num = group_num
        self.gate_threshold = gate_threshold

        # 组内动态卷积（支持膨胀）
        self.split_conv = nn.Conv2d(
            op_channel, op_channel,
            kernel_size=3,
            padding=dilation,  # 膨胀卷积适配
            dilation=dilation,
            groups=group_num if not dynamic_group else 1,
        )

    def forward(self, x):
        if self.dynamic_group:
            # 动态分组（示例：根据激活强度自动分组，实际需定制逻辑）
            b, c, h, w = x.shape
            avg_act = torch.mean(x, dim=(2, 3))
            group_assignment = (avg_act > self.gate_threshold).long().sum(dim=1) // 2
            group_num = max(min(group_assignment.item(), self.group_num), 1)
            self.split_conv.groups = group_num

        # 特征分割与重构逻辑（此处简写，具体需按SRU原论文实现）
        x_split = self.split_conv(x)
        x = x + x_split  # 残差连接
        return x


class CRU(nn.Module):
    """ 通道交互单元（支持动态分组与膨胀）"""

    def __init__(self,
                 op_channel: int,
                 alpha: float = 0.5,
                 squeeze_ratio: int = 2,
                 dilation: int = 1,  # 新增膨胀参数
                 group_size: int = 2):
        super().__init__()
        self.dilation = dilation
        self.up_conv = nn.Conv2d(
            int(op_channel * alpha), op_channel,
            kernel_size=3, padding=dilation,  # 膨胀支持
            dilation=dilation,
            groups=group_size  # 分组卷积
        )

    def forward(self, x):
        # 简写通道交互逻辑（参考原CRU实现）
        x_low, x_high = torch.chunk(x, 2, dim=1)
        x_high = self.up_conv(x_high)
        return torch.cat([x_low, x_high], dim=1)


class ScConv(nn.Module):
    """ 改进的ScConv：支持膨胀卷积与动态分组 """

    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_threshold: float = 0.5,
                 alpha: float = 0.5,
                 squeeze_ratio: int = 2,
                 dilation: int = 1,  # 扩展参数：膨胀率
                 dynamic_group: bool = False  # 开关：是否动态调整分组数
                 ):
        super().__init__()
        # 传递膨胀率和动态分组开关至 SRU/CRU
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_threshold=gate_threshold,
                       dilation=dilation,
                       dynamic_group=dynamic_group)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_ratio=squeeze_ratio,
                       dilation=dilation)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1, 32, 16, 16)
    model = ScConv(32)
    print(model(x).shape)