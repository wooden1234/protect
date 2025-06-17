import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientCrossAttention(nn.Module):

    def __init__(self, in_channels_x1, in_channels_x2, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels_x1 = in_channels_x1
        self.in_channels_x2 = in_channels_x2
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels_x2, key_channels, 1)
        self.queries = nn.Conv2d(in_channels_x1, key_channels, 1)
        self.values = nn.Conv2d(in_channels_x2, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels_x2, 1)

    def forward(self, x1, x2):
        n, c1, h1, w1 = x1.size()
        n, c2, h2, w2 = x2.size()
        keys = self.keys(x2).reshape((n, self.key_channels, h2 * w2))
        queries = self.queries(x1).reshape(n, self.key_channels, h1 * w1)
        values = self.values(x2).reshape((n, self.value_channels, h2 * w2))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                    context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h1, w1)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        # attention = reprojected_value + x1
        attention = reprojected_value

        return attention



if __name__ == "__main__":
    x1 = torch.randn(8, 48, 256, 256)
    x2 = torch.randn(8, 256, 64, 64)

    # 初始化 EfficientCrossAttention 模块
    in_channels_x1 = 48
    in_channels_x2 = 256
    key_channels = 64
    head_count = 4
    value_channels = 128
    cross_attention = EfficientCrossAttention(in_channels_x1, in_channels_x2, key_channels, head_count, value_channels)
    # 运行交叉注意力模块
    output = cross_attention(x1, x2)

    # 打印输出形状
    print("Output shape:", output.shape)