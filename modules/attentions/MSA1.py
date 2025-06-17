import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attentions.SCConv import ScConv

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq


        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)


class MFMSAttentionBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            scale_branches=4,  # 修改1：默认设为4分支
            frequency_branches=16,
            frequency_selection='top',
            block_repetition=1,
            min_channel=64,
            min_resolution=8,
            groups=32,
            sc_group_num = 4,
            sc_gate_threshold = 0.5,
            sc_alpha = 0.5
    ):
        super(MFMSAttentionBlock, self).__init__()
        self.scale_branches = scale_branches
        self.frequency_branches = frequency_branches
        self.block_repetition = block_repetition
        self.min_channel = min_channel
        self.min_resolution = min_resolution

        # ========== 多尺度分支扩展（支持四分支） ==========
        self.multi_scale_branches = nn.ModuleList([])
        for scale_idx in range(self.scale_branches):
            inter_channel = in_channels // 2 ** scale_idx
            inter_channel = max(inter_channel, self.min_channel)

            # 创建各分支的膨胀卷积层（dilation = 1 + scale_idx → 4分支对应dilation=4）
            self.multi_scale_branches.append(
                # nn.Sequential(
                #     nn.Conv2d(
                #         in_channels, in_channels, kernel_size=3, stride=1,
                #         padding=1 + scale_idx,  # padding适配膨胀率
                #         dilation=1 + scale_idx,  # dilation=1,2,3,4
                #         groups=groups,  # 分组卷积
                #         bias=False
                #     ),
                #     nn.BatchNorm2d(in_channels),
                #     nn.ReLU(inplace=True),
                nn.Sequential(
                    ScConv(
                        op_channel=in_channels,  # 输入与输出通道相同
                        group_num=sc_group_num,  # 分组数
                        gate_treshold=sc_gate_threshold,
                        alpha=sc_alpha
                    ),
                    nn.Conv2d(in_channels, inter_channel, kernel_size=1, bias=False),
                    nn.BatchNorm2d(inter_channel),
                    nn.ReLU(inplace=True)
                )
            )

        # 通道数→特征图尺寸映射表（据数据集定义）
        c2wh = {32: 112, 64: 56, 128: 28, 256: 14, 512: 7}

        # ========== 多频模块与注意力参数扩展 ==========
        self.multi_frequency_branches = nn.ModuleList([])
        self.multi_frequency_branches_conv1 = nn.ModuleList([])
        self.multi_frequency_branches_conv2 = nn.ModuleList([])

        # 可学习参数扩展：为4个分支分配独立的alpha和beta
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(self.scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(self.scale_branches)])

        for scale_idx in range(self.scale_branches):  # 共4个分支
            inter_channel = in_channels // 2 ** scale_idx
            inter_channel = max(inter_channel, self.min_channel)

            # 构建多频率通道注意力模块（若有）
            if self.frequency_branches > 0:
                self.multi_frequency_branches.append(
                    nn.Sequential(
                        MultiFrequencyChannelAttention(
                            inter_channel,
                            c2wh.get(in_channels, 7),
                            c2wh.get(in_channels, 7),
                            self.frequency_branches,
                            frequency_selection
                        )
                    )
                )

            # 空间注意力权重生成器（每个分支独立）
            self.multi_frequency_branches_conv1.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, 1, kernel_size=1, bias=False),
                    nn.Sigmoid()
                )
            )

            # 注意力特征增强（每个分支独立）
            self.multi_frequency_branches_conv2.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, in_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        feature_aggregation = 0

        # ========== 处理所有四分支 ==========
        for scale_idx in range(self.scale_branches):
            # 下采样条件判断（防止下采样过度）
            target_res = int(x.shape[2] // 2 ** scale_idx)
            if target_res >= self.min_resolution:
                feature = F.avg_pool2d(x, kernel_size=2 ** scale_idx, stride=2 ** scale_idx, padding=0)
            else:
                feature = x

            # 多尺度特征提取（含膨胀卷积）
            feature = self.multi_scale_branches[scale_idx](feature)

            # 多频通道注意力（若有）
            if self.frequency_branches > 0:
                feature = self.multi_frequency_branches[scale_idx](feature)

            # 空间注意力生成与融合
            spatial_attn = self.multi_frequency_branches_conv1[scale_idx](feature)
            feature = self.multi_frequency_branches_conv2[scale_idx](
                feature * (1 - spatial_attn) * self.alpha_list[scale_idx] +
                feature * spatial_attn * self.beta_list[scale_idx]
            )

            # 上采样（若被下采样过）
            if feature.shape[2:] != x.shape[2:]:
                feature = F.interpolate(feature, size=x.shape[2:], mode='bilinear', align_corners=False)

            feature_aggregation += feature

        # 特征聚合（平均）与残差连接
        feature_aggregation /= self.scale_branches
        feature_aggregation += x

        return feature_aggregation

if __name__=='__main__':


    net = MFMSAttentionBlock(256,4,16,'top',1)


# 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将网络移动到 GPU
    net.to(device)

# 定义一个输入张量
    input_tensor = torch.randn(2, 256, 512, 512).to(device)



# 网络处理输入
    output_tensor = net(input_tensor)

# 输出网络输出

    print(output_tensor.shape)
    # print(output_tensor[1].shape)