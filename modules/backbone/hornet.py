from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.fft


def get_dwconv(dim, kernel, bias):
    """
    创建一个深度卷积（Depthwise Convolution）层。

    :param dim: int, 输入和输出通道的数量，深度卷积中输入和输出维度相同。
    :param kernel: int, 卷积核的大小，是一个正方形卷积核的边长。
    :param bias: bool, 如果为True，将为卷积层添加偏置项。
    :return: nn.Conv2d, 返回一个深度卷积层实例。

    深度卷积是一种特殊的卷积方式，它对输入的每个通道单独进行卷积操作，而不是对所有通道进行综合。
    这种方法可以显著减少模型的计算量和参数数量，常用于轻量级神经网络架构中。
    """
    # 使用输入维度、输出维度、卷积核大小、填充大小、是否使用偏置以及分组参数来创建深度卷积层。
    # 这里分组参数等于输入维度，意味着每个输入通道都将被独立的卷积核处理。
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)



class GlobalLocalFilter(nn.Module):
    """
    GlobalLocalFilter类旨在处理输入的二维特征图，通过深度卷积和复数权重的傅里叶变换
    结合全局和局部信息。该类继承自nn.Module，是PyTorch中的一个可训练模块。

    参数:
    - dim: 特征图的通道数。
    - h: 特征图的高度，默认值为14。
    - w: 特征图的宽度，默认值为8。+
    """
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        # 深度卷积，用于处理特征图的局部信息
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        # 初始化复数权重，用于傅里叶变换以处理全局信息
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        # 前置和后置的LayerNorm，用于特征图的标准化
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        """
        前向传播函数，处理输入的特征图。

        参数:
        - x: 输入的特征图，形状为(B, C, H, W)。

        返回:
        - 经过处理后的特征图。
        """
        # 对输入特征图进行前置标准化
        x = self.pre_norm(x)
        # 将特征图分成两部分，分别处理
        x1, x2 = torch.chunk(x, 2, dim=1)
        #在第一个维度上进行通道分离，分别进行处理
        # 使用深度卷积处理x1的局部信息
        x1 = self.dw(x1)

        # 准备对x2进行傅里叶变换，处理全局信息
        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        # 根据x2的形状调整复数权重的形状
        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)

        # 将调整后的复数权重应用于傅里叶变换后的特征图
        weight = torch.view_as_complex(weight.contiguous())
        x2 = x2 * weight
        # 进行逆傅里叶变换，将全局信息与局部信息融合
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        # 将处理后的两部分特征图重新合并，并进行后置标准化
        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x



class gnconv(nn.Module):
    """
    实现了gnconv层，这是一种自定义的卷积层，用于神经网络中。

    参数:
    - dim: 输入维度
    - order: 卷积的阶数，默认为5
    - gflayer: 可选的全局特征层
    - h: 输入特征图的高度
    - w: 输入特征图的宽度
    - s: 尺度因子，默认为1.0
    """
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        # 根据阶数计算每个子空间的维度，并反转列表以便于后续操作
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        # 输入投影层，将输入维度映射到2倍的维度
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        # 根据是否提供gflayer参数，选择合适的深度卷积层
        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        # 输出投影层，将维度映射回原始维度
        self.proj_out = nn.Conv2d(dim, dim, 1)

        # 点卷积层列表，用于跨通道信息交流
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        # 打印gnconv层的信息
        #print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        """
        前向传播函数

        参数:
        - x: 输入特征图
        - mask: 可选的掩码
        - dummy: 哑变量，未使用

        返回:
        - x: 输出特征图
        """
        B, C, H, W = x.shape

        # 输入投影，分裂成pwa和abc两部分
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        # 应用深度卷积并缩放
        dw_abc = self.dwconv(abc) * self.scale

        # 分割dw_abc到不同的维度，计算最终输出
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        # 逐层计算，融合不同维度的信息
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        # 输出投影，得到最终特征图
        x = self.proj_out(x)

        return x



class Block(nn.Module):
    r""" HorNet block
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        #根据条件创建一个 可学习的参数 gamma1，并在构造时为它初始化一个特定的值。
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #DropPath正则化方法

    def forward(self, x):
        #self.norm1(x)：规范化输入（如 LayerNorm、BatchNorm）

        #self.gnconv(...)：某个空间操作（如卷积）

        #gamma1：通道缩放，控制每个通道对结果的贡献

        #drop_path：训练时随机丢掉整段分支，增强泛化能力

        #x + ...：残差结构（残差连接）
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
            #view类似于reshape
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        #残差模块
        return x


class HorNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], base_dim=96, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 gnconv=gnconv, block=Block, uniform_init=False, **kwargs):
        super().__init__()
        self.n_classes = num_classes
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

        # Encoder部分保持不变
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")#归一化
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 判断gnconv是否是列表
        if not isinstance(gnconv, list):
            gnconv = [gnconv, gnconv, gnconv, gnconv]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # 解码器部分
        self.decoder = nn.ModuleList()
        for i in reversed(range(3)):
            self.decoder.append(nn.Sequential(
                nn.Conv2d(dims[i + 1], dims[i], kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(dims[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(dims[i], dims[i], kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(dims[i]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ))

        # 最后的1x1卷积用于输出分割结果
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(dims[0], dims[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(dims[0] // 2, num_classes, kernel_size=1)
        )

        self.uniform_init = uniform_init
        self.apply(self._init_weights)
        #递归地对模型的所有子模块（包括自身）调用函数 func。

    # 截断初始化
    def _init_weights(self, m):
        if not self.uniform_init:
            #isinstance()判断某个对象是否是一个特定的类或子类。
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                #初始化权重
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                #hasattr() 是 Python 的一个内置函数，用于判断一个对象是否具有某个属性
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码器部分
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            for j, blk in enumerate(self.stages[i]):
                x = blk(x)
            if i < 3:  # 保存前三个阶段的特征用于跳跃连接
                features.append(x)

        print(len(features))
        print(x.shape)
        x3 = features[0]
        x2 = features[1]
        x1 = features[2]
        # print(x.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)



        # 解码器部分
        # for i, layer in enumerate(self.decoder):
        #     x = layer(x)
        #     print(f"{i}个解码{x.shape}")
        #     if i < len(features):
        #         print(f"{i}个特征{features[i].shape}")
        #         # 添加跳跃连接
        #         x = x + features[-(i + 1)]
        #
        # # 分割头
        # x = self.segmentation_head(x)
        return x,x1,x2,x3



# class HorNet(nn.Module):
#     def __init__(self, in_chans=3, num_classes=1000,
#                  depths=[3, 3, 9, 3], base_dim=96, drop_path_rate=0.,
#                  layer_scale_init_value=1e-6, head_init_scale=1.,
#                  gnconv=gnconv, block=Block, uniform_init=False, **kwargs
#                  ):
#         super().__init__()
#         dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
#
#         self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
#         stem = nn.Sequential(
#             nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#             LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
#         )
#         self.downsample_layers.append(stem)
#         for i in range(3):
#             downsample_layer = nn.Sequential(
#                 LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
#                 nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
#             )
#             self.downsample_layers.append(downsample_layer)
#
#         self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
#         dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#
#         if not isinstance(gnconv, list):
#             gnconv = [gnconv, gnconv, gnconv, gnconv]
#         else:
#             gnconv = gnconv
#             assert len(gnconv) == 4
#
#         cur = 0
#         for i in range(4):
#             stage = nn.Sequential(
#                 *[block(dim=dims[i], drop_path=dp_rates[cur + j],
#                         layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for j in range(depths[i])]
#             )
#             self.stages.append(stage)
#             cur += depths[i]
#
#
#         self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
#         self.head = nn.Linear(dims[-1], num_classes)
#
#         self.uniform_init = uniform_init
#
#         self.apply(self._init_weights)
#         self.head.weight.data.mul_(head_init_scale)
#         self.head.bias.data.mul_(head_init_scale)
#
#     def _init_weights(self, m):
#         if not self.uniform_init:
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 trunc_normal_(m.weight, std=.02)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#         else:
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward_features(self, x):
#         for i in range(4):
#             x = self.downsample_layers[i](x)
#             for j, blk in enumerate(self.stages[i]):
#                 x = blk(x)
#         return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_model(model_name,inchannels,n_class):
    if model_name == "hornet_tiny_7x7":
        model = hornet_tiny_7x7(inchannels,n_class,pretrained=False,in_22k=False)
    elif model_name == "hornet_small_gf":
        model = hornet_small_gf(inchannels,n_class,pretrained=False,in_22k=False)
    else:
        raise NotImplementedError
    return model

@register_model
def hornet_tiny_7x7(inchannels,n_class,pretrained=False, in_22k=False, **kwargs):
    s = 1.0 / 3.0
    model = HorNet(in_chans=inchannels,num_classes=n_class,depths=[2, 3, 18, 2], base_dim=96, block=Block,
                   gnconv=[
                       partial(gnconv, order=2, s=s),
                       partial(gnconv, order=3, s=s),
                       partial(gnconv, order=4, s=s),
                       partial(gnconv, order=5, s=s),
                   ],
                   **kwargs
                   )
    return model


@register_model
def hornet_tiny_gf(pretrained=False, in_22k=False, **kwargs):
    s = 1.0 / 3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=64, block=Block,
                   gnconv=[
                       partial(gnconv, order=2, s=s),
                       partial(gnconv, order=3, s=s),
                       partial(gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
                       partial(gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter),
                   ],
                   **kwargs
                   )
    return model


@register_model
def hornet_small_7x7(inchannels,n_class,pretrained=False, in_22k=False, **kwargs):
    s = 1.0 / 3.0
    model = HorNet(in_chans=inchannels,num_classes=n_class,depths=[2, 3, 18, 2], base_dim=96, block=Block,
                   gnconv=[
                       partial(gnconv, order=2, s=s),
                       partial(gnconv, order=3, s=s),
                       partial(gnconv, order=4, s=s),
                       partial(gnconv, order=5, s=s),
                   ],
                   **kwargs
                   )
    return model


@register_model
def hornet_small_gf(inchannels,n_class,pretrained=False, in_22k=False, **kwargs):
    s = 1.0 / 3.0
    model = HorNet(in_chans=inchannels,num_classes=n_class,depths=[2, 3, 18, 2], base_dim=96, block=Block,
                   gnconv=[
                       partial(gnconv, order=2, s=s),
                       partial(gnconv, order=3, s=s),
                       partial(gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
                       partial(gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter),
                   ],
                   **kwargs
                   )
    return model


@register_model
def hornet_base_7x7(pretrained=False, in_22k=False, **kwargs):
    s = 1.0 / 3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=128, block=Block,
                   gnconv=[
                       partial(gnconv, order=2, s=s),
                       partial(gnconv, order=3, s=s),
                       partial(gnconv, order=4, s=s),
                       partial(gnconv, order=5, s=s),
                   ],
                   **kwargs
                   )
    return model


@register_model
def hornet_base_gf(pretrained=False, in_22k=False, **kwargs):
    s = 1.0 / 3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=128, block=Block,
                   gnconv=[
                       partial(gnconv, order=2, s=s),
                       partial(gnconv, order=3, s=s),
                       partial(gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
                       partial(gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter),
                   ],
                   **kwargs
                   )
    return model


@register_model
def hornet_base_gf_img384(pretrained=False, in_22k=False, **kwargs):
    s = 1.0 / 3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=128, block=Block,
                   gnconv=[
                       partial(gnconv, order=2, s=s),
                       partial(gnconv, order=3, s=s),
                       partial(gnconv, order=4, s=s, h=24, w=13, gflayer=GlobalLocalFilter),
                       partial(gnconv, order=5, s=s, h=12, w=7, gflayer=GlobalLocalFilter),
                   ],
                   **kwargs
                   )
    return model


@register_model
def hornet_large_7x7(pretrained=False, in_22k=False, **kwargs):
    s = 1.0 / 3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=192, block=Block,
                   gnconv=[
                       partial(gnconv, order=2, s=s),
                       partial(gnconv, order=3, s=s),
                       partial(gnconv, order=4, s=s),
                       partial(gnconv, order=5, s=s),
                   ],
                   **kwargs
                   )
    return model


@register_model
def hornet_large_gf(in_channal,n_class,pretrained=False, in_22k=False, **kwargs):
    s = 1.0 / 3.0
    model = HorNet(in_chans=in_channal,num_classes=n_class,depths=[2, 3, 18, 2], base_dim=192, block=Block,
                   gnconv=[
                       partial(gnconv, order=2, s=s),
                       partial(gnconv, order=3, s=s),
                       partial(gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
                       partial(gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter),
                   ],
                   **kwargs
                   )
    return model


@register_model
def hornet_large_gf_img384(pretrained=False, in_22k=False, **kwargs):
    s = 1.0 / 3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=192, block=Block,
                   gnconv=[
                       partial(gnconv, order=2, s=s),
                       partial(gnconv, order=3, s=s),
                       partial(gnconv, order=4, s=s, h=24, w=13, gflayer=GlobalLocalFilter),
                       partial(gnconv, order=5, s=s, h=12, w=7, gflayer=GlobalLocalFilter),
                   ],
                   **kwargs
                   )
    return model


if __name__=='__main__':


    net = HorNet(in_chans=4, num_classes=2)

# 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将网络移动到 GPU
    net.to(device)

# 定义一个输入张量
    input_tensor = torch.randn(4, 4, 512, 512).to(device)

# 输出网络输入


# 网络处理输入
    output_tensor = net(input_tensor)

# 输出网络输出
  #  print("GPU处理后的张量：",output_tensor.shape)