import torch
import itertools
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from mmcv.cnn import ConvModule
from modules.backbone.config.SS2D import SS2D
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data
from timm.models.registry import register_model
from timm.models.layers import DropPath
from networks.ResNet import ResNet34,ResNet50
from modules.attentions.CBAM import CBAM
from models.CMTFMnet import MAF
from modules.block.branchs import RFB_modified


class ModelRegistry(dict):
    def register_module(self, fn=None, *, name=None, override=False):
        if fn is None:
            # 支持带参数装饰器写法
            def decorator(f):
                key = name or f.__name__
                if not override and key in self:
                    raise KeyError(f"Module '{key}' already registered!")
                self[key] = f
                return f

            return decorator
        else:
            key = name or fn.__name__
            if not override and key in self:
                raise KeyError(f"Module '{key}' already registered!")
            self[key] = fn
            return fn


MODEL = ModelRegistry()


# 普通卷积操作
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


# 上采样
class UpsampleBilinear(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super(UpsampleBilinear, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners)


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# class MBWTConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',ssm_ratio=1,forward_type="v05",):
#         super(MBWTConv2d, self).__init__()
#
#         assert in_channels == out_channels
#
#         self.in_channels = in_channels
#         self.wt_levels = wt_levels
#         self.stride = stride
#         self.dilation = 1
#         self.brance =RFB_modified(in_channels,in_channels)
#
#         self.global_atten =SS2D(d_model=in_channels, d_state=1,
#              ssm_ratio=ssm_ratio, initialize="v2", forward_type=forward_type, channel_first=True, k_group=2)
#         #可学习模块
#         self.base_scale = _ScaleModule([1, in_channels, 1, 1])
#
#
#
#         if self.stride > 1:
#             self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
#             self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
#                                                    groups=in_channels)
#         else:
#             self.do_stride = None
#
#     def forward(self, x):
#
#         x_brance =self.brance(x)
#         x = self.base_scale(self.global_atten(x))
#         x = x+x_brance
#
#         if self.do_stride is not None:
#             x = self.do_stride(x)
#
#         return x
# 小波变换卷积层
class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',
                 ssm_ratio=1, forward_type="v05", ):
        super(MBWTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        # 小波变换和逆小波变换
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.global_atten = SS2D(d_model=in_channels, d_state=1,
                                 ssm_ratio=ssm_ratio, initialize="v2", forward_type=forward_type, channel_first=True,
                                 k_group=2)

        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )

        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.global_atten(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


# 可学习缩放模块
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                  groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

        # Initialize batch norm weights
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3 and bn1
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1 and bn2
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups,
                                    device=dwconv1x1.weight.device)
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        # Create a new sequential model with fused layers
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, ):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


# 类似下采样模块
class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, )
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, )
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, )

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


# 残差连接
class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


# 可替换E-FFN
class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(self.avg_pool(x))
        return x * scale


# 可替换E-FFN
class E_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 ksize=5, dilation=1, act_layer=nn.ReLU6, drop=0.,
                 fusion='add', use_se=True):
        super(E_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)

        self.conv1 = ConvBNReLU(
            in_channels=hidden_features,
            out_channels=hidden_features,
            kernel_size=ksize,
            dilation=dilation,
            groups=hidden_features
        )

        self.conv2 = ConvBNReLU(
            in_channels=hidden_features,
            out_channels=hidden_features,
            kernel_size=3,
            dilation=dilation,
            groups=hidden_features
        )

        self.fusion = fusion
        if fusion == 'concat':
            self.fusion_conv = ConvBNReLU(
                in_channels=hidden_features * 2,
                out_channels=hidden_features,
                kernel_size=1
            )

        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features)
        )

        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.act = act_layer()
        self.use_se = use_se
        if use_se:
            self.se = SEModule(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        if self.fusion == 'add':
            x = x1 + x2
        elif self.fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
            x = self.fusion_conv(x)
        else:
            raise ValueError("fusion must be 'add' or 'concat'")

        x = self.fc2(x)
        x = self.drop(x)

        if self.use_se:
            x = self.se(x)

        x = self.act(x)
        return x


# 优化内存
def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16

        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple
        else:
            return upper_multiple


# 将输入张量 x 按通道维度（dim=1），分成三部分：x1, x2, x3。
class MobileMambaModule(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=3, ssm_ratio=1, forward_type="v052d", ):
        super().__init__()
        self.dim = dim
        self.global_channels = nearest_multiple_of_16(int(global_ratio * dim))
        if self.global_channels + int(local_ratio * dim) > dim:
            self.local_channels = dim - self.global_channels
        else:
            self.local_channels = int(local_ratio * dim)
        self.identity_channels = self.dim - self.global_channels - self.local_channels
        if self.local_channels != 0:
            self.local_op = DWConv2d_BN_ReLU(self.local_channels, self.local_channels, kernels)
        else:
            self.local_op = nn.Identity()
        if self.global_channels != 0:
            self.global_op = MBWTConv2d(self.global_channels, self.global_channels, kernels, wt_levels=1,
                                        ssm_ratio=ssm_ratio, forward_type=forward_type, )
        else:
            self.global_op = nn.Identity()

        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0, ))

    def forward(self, x):  # x (B,C,H,W)
        x1, x2, x3 = torch.split(x, [self.global_channels, self.local_channels, self.identity_channels], dim=1)
        x1 = self.global_op(x1)
        x2 = self.local_op(x2)
        x = self.proj(torch.cat([x1, x2, x3], dim=1))
        return x


class MobileMambaBlockWindow(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=5, ssm_ratio=1, forward_type="v052d", ):
        super().__init__()
        self.dim = dim
        self.attn = MobileMambaModule(dim, global_ratio=global_ratio, local_ratio=local_ratio,
                                      kernels=kernels, ssm_ratio=ssm_ratio, forward_type=forward_type, )

    def forward(self, x):
        x = self.attn(x)
        return x


# nn.Identity()输入什么，输出也是什么。
class MobileMambaBlock(torch.nn.Module):
    def __init__(self, type,
                 ed, global_ratio=0.25, local_ratio=0.25,
                 kernels=5, drop_path=0., has_skip=True, ssm_ratio=1, forward_type="v052d"):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))

        if type == 's':
            self.mixer = Residual(MobileMambaBlockWindow(ed, global_ratio=global_ratio, local_ratio=local_ratio,
                                                         kernels=kernels, ssm_ratio=ssm_ratio,
                                                         forward_type=forward_type))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., ))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))

        self.has_skip = has_skip
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)
        self.CBAM = CBAM(in_chans=dim)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        x = self.CBAM(x)
        return x


class Block(nn.Module):
    def __init__(self, stg, ed, gr=0.5, lr=0.2, kernels=5, drop_path=0., ssm_ratio=1, forward_type="v052d"):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(ed)
        self.mab = MobileMambaBlock(stg, ed, gr, lr, kernels, drop_path, ssm_ratio=ssm_ratio, forward_type=forward_type)

        self.conv_bn = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.FFN = Residual(FFN(ed, int(ed * 2)))
        # self.FFN = Residual(E_FFN(ed, int(2 * ed), fusion='concat', dilation=2, act_layer=nn.GELU))

    def forward(self, x):
        x = x + self.drop_path(self.norm1(self.mab(self.conv_bn(x))))
        x = x + self.drop_path(self.FFN(x))
        return x


class RBnet(torch.nn.Module):
    def __init__(self,
                 in_chans=4,
                 # num_classes=1000,
                 # stages=['s', 's', 's','s','s'],
                 # embed_dim=[256,160,80,48,32],
                 # global_ratio=[0.5,0.5,0.6,0.7,0.8],
                 # local_ratio=[0.2, 0.2, 0.3, 0.2, 0.2],
                 # depth=[1, 1 , 1, 1, 1],
                 # kernels=[3, 3, 5, 5, 7],
                 # backbone =MobileNetV4,
                 # drop_path=0.,
                 num_classes=1000,
                 stages=['s', 's', 's', 's'],
                 embed_dim=[512, 256,128,64],
                 global_ratio=[0.5, 0.5, 0.6, 0.7],
                 local_ratio=[0.2, 0.2, 0.3, 0.2],
                 depth=[1, 1, 1, 1],
                 kernels=[3, 3, 5, 7],
                 backbone=ResNet34,
                 drop_path=0.,
                 ssm_ratio=1, forward_type="v052d"):
        super().__init__()
        self.n_classes = num_classes
        self.backbone = backbone()
        tot_channels = sum(embed_dim)
        # Patch embedding
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depth))]

        self.c5 = ConvBNReLU(embed_dim[0], embed_dim[0], 1)
        self.b5 = Block(stages[0], embed_dim[0], global_ratio[0], local_ratio[0], kernels[0], dprs[0])

        self.c4 = ConvBNReLU(embed_dim[0], embed_dim[1], 1)
        self.p4 = Fusion(embed_dim[1])
        self.b4 = Block(stages[1], embed_dim[1], global_ratio[1], local_ratio[1], kernels[1], dprs[1])

        self.c3 = ConvBNReLU(embed_dim[1], embed_dim[2], 1)
        self.p3 = Fusion(embed_dim[2])
        self.b3 = Block(stages[2], embed_dim[2], global_ratio[2], local_ratio[2], kernels[2], dprs[2])

        self.c2 = ConvBNReLU(embed_dim[2], embed_dim[3], 1)
        self.p2 = Fusion(embed_dim[3])
        self.b2 = Block(stages[3], embed_dim[3], global_ratio[3], local_ratio[3], kernels[3], dprs[3])

        # self.c1 = ConvBNReLU(embed_dim[3], embed_dim[4], 1)
        # self.p1 = Fusion(embed_dim[4])
        # self.b1 =Block(stages[4],embed_dim[4],global_ratio[4],local_ratio[4],kernels[4],dprs[4])
        # self.MAF = MAF(dim=48, fc_ratio=4, dropout=0.1, num_classes=num_classes)
        # self.up = UpsampleBilinear(scale_factor=4)

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embed_dim[2],
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(embed_dim[3], embed_dim[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim[3]),
            nn.ReLU(),
            UpsampleBilinear(scale_factor=4),
            nn.Conv2d(embed_dim[3], num_classes, kernel_size=1, stride=1))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        h, w = x.size()[-2:]
        # h=int(h/4)
        # w=int(w/4)
        x2, x3, x4, x5 = self.backbone(x)
        x5 = self.c5(x5)
        x5 = self.b5(x5)

        x_5 = self.c4(x5)
        x4 = self.p4(x_5, x4)
        x4 = self.b4(x4)

        x_4 = self.c3(x4)
        x3 = self.p3(x_4, x3)
        x3 = self.b3(x3)

        x_3 = self.c2(x3)
        x2 = self.p2(x_3, x2)
        x2 = self.b2(x2)

        # x =self.c1(x)
        # x =self.p1(x,x1)
        # x =self.b1(x)

        x = self.last_conv(x2)
        # x =self.MAF(x)
        # x =self.up(x)
        return x


class RBnet_1(torch.nn.Module):
    def __init__(self,
                 in_chans=4,
                 # num_classes=1000,
                 # stages=['s', 's', 's','s','s'],
                 # embed_dim=[256,160,80,48,32],
                 # global_ratio=[0.5,0.5,0.6,0.7,0.8],
                 # local_ratio=[0.2, 0.2, 0.3, 0.2, 0.2],
                 # depth=[1, 1 , 1, 1, 1],
                 # kernels=[3, 3, 5, 5, 7],
                 # backbone =MobileNetV4,
                 # drop_path=0.,
                 num_classes=1000,
                 stages=['s', 's', 's', 's'],
                 embed_dim=[512, 192, 96, 48],
                 global_ratio=[0.5, 0.5, 0.6, 0.7],
                 local_ratio=[0.2, 0.2, 0.3, 0.2],
                 depth=[1, 1, 1, 1],
                 kernels=[3, 3, 5, 7],
                 backbone=ResNet34,
                 drop_path=0.,
                 ssm_ratio=1, forward_type="v052d"):
        super().__init__()
        self.n_classes = num_classes
        self.backbone = backbone()
        tot_channels = sum(embed_dim)
        # Patch embedding
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depth))]

        self.c5 = ConvBNReLU(embed_dim[0], embed_dim[0], 1)
        self.b5 = Block(stages[0], embed_dim[0], global_ratio[0], local_ratio[0], kernels[0], dprs[0])

        self.c4 = ConvBNReLU(embed_dim[0], embed_dim[1], 1)
        self.p4 = Fusion(embed_dim[1])
        self.b4 = Block(stages[1], embed_dim[1], global_ratio[1], local_ratio[1], kernels[1], dprs[1])

        self.c3 = ConvBNReLU(embed_dim[1], embed_dim[2], 1)
        self.p3 = Fusion(embed_dim[2])
        self.b3 = Block(stages[2], embed_dim[2], global_ratio[2], local_ratio[2], kernels[2], dprs[2])

        self.c2 = ConvBNReLU(embed_dim[2], embed_dim[3], 1)
        self.p2 = Fusion(embed_dim[3])
        self.b2 = Block(stages[3], embed_dim[3], global_ratio[3], local_ratio[3], kernels[3], dprs[3])

        # self.c1 = ConvBNReLU(embed_dim[3], embed_dim[4], 1)
        # self.p1 = Fusion(embed_dim[4])
        # self.b1 =Block(stages[4],embed_dim[4],global_ratio[4],local_ratio[4],kernels[4],dprs[4])
        # self.MAF = MAF(dim=48, fc_ratio=4, dropout=0.1, num_classes=num_classes)
        # self.up = UpsampleBilinear(scale_factor=4)

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embed_dim[2],
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(embed_dim[2], embed_dim[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim[2]),
            nn.ReLU(),
            # UpsampleBilinear(scale_factor=4),
            nn.Conv2d(embed_dim[2], num_classes, kernel_size=1, stride=1))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        h, w = x.size()[-2:]
        # h=int(h/4)
        # w=int(w/4)
        x2, x3, x4, x5 = self.backbone(x)
        x5 = self.c5(x5)
        x5 = self.b5(x5)

        x_5 = self.c4(x5)
        x4 = self.p4(x_5, x4)
        x4 = self.b4(x4)

        x_4 = self.c3(x4)
        x3 = self.p3(x_4, x3)
        x3 = self.b3(x3)

        x_3 = self.c2(x3)
        x2 = self.p2(x_3, x2)
        x2 = self.b2(x2)

        # x =self.c1(x)
        # x =self.p1(x,x1)
        # x =self.b1(x)

        x5 = F.interpolate(x5, size=(h, w), mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(h, w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=False)

        x1 = self.linear_fuse(torch.cat([x2, x3, x4, x5], dim=1))

        x = self.last_conv(x1)
        # x =self.MAF(x)
        # x =self.up(x)
        return x


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


if __name__ == "__main__":
    net = RBnet(num_classes=2)

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将网络移动到 GPU
    net.to(device)

    # 定义一个输入张量
    input_tensor = torch.randn(4, 4, 512, 512).to(device)

    # 输出网络输入

    # 网络处理输入
    x = net(input_tensor)

    print(x.shape)
# CFG_MobileMamba_T2 = {
#         'img_size': 192,
#         'embed_dim': [144, 272, 368],
#         'depth': [1, 2, 2],
#         'global_ratio': [0.8, 0.7, 0.6],
#         'local_ratio': [0.2, 0.2, 0.3],
#         'kernels': [7, 5, 3],
#         'drop_path': 0,
#         'ssm_ratio': 2,
#     }
# CFG_MobileMamba_T4 = {
#         'img_size': 192,
#         'embed_dim': [176, 368, 448],
#         'depth': [1, 2, 2],
#         'global_ratio': [0.8, 0.7, 0.6],
#         'local_ratio': [0.2, 0.2, 0.3],
#         'kernels': [7, 5, 3],
#         'drop_path': 0,
#         'ssm_ratio': 2,
#     }
# CFG_MobileMamba_S6 = {
#         'img_size': 224,
#         'embed_dim': [192, 384, 448],
#         'depth': [1, 2, 2],
#         'global_ratio': [0.8, 0.7, 0.6],
#         'local_ratio': [0.2, 0.2, 0.3],
#         'kernels': [7, 5, 3],
#         'drop_path': 0,
#         'ssm_ratio': 2,
#     }
# CFG_MobileMamba_B1 = {
#         'img_size': 256,
#         'embed_dim': [200, 376, 448],
#         'depth': [2, 3, 2],
#         'global_ratio': [0.8, 0.7, 0.6],
#         'local_ratio': [0.2, 0.2, 0.3],
#         'kernels': [7, 5, 3],
#         'drop_path': 0.03,
#         'ssm_ratio': 2,
#     }
# CFG_MobileMamba_B2 = {
#         'img_size': 384,
#         'embed_dim': [200, 376, 448],
#         'depth': [2, 3, 2],
#         'global_ratio': [0.8, 0.7, 0.6],
#         'local_ratio': [0.2, 0.2, 0.3],
#         'kernels': [7, 5, 3],
#         'drop_path': 0.03,
#         'ssm_ratio': 2,
#     }
# CFG_MobileMamba_B4 = {
#         'img_size': 512,
#         'embed_dim': [192, 384, 448],
#         'depth': [2, 3, 2],
#         'global_ratio': [0.8, 0.7, 0.6],
#         'local_ratio': [0.2, 0.2, 0.3],
#         'kernels': [7, 5, 3],
#         'drop_path': 0.03,
#         'ssm_ratio': 2,
#     }
#
#
# @register_model
# def MobileMamba_T2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_T2):
#     model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
#     if fuse:
#         replace_batchnorm(model)
#     return model
# @register_model
# def MobileMamba_T4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_T4):
#     model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
#     if fuse:
#         replace_batchnorm(model)
#     return model
# @MODEL.register_module
# def MobileMamba_S6(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_S6):
#     model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
#     if fuse:
#         replace_batchnorm(model)
#     return model
# @MODEL.register_module
# def MobileMamba_B1(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_B1):
#     model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
#     if fuse:
#         replace_batchnorm(model)
#     return model
# @MODEL.register_module
# def MobileMamba_B2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_B2):
#     model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
#     if fuse:
#         replace_batchnorm(model)
#     return model
# @MODEL.register_module
# def MobileMamba_B4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_B4):
#     model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
#     if fuse:
#         replace_batchnorm(model)
#     return model


# from modules.backbone.config.SS2D import FLOPs
# import time
# import argparse
#
# def get_timepc():
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     return time.perf_counter()
#
#
# model_dict = {
#     "MobileMamba_T2": MobileMamba_T2,
#     "MobileMamba_T4": MobileMamba_T4,
#     "MobileMamba_S6": MobileMamba_S6,
#     "MobileMamba_B1": MobileMamba_B1,
#     "MobileMamba_B2": MobileMamba_B2,
#     "MobileMamba_B4": MobileMamba_B4,
# }
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batchsize', type=int, default=256)
# parser.add_argument('-i', '--imagesize', type=int, default=224)
# parser.add_argument('-m', '--modelname', default="MobileMamba_S6")
# cfg = parser.parse_args()
# bs = cfg.batchsize
# img_size = cfg.imagesize
# model_name = cfg.modelname
# print('batch_size is:', bs, 'img_size is:', img_size, 'model_name is:', model_dict[model_name])
# gpu_id = 0
# speed = True
# latency = True
# with torch.no_grad():
#     x = torch.randn(bs, 3, img_size, img_size)
#     net = model_dict[model_name]()
#     replace_batchnorm(net)
#     net.eval()
#     pre_cnt, cnt = 2, 5
#     if gpu_id > -1:
#         torch.cuda.set_device(gpu_id)
#         x = x.cuda()
#         net.cuda()
#         pre_cnt, cnt = 50, 20
#     FLOPs.fvcore_flop_count(net, torch.randn(1, 3, img_size, img_size).cuda(), show_arch=False)
#
#     #GPU
#     for _ in range(pre_cnt):
#         net(x)
#     t_s = get_timepc()
#     for _ in range(cnt):
#         net(x)
#     t_e = get_timepc()
#     speed = f'{bs * cnt / (t_e - t_s):>7.3f}'
#     print(f'[Batchsize: {bs}]\t [GPU-Speed: {speed}]\t')