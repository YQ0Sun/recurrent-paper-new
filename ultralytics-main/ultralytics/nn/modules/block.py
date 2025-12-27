# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d


from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, DWConvNoAct, DWConvWithAct, CBAM, AGCA, CBAM_LD_SAM
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'DWConvC2f', 'DWBottleneck3',
           'DWBottleneck5', 'DWConvC2f_total', 'DWConvC2f_5', 'DWConvC2f_total_5_3', 'DWConvC2f_5_CBAM',
           'DWBottleneck5_CBAM', 'DWBottleneck5_CBAM_there', 'DWConvC2f_5_CBAM_there', 'DWBottleneck5_afterConcatCBAM',
           'DWConvC2f_5_CBAM_afterConcatCBAM', 'ASFF2', 'ASFF3', 'C2f_with_AFPN', 'DWBottleneck5_CBAM_new',
           'DWBottleneck7', 'DWBottleneck5_AGCA', 'DWBottleneck5_CBAM_LD_SAM', 'C2f_with_weight')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class Upsample(nn.Module):
    """Applies convolution followed by upsampling."""
    # CARAFE算法？

    def __init__(self, c1, c2, scale_factor=2):
        super().__init__()
        # self.cv1 = Conv(c1, c2, 1)
        # self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')  # or model='bilinear' non-deterministic
        if scale_factor == 2:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        elif scale_factor == 4:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 4, 4, 0, bias=True)  # nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, x):
        # return self.upsample(self.cv1(x))
        return self.cv1(x)


class ASFF2(nn.Module):
    """ASFF2 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_h = c1[0], c1[1]
        self.level = level
        self.dim = c1_l, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_h, self.inter_dim)
        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # downsample 2x

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1 = x[0], x[1]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weights_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1] + level_1_resized * levels_weight[:, 1:2]
        return self.conv(fused_out_reduced)


class ASFF3(nn.Module):
    """ASFF3 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = level
        self.dim = c1_l, c1_m, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)

        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # downsample 2x
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)

        if level == 2:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 4, 4, 0)  # downsample 4x
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 2, 2, 0)  # downsample 2x

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        w = self.weights_levels(levels_weight_v)
        w = F.softmax(w, dim=1)

        fused_out_reduced = level_0_resized * w[:, :1] + level_1_resized * w[:, 1:2] + level_2_resized * w[:, 2:]
        return self.conv(fused_out_reduced)

class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

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

class MLKA_Ablation(nn.Module):
    def __init__(self, n_feats):
        super(MLKA_Ablation, self).__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = BatchNorm2d(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        k = 2

        # Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // k, n_feats // k, 7, 1, 7 // 2, groups=n_feats // k),
            nn.Conv2d(n_feats // k, n_feats // k, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // k, dilation=4),
            nn.Conv2d(n_feats // k, n_feats // k, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // k, n_feats // k, 5, 1, 5 // 2, groups=n_feats // k),
            nn.Conv2d(n_feats // k, n_feats // k, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // k, dilation=3),
            nn.Conv2d(n_feats // k, n_feats // k, 1, 1, 0))
        '''self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//k, n_feats//k, 3, 1, 1, groups= n_feats//k),  
            nn.Conv2d(n_feats//k, n_feats//k, 5, stride=1, padding=(5//2)*2, groups=n_feats//k, dilation=2),
            nn.Conv2d(n_feats//k, n_feats//k, 1, 1, 0))'''

        # self.X3 = nn.Conv2d(n_feats//k, n_feats//k, 3, 1, 1, groups= n_feats//k)
        self.X5 = nn.Conv2d(n_feats // k, n_feats // k, 5, 1, 5 // 2, groups=n_feats // k)
        self.X7 = nn.Conv2d(n_feats // k, n_feats // k, 7, 1, 7 // 2, groups=n_feats // k)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        # u_1, u_2, u_3= torch.chunk(u, 3, dim=1)
        a_1, a_2 = torch.chunk(a, 2, dim=1)

        a = torch.cat([self.LKA7(a_1) * self.X7(a_1), self.LKA5(a_2) * self.X5(a_2)], dim=1)

        x = self.proj_last(x * a) * self.scale + shortcut

        return x

# LCAM 模块
class LCAM(nn.Module):
    def __init__(self, channel):
        super(LCAM, self).__init__()

        # 池化操作保持输入大小不变
        # self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 确保卷积层输入通道与输入数据一致
        self.conv1_max = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv1_avg = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

        # 最后的调整卷积
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 最大池化分支
        max_out = self.conv1_max(x)
        max_out = self.relu(max_out)

        # 平均池化分支
        avg_out = self.conv1_avg(x)
        avg_out = self.relu(avg_out)

        # 融合池化结果
        out = max_out + avg_out
        # out = self.conv2(out)  # 最终调整
        out = torch.sigmoid(out)

        return out


# LD-SAM 模块
class LD_SAM(nn.Module):
    def __init__(self):
        super(LD_SAM, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([max_out, mean_out], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
# LCBHAM 模块
class LCBHAM(nn.Module):
    def __init__(self, channel):
        super(LCBHAM, self).__init__()
        # self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)  # k=3, s=2
        # self.bn = nn.BatchNorm2d(channel)
        # self.act = h_swish()

        self.lcam = LCAM(channel)
        self.ld_sam = LD_SAM()

    def forward(self, x):
        # x = self.conv(x)
        # x = self.bn(x)
        # x = self.act(x)

        lcam_out = self.lcam(x)
        print(lcam_out.shape)
        print(x.shape)
        lcam_out = x * lcam_out

        ld_sam_out = self.ld_sam(lcam_out)
        output = ld_sam_out * lcam_out

        return output

# class C2f_with_weight(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions, with added weight scaling."""
#
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """
#         Initialize CSP bottleneck layer with two convolutions, with weight scaling.
#         :param c1: Input channels.
#         :param c2: Output channels.
#         :param n: Number of Bottleneck modules.
#         :param shortcut: Whether to use residual connections in Bottleneck.
#         :param g: Groups for convolutions.
#         :param e: Expansion ratio for hidden channels.
#         :param compress_c: Compressed channel dimension for weight scaling.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial convolution
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Final convolution
#         self.m = nn.ModuleList(
#             DWBottleneck5(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
#         )  # Bottleneck layers
#
#         # Add weight scaling modules for each Bottleneck layer
#         self.weight_modules = nn.ModuleList(
#             Conv(self.c, self.c, 1, 1) for _ in range(n)
#         )
#
#     def forward(self, x):
#         """Forward pass through C2f layer with weight scaling."""
#         y = list(self.cv1(x).chunk(2, 1))  # Split input into two parts
#         for i, m in enumerate(self.m):
#             bottleneck_out = m(y[-1])  # Pass through Bottleneck
#             weight = self.weight_modules[i](bottleneck_out)  # Compute weight
#             scaled_out = bottleneck_out * torch.sigmoid(weight)  # Scale with sigmoid of weight
#             y.append(scaled_out)  # Append scaled output to list
#         return self.cv2(torch.cat(y, 1))  # Concatenate and pass through final Conv

class C2f_with_weight(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with added weight scaling."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize CSP bottleneck layer with two convolutions, with weight scaling.
        :param c1: Input channels.
        :param c2: Output channels.
        :param n: Number of Bottleneck modules.
        :param shortcut: Whether to use residual connections in Bottleneck.
        :param g: Groups for convolutions.
        :param e: Expansion ratio for hidden channels.
        :param compress_c: Compressed channel dimension for weight scaling.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial convolution
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Final convolution
        self.m = nn.ModuleList(
            # train35 到 train38都是这个
            # Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
            # train39 DWConv sigmoid DWBottleneck5
            # DWBottleneck5(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
            # train41 在train40的基础上将DWBottleneck5替换成DWBottleneck5_CBAM_new
            # DWBottleneck5_CBAM_new(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
            # DWBottleneck5_CBAM(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
            DWBottleneck5_CBAM_LD_SAM(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )  # Bottleneck layers

        # Add weight scaling modules for each Bottleneck layer
        self.weight_modules = nn.ModuleList(
            # train35 have sigmoid(weight) Conv
            # Conv(self.c, self.c, 1, 1) for _ in range(n)
            # train38 use DWConv use sigmoid
            # DWConv(self.c, self.c, 1, 1) for _ in range(n)
            # train40 Conv sigmoid DWBottleneck5 效果好
            Conv(self.c, self.c, 1, 1) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer with weight scaling."""
        y = list(self.cv1(x).chunk(2, 1))  # Split input into two parts
        for i, m in enumerate(self.m):
            bottleneck_out = m(y[-1])  # Pass through Bottleneck
            weight = self.weight_modules[i](bottleneck_out)  # Compute weight
            # train37 use torch.softmax Conv sigmoidc
            # scaled_out = bottleneck_out * torch.softmax(weight, dim=1)  # Scale with sigmoid of weight
            # train35 have sigmoid(weight) Conv sigmoid
            scaled_out = bottleneck_out * torch.sigmoid(weight)  # Scale with sigmoid of weight
            # train36 don't use sigmoid(weight) sigmoid use Conv
            # scaled_out = bottleneck_out * weight  # Scale with sigmoid of weight
            y.append(scaled_out)  # Append scaled output to list
        return self.cv2(torch.cat(y, 1))  # Concatenate and pass through final Conv

    # def forward(self, x):
    #     """Forward pass through C2f layer with weight scaling."""
    #     y = list(self.cv1(x).chunk(2, 1))  # Split input into two parts
    #     for i, m in enumerate(self.m):
    #         bottleneck_out = m(y[-1])  # Pass through Bottleneck
    #         weight = self.weight_modules[i](bottleneck_out)  # Compute weight
    #         # train37 use torch.softmax Conv sigmoid
    #         # scaled_out = bottleneck_out * torch.softmax(weight, dim=1)  # Scale with sigmoid of weight
    #         # train35 have sigmoid(weight) Conv sigmoid
    #         scaled_out = bottleneck_out * torch.sigmoid(weight)  # Scale with sigmoid of weight
    #         # train36 don't use sigmoid(weight) sigmoid use Conv
    #         # scaled_out = bottleneck_out * weight  # Scale with sigmoid of weight
    #         y.append(scaled_out)  # Append scaled output to list
    #     return self.cv2(torch.cat(y, 1))  # Concatenate and pass through final Conv


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

# 论文：A deep learning model for steel surface defect detection



class DWBottleneck5_CBAM_new(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConvNoAct(c1, c_, 1, 1, padding=1)  # 1x1 convolution
        self.cv2 = DWConvWithAct(c_, c2, 5, 1, padding=1)  # 5x5 convolution with BatchNorm and SiLU
        self.cbam = CBAM(c1)  # Initialize CBAM with input channels c1
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        y1 = self.cv2(self.cv1(x))
        y2 = self.cbam(y1)
        # train40y1 + y2 + x if self.add else y1
        return y1 + y2 + x if self.add else y2

class DWBottleneck5_CBAM_LD_SAM(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConvNoAct(c1, c_, 1, 1, padding=1)  # 1x1 convolution
        self.cv2 = DWConvWithAct(c_, c2, 5, 1, padding=1)  # 5x5 convolution with BatchNorm and SiLU
        self.cbam_ld_sam = CBAM_LD_SAM(c1)  # Initialize CBAM with input channels c1
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        y1 = self.cv2(self.cv1(x))
        y2 = self.cbam_ld_sam(x)
        return y1 + y2 if self.add else y1

class DWBottleneck5_CBAM(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConvNoAct(c1, c_, 1, 1, padding=1)  # 1x1 convolution
        self.cv2 = DWConvWithAct(c_, c2, 5, 1, padding=1)  # 5x5 convolution with BatchNorm and SiLU
        self.cbam = CBAM(c2)  # Initialize CBAM with input channels c1
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        y1 = self.cv2(self.cv1(x))
        y2 = self.cbam(x)
        return y1 + y2 if self.add else y1

class DWBottleneck5(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConvNoAct(c1, c_, 1, 1, padding=1)  # 1x1 convolution
        self.cv2 = DWConvWithAct(c_, c2, 5, 1, padding=1)  # 3x3 convolution with BatchNorm and SiLU
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class DWBottleneck7(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConvNoAct(c1, c_, 1, 1, padding=1)  # 1x1 convolution
        self.cv2 = DWConvWithAct(c_, c2, 7, 1, padding=2)  # 3x3 convolution with BatchNorm and SiLU
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DWBottleneck3(nn.Module):
    """Standard bottleneck with depth-wise convolutions."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initialize bottleneck with given input/output channels, shortcut option, kernels, and expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConvNoAct(c1, c_, 1, 1, padding=0)  # 1x1 convolution
        self.cv2 = DWConvWithAct(c_, c2, 3, 1, padding=1)  # 3x3 convolution with BatchNorm and SiLU
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass through the bottleneck."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# 论文：A deep learning model for steel surface defect detection

class DWConvC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial convolution layer
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Output convolution layer
        self.m = nn.ModuleList(DWBottleneck3(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))   # Use DWBottleneck3

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))  # Split the output of cv1 into two parts
        y.extend(m(y[-1]) for m in self.m)  # Apply each DWBottleneck3 module to the last part
        return self.cv2(torch.cat(y, 1))  # Concatenate and apply cv2

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class DWConvC2f_5(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial convolution layer
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Output convolution layer
        self.m = nn.ModuleList(DWBottleneck5(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))   # Use DWBottleneck3

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))  # Split the output of cv1 into two parts
        y.extend(m(y[-1]) for m in self.m)  # Apply each DWBottleneck3 module to the last part
        return self.cv2(torch.cat(y, 1))  # Concatenate and apply cv2

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class DWConvC2f_total(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial convolution layer
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Output convolution layer
        if n == 1:
            self.m = nn.ModuleList([DWBottleneck3(self.c, self.c, shortcut, g, e=1.0)])
        elif n == 2:
            self.m = nn.ModuleList([DWBottleneck3(self.c, self.c, shortcut, g, e=1.0),
                                    DWBottleneck5(self.c, self.c, shortcut, g, e=1.0)])
        else:
            self.m = nn.ModuleList(DWBottleneck3(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))  # Split the output of cv1 into two parts
        y.extend(m(y[-1]) for m in self.m)  # Apply each DWBottleneck3 or DWBottleneck5 module to the last part
        return self.cv2(torch.cat(y, 1))  # Concatenate and apply cv2

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class DWConvC2f_total_5_3(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial convolution layer
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Output convolution layer
        if n == 1:
            self.m = nn.ModuleList([DWBottleneck5(self.c, self.c, shortcut, g, e=1.0)])
        elif n == 2:
            self.m = nn.ModuleList([DWBottleneck5(self.c, self.c, shortcut, g, e=1.0),
                                    DWBottleneck3(self.c, self.c, shortcut, g, e=1.0)])
        else:
            self.m = nn.ModuleList(DWBottleneck5(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))  # Split the output of cv1 into two parts
        y.extend(m(y[-1]) for m in self.m)  # Apply each DWBottleneck3 or DWBottleneck5 module to the last part
        return self.cv2(torch.cat(y, 1))  # Concatenate and apply cv2

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class DWConvC2f_5_CBAM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial convolution layer
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Output convolution layer
        self.m = nn.ModuleList(DWBottleneck5_CBAM(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))   # Use DWBottleneck3

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))  # Split the output of cv1 into two parts
        y.extend(m(y[-1]) for m in self.m)  # Apply each DWBottleneck3 module to the last part
        return self.cv2(torch.cat(y, 1))  # Concatenate and apply cv2

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class DWBottleneck5_CBAM_there(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConvNoAct(c1, c_, 1, 1, padding=1)  # 1x1 convolution
        self.cv2 = DWConvWithAct(c_, c2, 5, 1, padding=1)  # 5x5 convolution with BatchNorm and SiLU
        self.cbam = CBAM(c1)  # Initialize CBAM with input channels c1
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        y1 = self.cv2(self.cv1(x))
        y2 = self.cbam(x)
        return x + y1 + y2 if self.add else y1

class DWConvC2f_5_CBAM_there(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial convolution layer
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Output convolution layer
        self.m = nn.ModuleList(DWBottleneck5_CBAM_there(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))   # Use DWBottleneck3

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))  # Split the output of cv1 into two parts
        y.extend(m(y[-1]) for m in self.m)  # Apply each DWBottleneck3 module to the last part
        return self.cv2(torch.cat(y, 1))  # Concatenate and apply cv2

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class DWBottleneck5_afterConcatCBAM(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConvNoAct(c1, c_, 1, 1, padding=1)  # 1x1 convolution
        self.cv2 = DWConvWithAct(c_, c2, 5, 1, padding=1)  # 5x5 convolution with BatchNorm and SiLU
        self.cbam = CBAM(c2)  # Initialize CBAM with output channels c2
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        y = self.cv2(self.cv1(x))
        if self.add:
            y = x + y  # Apply shortcut if necessary
        return self.cbam(y)  # Apply CBAM after Concat or cv2

class DWConvC2f_5_CBAM_afterConcatCBAM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial convolution layer
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Output convolution layer
        self.m = nn.ModuleList(DWBottleneck5_afterConcatCBAM(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))   # Use DWBottleneck3

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))  # Split the output of cv1 into two parts
        y.extend(m(y[-1]) for m in self.m)  # Apply each DWBottleneck3 module to the last part
        return self.cv2(torch.cat(y, 1))  # Concatenate and apply cv2

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))