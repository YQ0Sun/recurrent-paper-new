# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from .NLBlockND import NLBlockND
from .MultiHeadAttention import *
import torch.nn.functional as F



__all__ = ('Conv', 'Conv2', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv', 'DWConvWithAct', 'DWConvNoAct', 'Add',
           'AGCA', 'CBAM_LD_SAM', 'EMA', 'AddNL', 'CrossAttention')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
# class DWConv1(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False):
#         super(DWConv1, self).__init__()
#         self.dwconv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, bias),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         x = self.dconv(x)
#         return x
#
# class DWConv2(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, kernel_size=(1, 5), stride=1, padding=(0, 1), groups=in_channels, bias=False):
#         super(DWConv2, self).__init__()
#         self.dwconv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, bias),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         x = self.dconv(x)
#         return x

class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class AGCA(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        # Choose to deploy A0 on GPU or CPU according to your needs
        # self.A0 = torch.eye(hide_channel).to('cuda')
        self.A0 = torch.eye(hide_channel)
        # A2 is initialized to 1e-6
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        if A1.device != self.A0.device:
            self.A0 = self.A0.to('cuda')
        A = (self.A0 * A1) + self.A2

        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))

        return x * y

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x,self.d)

# 论文：A deep learning model for steel surface defect detection
class DWConvNoAct(nn.Module):
    """Depth-wise convolution without BatchNorm and activation."""

    def __init__(self, c1, c2, k=1, s=1, d=1, padding=None):
        """Initialize Depth-wise convolution without BatchNorm and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=padding, groups=math.gcd(c1, c2), dilation=d)

    def forward(self, x):
        return self.conv(x)
class DWConvWithAct(nn.Module):
    """Depth-wise convolution with BatchNorm and activation."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True, padding=None):
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=padding, groups=math.gcd(c1, c2), dilation=d)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Add(nn.Module):
    """Add a list of tensors element-wise."""

    def __init__(self, param1, param2):
        """Initialize the Add module."""
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def forward(self, x):
        """Forward pass to element-wise add a list of tensors."""
        # Assuming x is a list of tensors
        if not isinstance(x, tuple):
            x = (x,)  # Convert x to a tuple of tensors if it's not already

        # Stack tensors along a new dimension (dim=0) and sum them along the same dimension
        # 实验415之前的
        stacked = torch.stack(x, dim=0)  # Stack tensors along dim=0
        result = torch.sum(stacked, dim=0)  # Sum along dim=0
        # 实验416之后的
        # result = torch.concat(x, dim=1)  # Sum along dim=0
        return result

class AddNL(nn.Module):
    """Add Non-Local attention to a single input tensor"""
    def __init__(self, param1, param2, channels=8):
        """Initialize the Add module."""
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.non_local = NLBlockND(in_channels=param1, dimension=2)


    def forward(self, x):
        x = self.non_local(x)
        return x



# h_swish 和 h_sigmoid 激活函数
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




# class ChannelAttention_1(nn.Module):
#     def __init__(self, channels: int) -> None:
#         super(ChannelAttention_1, self).__init__()
    #     self.pool = nn.AdaptiveAvgPool2d(1)
    #     self.group_linear = nn.Linear(channels, channels)
    #     self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
    #     self.act = nn.Sigmoid()
    #     self.relu = nn.ReLU()
    #
    # def forward(self, x):
    #     # Pool to [batch_size, channels, 1, 1]
    #     pooled = self.pool(x)
    #     # Flatten to [batch_size, channels]
    #     flattened = pooled.view(pooled.size(0), -1)
    #     # Pass through the first group_linear layer (fully connected)
    #     out = self.group_linear(flattened)
    #     # Reshape to match expected input for the Conv2d layer
    #     out = out.view(out.size(0), out.size(1), 1, 1)  # Reshape to [batch_size, channels, 1, 1]
    #     # Pass through the convolution layer
    #     out = self.fc(out)
    #     # Apply activation
    #     out = self.act(out)
    #     # Reshape back to match x's shape and multiply
    #     out = out.view(x.size(0), x.size(1), 1, 1)
    #     return x * out
    # super().__init__()
    # self.pool = nn.AdaptiveAvgPool2d(1)
    # self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
    # self.act = nn.Sigmoid()
class ChannelAttention_1(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.Apool = nn.AdaptiveAvgPool2d(1)
        self.Mpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.fc1 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=True)
        self.act = nn.ReLU()
        # self.softmax = nn.Softmax(-1)
        self.act1 = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x1 = self.act(self.fc(self.Apool(x)))
        # x2 = self.act(self.fc(self.Mpool(x)))
        # x3 = self.act1(x1 + x2)
        x1 = self.fc(self.Apool(x))
        x2 = self.fc(self.Mpool(x))
        x3 = x1 + x2

        x4 = self.act(self.fc1(x))
        return x * (self.act1(x3) + x4)

class ChannelAttentionWithSkip(nn.Module):
    """Channel-attention module with skip connection."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()

        # First convolution block with pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm1 = nn.BatchNorm2d(channels)
        self.act1 = nn.Sigmoid()

        # Second convolution block
        self.fc2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm2 = nn.BatchNorm2d(channels)
        self.act2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass with skip connection."""
        # First block: pooling -> fc1 -> normalization -> ReLU
        out = self.pool(x)
        out = self.act1(self.norm1(self.fc1(out)))

        # Second block: fc2 -> normalization -> sigmoid
        out = self.act2(self.norm2(self.fc2(out)))

        # Skip connection: multiply input with attention map
        return x * out



class SpatialAttention_1(nn.Module):
    """Spatial-attention module."""

    def __init__(self):
        """Initialize Spatial-attention module with kernel size argument."""
        super(SpatialAttention_1, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.sigmoid(self.conv2d(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))



class CBAM_LD_SAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention_1(c1)
        self.spatial_attention = SpatialAttention_1()
        # self.weight = Conv(c1, c1, 1, 1)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        # channel_attention_out = self.channel_attention(x) * x
        # weight = torch.sigmoid(self.weight(x))  # self.weight(x) 返回 Tensor
        channel_attention_out = self.channel_attention(x)
        return self.spatial_attention(channel_attention_out)



class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model


        # 注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        """
        T, V: (B, C, H, W)
        """
        T, V = x
        B, C, H, W = T.shape

        # reshape -> (B, H*W, C)
        T = T.flatten(2).transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)
        V = V.flatten(2).transpose(1, 2)

        out, _ = self.attn(query=T, key=V, value=V)
        # reshape 回 (B, C, H, W)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out

# class LCBHAM(nn.Module):
#     def __init__(self, channel):
#         super(LCBHAM, self).__init__()
#         self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)  # k=3, s=2
#         # self.conv = GhostConv(channel, channel, k=3, s=2)
#
#         self.bn = nn.GroupNorm(num_groups=32, num_channels=channel)
#         # 实验412
#         # self.act = h_swish()
#         self.act = h_sigmoid()
#
#         # self.lcam = AGCA(channel)
#         # self.lcam = LCAM(channel)
#         # self.ld_sam = LD_SAM()
#
#     def forward(self, x):
#         output = self.conv(x)
#         output = self.bn(output)
#         output = self.act(output)
#
#         # lcam_out = self.lcam(x)
#         output = output + x
#         # lcam_out = x * lcam_out
#
#         # ld_sam_out = self.ld_sam(lcam_out)
#         # output = ld_sam_out * lcam_out
#
#         return output
