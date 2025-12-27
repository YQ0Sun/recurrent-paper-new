import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import SiLU


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


# LCAM 模块
class LCAM(nn.Module):
    def __init__(self, channel):
        super(LCAM, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.conv1_max = nn.Conv2d(channel, channel // 4, kernel_size=1, stride=1, padding=0)
        # self.conv1_avg = nn.Conv2d(channel, channel // 4, kernel_size=1, stride=1, padding=0)
        self.conv1_max = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv1_avg = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(channel // 4, channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        max_out = self.conv1_max(self.max_pool(x))
        max_out = self.relu(max_out)

        avg_out = self.conv1_avg(self.avg_pool(x))
        avg_out = self.relu(avg_out)

        out = max_out + avg_out
        out = torch.sigmoid(out)  # 这里直接使用 out 而不进行进一步的卷积

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


# LCBHAM 模块
class LCBHAM(nn.Module):
    def __init__(self, channel):
        super(LCBHAM, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)  # k=3, s=2
        self.bn = nn.BatchNorm2d(channel)
        self.act = h_swish()

        self.lcam = LCAM(channel)
        self.ld_sam = LD_SAM()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        lcam_out = self.lcam(x)
        lcam_out = x * lcam_out

        ld_sam_out = self.ld_sam(lcam_out)
        output = ld_sam_out * lcam_out

        return output

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

        self.norm = LayerNorm(n_feats, data_format='channels_first')
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

class CBAM_MLKA_LD_SAM(nn.Module):
    def __init__(self, channel):
        super(CBAM_MLKA_LD_SAM, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)  # k=3, s=2
        self.bn = nn.BatchNorm2d(channel)
        # self.act = h_swish()
        self.act = SiLU()

        self.lcam = MLKA_Ablation(channel)
        self.ld_sam = LD_SAM()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        lcam_out = self.lcam(x)
        lcam_out = x * lcam_out

        ld_sam_out = self.ld_sam(lcam_out)
        output = ld_sam_out * lcam_out

        return output

class CBAM_MLKA(nn.Module):
    def __init__(self, channel):
        super(CBAM_MLKA, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)  # k=3, s=2
        self.bn = nn.BatchNorm2d(channel)
        # self.act = h_swish()
        self.act = SiLU()

        self.lcam = MLKA_Ablation(channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        lcam_out = self.lcam(x)
        lcam_out = x * lcam_out

        return lcam_out

class CBAM_LCAM_MLKA(nn.Module):
    def __init__(self, channel):
        super(CBAM_LCAM_MLKA, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)  # k=3, s=2
        self.bn = nn.BatchNorm2d(channel)
        # self.act = h_swish()
        self.act = SiLU()

        self.lcam = LCAM(channel)
        self.ld_sam = MLKA_Ablation(channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # self.act = h_swish()
        self.act = SiLU()

        lcam_out = self.lcam(x)
        lcam_out = x * lcam_out

        ld_sam_out = self.ld_sam(lcam_out)
        output = ld_sam_out * lcam_out

        return output


# 测试 CBAM_MLKA_LD_SAM 模块
if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 32, 32)  # 示例输入
    model = CBAM_LCAM_MLKA(64)
    output = model(input_tensor)
    print(output.shape)  # torch.Size([1, 64, 16, 16])

    # input_tensor = torch.randn(1, 64, 32, 32)  # Example input
    # model = Conv(64, 64)  # Assuming 64 input channels and 128 output channels
    # output = model(input_tensor)
    # print(output.shape) # torch.Size([1, 64, 32, 32])

