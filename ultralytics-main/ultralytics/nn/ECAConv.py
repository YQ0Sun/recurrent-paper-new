import torch
import torch.nn as nn
import torch.nn.functional as F


# from MultiHeadAttention import *

# 论文DsP-YOLO: An anchor-free network with DsPAN for small object detection of multiscale defects

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

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


class ECAConv(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.act = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, dilation=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.act(output)
        y = self.avg_pool(output)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv1(y)
        y = y.transpose(1, 2).unsqueeze(-1)  # 这里 y 的通道数和 x 一致
        y = self.sigmoid(y)


        x = F.interpolate(x, size=output.shape[2:], mode='bilinear', align_corners=False)

        return x * y.expand_as(x)  # 可以直接乘


# 测试 LCBHAM 模块
if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 32, 32)  # 示例输入
    model = ECAConv(64, 3)
    output = model(input_tensor)
    print(output.shape)  # torch.Size([1, 64, 16, 16])

