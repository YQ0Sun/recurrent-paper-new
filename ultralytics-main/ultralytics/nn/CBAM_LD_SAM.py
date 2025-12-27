
import math

import numpy as np
import torch
import torch.nn as nn

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

    def __init__(self):
        """Initialize Spatial-attention module with kernel size argument."""
        super(SpatialAttention, self).__init__()
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
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention()
        self.weight = Conv(self.c, self.c, 1, 1)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        # channel_attention_out = self.channel_attention(x) * x
        channel_attention_out = self.channel_attention(x) * torch.sigmoid(self.weight)
        return self.spatial_attention(channel_attention_out) * torch.sigmoid(self.weight)
# 测试 LCBHAM 模块
if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 32, 32)  # 示例输入
    model = CBAM_LD_SAM(64)
    output = model(input_tensor)
    print(output.shape)  # torch.Size([1, 64, 16, 16])