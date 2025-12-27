import torch
import torch.nn as nn
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


# LCAM 模块
class LCAM(nn.Module):
    def __init__(self, channel):
        super(LCAM, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.conv1_max = nn.Conv2d(channel, channel // 4, kernel_size=1, stride=1, padding=0)
        # self.conv1_avg = nn.Conv2d(channel, channel // 4, kernel_size=1, stride=1, padding=0)
        self.conv1_max = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1_avg = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=True)
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
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([max_out, mean_out], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

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

class AGCA(nn.Module):
    def __init__(self, in_channel, ratio=1):
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
        nn.init.constant_(self.A2, 1e-6)
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

# LCBHAM 模块
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

class LCBHAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
        self.bn = nn.GroupNorm(num_groups=32, num_channels=channel)
        self.act = nn.ReLU()

        # shortcut 卷积，用于匹配尺寸
        self.shortcut = nn.Conv2d(channel, channel, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.act(output)

        x_res = self.shortcut(x)  # 让 x 和 output 尺寸一致
        output = output + x_res
        return output


# 测试 LCBHAM 模块
if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 32, 32)  # 示例输入
    model = LCBHAM(64)
    output = model(input_tensor)
    print(output.shape)  # torch.Size([1, 64, 16, 16])

