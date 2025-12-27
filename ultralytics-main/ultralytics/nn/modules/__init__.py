# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, DWBottleneck3, DWBottleneck5, DWConvC2f, DWConvC2f_total,
                    DWConvC2f_5, DWConvC2f_total_5_3, DWConvC2f_5_CBAM, DWBottleneck5_CBAM, DWBottleneck5_CBAM_there,
                    DWConvC2f_5_CBAM_there, DWBottleneck5_afterConcatCBAM, DWConvC2f_5_CBAM_afterConcatCBAM,
                    DWBottleneck5_CBAM_new, DWBottleneck7, ASFF2, ASFF3, C2f_with_weight, DWBottleneck5_CBAM_new,
                    DWBottleneck7, AGCA, DWBottleneck5_CBAM_LD_SAM, CBAM_LD_SAM,)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention, DWConvWithAct, DWConvNoAct, Add, AGCA, CBAM_LD_SAM,
                   CBAM_LD_SAM, EMA, AddNL, CrossAttention)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)
from .NLBlockND import NLBlockND
from .MultiHeadAttention import *
from .Attention import *

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP'
           , 'DWBottleneck3', 'DWBottleneck5', 'DWConvWithAct', 'DWConvNoAct', 'DWConvC2f', 'DWConvC2f_total',
           'DWConvC2f_5', 'Add', 'DWConvC2f_total_5_3', 'DWConvC2f_5_CBAM', 'DWBottleneck5_CBAM',
           'DWBottleneck5_CBAM_there', 'DWConvC2f_5_CBAM_there', 'DWBottleneck5_afterConcatCBAM',
           'DWConvC2f_5_CBAM_afterConcatCBAM', 'C2f_with_AFPN', 'DWBottleneck5_CBAM_new', 'DWBottleneck7', 'AGCA',
           'DWBottleneck5_AGCA', 'CBAM_LD_SAM', 'EMA', 'NLBlockND', 'AddNL', 'ASFF2', 'ASFF3', 'C2f_with_weight',
           'DWBottleneck5_CBAM_new', 'DWBottleneck7', 'AGCA', 'DWBottleneck5_CBAM_LD_SAM', 'CBAM_LD_SAM', 'CrossAttention',
           'MultiHeadAttention')
