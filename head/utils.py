# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
import warnings
import torch.nn as nn
from torch import Tensor
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmengine.model import BaseModule
#from mmseg.utils import OptConfigType


class BasicBlock(BaseModule):
    """Basic block from `ResNet <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Input channels.
        channels (int): Output channels.
        stride (int): Stride of the first block. Default: 1.
        downsample (nn.Module, optional): Downsample operation on identity.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU', inplace=True).
        act_cfg_out (dict, optional): Config dict for activation layer at the
            last of the block. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    expansion = 1

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: nn.Module = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 act_cfg_out: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.downsample = downsample
        if act_cfg_out:
            self.act = MODELS.build(act_cfg_out)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual

        if hasattr(self, 'act'):
            out = self.act(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block from `ResNet <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Input channels.
        channels (int): Output channels.
        stride (int): Stride of the first block. Default: 1.
        downsample (nn.Module, optional): Downsample operation on identity.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU', inplace=True).
        act_cfg_out (dict, optional): Config dict for activation layer at
            the last of the block. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    expansion = 2

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 act_cfg_out: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(
            channels,
            channels,
            3,
            stride,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            channels,
            channels * self.expansion,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        if act_cfg_out:
            self.act = MODELS.build(act_cfg_out)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual

        if hasattr(self, 'act'):
            out = self.act(out)

        return out

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super().__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)
