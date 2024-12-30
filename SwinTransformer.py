import math
import torch
import torch.nn as nn
from typing import Sequence
import pytorch_lightning as pl
import torch.nn.functional as F
from utils import ShiftWindowMSA
from cnn_encoder import CNN_Encoder
from mmengine.model import BaseModule
from base_backbone import BaseBackbone
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmcv.cnn.bricks.transformer import PatchMerging, FFN#PatchEmbed, FFN

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dims, kernel_size):
        super().__init__()
        self.embed_dims = embed_dims
        self.conv = nn.Conv2d(in_channels, embed_dims, kernel_size, kernel_size, padding=kernel_size // 2 - 1)
        self.norm = nn.LayerNorm(embed_dims)
        self.patch_size = kernel_size
    
    def forward(self, x):
        B, C, H, W = x.size()
        return self.norm(self.conv(x).view(B, self.embed_dims, -1).transpose(1, 2)), (H // self.patch_size, W // self.patch_size)

class DropPath(nn.Module):
    """ Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class SwinTransformerBlock(BaseModule):
    """ Swin Transformer Block """
    def __init__(self, embed_dims, num_heads, window_size=7, shift_size=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 ffn_cfgs=dict(), ffn_ratio=4., attn_cfgs=dict()):
        super().__init__()
        self.dim = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(embed_dims)

        _ffn_cfgs = {
                    'embed_dims': embed_dims,
                    'feedforward_channels': int(embed_dims * ffn_ratio),
                    'num_fcs': 2,
                    'ffn_drop': 0,
                    'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
                    'act_cfg': dict(type='GELU'),
                    **ffn_cfgs
                }
        self.ffn = FFN(**_ffn_cfgs)
        _attn_cfgs = {
            'embed_dims': embed_dims,
            'num_heads': num_heads,
            'shift_size': shift_size,
            'window_size': window_size,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'pad_small_map': False,
            **attn_cfgs
        }
        self.attn = ShiftWindowMSA(**_attn_cfgs)
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dims)

    def forward(self, x, hw_shape):
        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x
        x = _inner_forward(x)

        return x


class BasicLayer(BaseModule):
    """ Basic Layer """
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=PatchMerging):
        super().__init__()
        if not isinstance(drop_path, Sequence):
            drop_path = [drop_path] * depth
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList()
        for i in range(depth):
            _block_cfgs = {
                'embed_dims': dim,
                'num_heads': num_heads,
                'window_size': window_size,
                'shift_size': self.shift_size if i % 2 != 0 else 0,
                'ffn_ratio': mlp_ratio,
                'drop_path': drop_path[i],

            }
            block = SwinTransformerBlock(**_block_cfgs)
            self.blocks.append(block)
        if downsample is not None:
            self.downsample = downsample(in_channels=dim, out_channels=dim * 2, norm_cfg=dict(type='LN'))
        else:
            self.downsample = None

    def forward(self, x, in_shape):
        for blk in self.blocks:
            x = blk(x, in_shape)
        if self.downsample is not None:
            x, in_shape = self.downsample(x, in_shape)
            #H, W = (H + 1) // 2, (W + 1) // 2
        return x, in_shape
    
    @property
    def out_channels(self):
        if self.downsample:
            return self.downsample.out_channels
        else:
            return self.dim

class SwinTransformer(BaseBackbone):
    """ Swin Transformer """
    def __init__(self, patch_size=2, in_chans=3, num_classes=4, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, frozen_stages=-1, out_indices=[0, 1, 2, 3], norm_eval=False):
        super().__init__()
        self.pre_conv = CNN_Encoder(in_channels=embed_dim, embed_dims=2 * embed_dim, kernel_size=3)
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        embed_dims = [embed_dim]
        self.patch_embed = PatchEmbed(kernel_size=patch_size, in_channels=in_chans, embed_dims=embed_dim)#)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        for i_layer, (depth, num_head) in enumerate(zip(depths, num_heads)):
            stages = BasicLayer(dim=embed_dims[-1],
                                depth=depth,
                                num_heads=num_head,
                                window_size=window_size, mlp_ratio=mlp_ratio,
                                drop_path=dpr[:depth],
                                norm_layer=norm_layer, downsample=PatchMerging if (i_layer < self.num_layers) else None)#原来要减1
            self.stages.append(stages)
            dpr = dpr[depth:]
            embed_dims.append(stages.out_channels)
        self.norm = norm_layer(self.num_features)
        self.norm_list = nn.ModuleList()
        self.num_features = embed_dims[1:]
        for i in out_indices:
            norm = norm_layer(self.num_features[i])
            self.norm_list.append(norm)

        #self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.size(0)
        x, hw_shape = self.patch_embed(x)
        x = self.pos_drop(x)
        x = x.view(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2)
        outs = []
        for i, layer in enumerate(self.stages):
            if i == 0:
                x = self.pre_conv(x)
                hw_shape = tuple(x.shape[-2:])
                outs.append(x)
                x = x.view(B, -1, hw_shape[0] * hw_shape[1]).transpose(1, 2)
                continue
            x, hw_shape = layer(x, hw_shape)
            if i in self.out_indices:
                out = self.norm_list[i](x)
                out = out.view(-1, *hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            '''if layer.downsample is not None:
                x, hw_shape = layer.downsample(x, hw_shape)'''
        #x = self.norm(x)
        #x = self.avgpool(x.transpose(1, 2))
        #x = torch.flatten(x, 1)
        #x = self.head(x)
        return tuple(outs)
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for i in self.out_indices:
            if i <= self.frozen_stages:
                for param in getattr(self, f'norm{i}').parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

if __name__ == '__main__':
    data = torch.randn((1, 1, 224, 224)).cuda()
    backbone = SwinTransformer(in_chans=1).cuda()
    out = backbone(data)
    drop_path_rate = 0.1
    depths = [2, 2, 6, 2]
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
    for i_layer in range(len(depths)):
        depth = depths[i_layer]
        print(dpr[:depth])
        dpr = dpr[depth:]
        print(dpr, len(dpr))
    #print(dpr, len(dpr))