import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ShiftWindowMSA
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import PatchMerging, PatchEmbed, FFN
    
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
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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


class BasicLayer(nn.Module):
    """ Basic Layer """
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            _block_cfgs = {
                'embed_dims': dim,
                'num_heads': num_heads,
                'window_size': window_size,
                'shift_size': self.shift_size if i % 2 != 0 else 0,
                'ffn_ratio': mlp_ratio,
                'drop_path': drop_path,

            }
            block = SwinTransformerBlock(**_block_cfgs)
            self.blocks.append(block)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # Create attention mask
        B, L, C = x.shape
        img_mask = torch.zeros((B, H, W), dtype=torch.bool).to(x.device)  # Init mask
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, cnt:cnt + self.window_size, cnt:cnt + self.window_size] = True
                cnt += self.window_size

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, 0)
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
        return x, H, W

class SwinTransformer(nn.Module):
    """ Swin Transformer """
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, frozen_stages=-1):
        super().__init__()
        self.frozen_stages = frozen_stages
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer, downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layers)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            m = self.stages[i]
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