import torch
import torch.nn as nn
import pytorch_lightning as pl
from .datasets import BTS_data
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SwinTransformer import SwinTransformer

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super(Model, self).__init__()
        patch_size = cfg['patch_size']
        num_classes = cfg['num_classes']
        embed_dim = cfg['embed_dim']
        in_chans = cfg['in_chans']
        depths = cfg['depths']
        num_heads = cfg['num_heads']
        self.network = SwinTransformer(patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                                       embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                       window_size=7, mlp_ratio=4)
    
    def forward(self, x):
        out = self.network(x)
    
    def training_step(self, batch, batch_idx):
        # 训练步骤
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)  # 记录训练损失
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证步骤
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)  # 记录验证损失
        return loss

    def test_step(self, batch, batch_idx):
        # 测试步骤
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)  # 记录测试损失
        return loss

    def configure_optimizers(self):
        # 配置优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer