import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import nn, optim

# 1. 定义模型
class LitAutoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        reconstructed = self(x)
        loss = nn.MSELoss()(reconstructed, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

# 2. 数据加载
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 3. 配置训练器
trainer = Trainer(max_epochs=10)

# 4. 训练模型
model = LitAutoencoder()
trainer.fit(model, train_loader, val_loader)

# 5. 验证和测试
# PyTorch Lightning会自动在训练时处理验证步骤，并在训练结束后处理测试步骤。