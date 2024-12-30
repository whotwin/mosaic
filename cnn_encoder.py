import torch
import torch.nn as nn

class CNN_Encoder(nn.Module):
    def __init__(self, in_channels=1, embed_dims=96, kernel_size=3):
        super(CNN_Encoder, self).__init__()
        self.dilation_1 = nn.Conv2d(in_channels=in_channels, out_channels=embed_dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                            dilation=1)
        self.dilation_2 = nn.Conv2d(in_channels=in_channels, out_channels=embed_dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2 + 1,
                            dilation = 2)
        self.conv = nn.Conv2d(in_channels=2 * embed_dims, out_channels=embed_dims, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(embed_dims)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x):
        identity = x
        out_1 = self.dilation_1(x)
        out_2 = self.dilation_2(x)
        out = self.conv(torch.concatenate([out_1, out_2], dim=1))
        out = self.pool(self.relu(self.bn(out)) + identity)
        return out
    
if __name__ == '__main__':
    x = torch.randn((1, 1, 224, 224)).cuda()
    model = CNN_Encoder().cuda()
    out = model(x)
    print('')