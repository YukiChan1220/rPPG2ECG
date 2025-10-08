import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleConvAE(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1,16, kernel_size=9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(16,32, kernel_size=9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(32,64, kernel_size=9, stride=2, padding=4), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(64,32, kernel_size=9, stride=2, padding=4, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32,16, kernel_size=9, stride=2, padding=4, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16,1,  kernel_size=9, stride=2, padding=4, output_padding=1),
        )
    def forward(self,x):
        z = self.enc(x)
        out = self.dec(z)
        return out

class UNet(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(1,16, kernel_size=15, stride=1, padding=7), nn.LeakyReLU(0.1), # L
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16,32, kernel_size=16, stride=2, padding=7), nn.LeakyReLU(0.1),    # L / 2
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(32,64, kernel_size=16, stride=2, padding=7), nn.LeakyReLU(0.1),    # L / 4
        )  
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64,128, kernel_size=31, stride=1, padding=15), nn.LeakyReLU(0.1),   # L / 4
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(192,64, kernel_size=16, stride=2, padding=7), nn.LeakyReLU(0.1),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(96,32, kernel_size=16, stride=2, padding=7), nn.LeakyReLU(0.1),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(48,16, kernel_size=15, stride=1, padding=7), nn.LeakyReLU(0.1),
        )
        self.output_layer = nn.Sequential(
            nn.ConvTranspose1d(16,1, kernel_size=15, stride=1, padding=7), nn.Tanh()
        )
    
    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bn = self.bottleneck(e3)



        d1 = self.dec1(torch.cat([bn, e3], dim=1))
        d2 = self.dec2(torch.cat([d1, e2], dim=1))
        d3  = self.dec3(torch.cat([d2, e1], dim=1))
        out = self.output_layer(d3)
        return out
    

class PPG2ECGps(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.enc1 = PoolingConvBNLeakyReLU(1,24, kernel_size=9, stride=1, padding=4, pool_size=2)
        self.enc2 = PoolingConvBNLeakyReLU(24,48, kernel_size=9, stride=1, padding=4, pool_size=2)
        self.enc3 = PoolingConvBNLeakyReLU(48,72, kernel_size=9, stride=1, padding=4, pool_size=2)
        self.enc4 = PoolingConvBNLeakyReLU(72,96, kernel_size=9, stride=1, padding=4, pool_size=2)
        self.enc5 = PoolingConvBNLeakyReLU(96,120, kernel_size=9, stride=1, padding=4, pool_size=2)
        self.enc6 = PoolingConvBNLeakyReLU(120,144, kernel_size=9, stride=1, padding=4, pool_size=2)
        self.enc7 = PoolingConvBNLeakyReLU(144,168, kernel_size=9, stride=1, padding=4, pool_size=2)
        self.bottleneck = PoolingConvBNLeakyReLU(168,192, kernel_size=9, stride=1, padding=4, pool_size=2)
        self.dec1 = ConvBNLeakyReLU(192,192, kernel_size=9, stride=1, padding=4)

    def forward(self,x):
        out = self.unet(x)
        return out
    
class PoolingConvBNLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool1d(pool_size)
    def forward(self,x):
        return self.pool(self.relu(self.bn(self.conv(x))))

class ConvBNLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))
