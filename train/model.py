import torch.nn as nn

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
