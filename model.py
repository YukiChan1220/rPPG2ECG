"""
CardioGAN (PyTorch) - model definitions
Generator: 1D Attention U-Net (encoder-decoder with attention gates)
Discriminators:
 - TimeDiscriminator: 1D conv net on raw signal
 - SpecDiscriminator: 2D conv net on magnitude spectrogram (freq x time)
Notes:
 - Input shapes: (batch, 1, L)
 - Spectrogram uses torch.stft -> magnitude -> log(1+mag) normalization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Attention gating block
# -------------------------
class AttentionGate1D(nn.Module):
    """
    Self-gated soft attention unit for 1D features (adapted from Attention U-Net).
    x: skip feature (from encoder) - shape (B, C_x, T)
    g: gating signal (from deeper layer) - shape (B, C_g, T_g)
    We transform both to an intermediate channel and compute attention coefficients.
    """
    def __init__(self, in_channels_x, in_channels_g, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, in_channels_x // 2)
        self.W_x = nn.Conv1d(in_channels_x, inter_channels, kernel_size=1, bias=True)
        self.W_g = nn.Conv1d(in_channels_g, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Conv1d(inter_channels, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # up/downsample g to x's time dimension if necessary
        if g.shape[-1] != x.shape[-1]:
            g = F.interpolate(g, size=x.shape[-1], mode='linear', align_corners=False)
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        out = self.relu(x1 + g1)
        psi = self.sigmoid(self.psi(out))
        return x * psi  # element-wise multiplication (broadcasted along channel dim)


# -------------------------
# Basic conv blocks
# -------------------------
def conv_block(in_ch, out_ch, kernel_size=7, stride=1, padding=3, norm=True, activation='leaky'):
    layers = []
    layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding))
    if norm:
        layers.append(nn.LayerNorm([out_ch,]))  # we'll reshape in forward to satisfy layernorm
    if activation == 'leaky':
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def deconv_block(in_ch, out_ch, kernel_size=7, stride=1, padding=3, norm=True, activation='relu'):
    # transposed conv for upsampling
    layers = []
    layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding))
    if norm:
        layers.append(nn.LayerNorm([out_ch,]))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'leaky':
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


# Helper to apply LayerNorm after conv (since LayerNorm expects (N, C, L) we use Channel-first support)
class LayerNorm1dWrapper(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        # LayerNorm over (C, L) per sample: use normalized_shape=(channels, )
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # x shape: (B, C, L) -> permute to (B, L, C) apply LN over last dim then back
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        return x


# -------------------------
# Attention U-Net style 1D Generator
# -------------------------
class AttentionUNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, depth=6, kernel_size=16):
        super().__init__()
        ks = kernel_size  # Fixed kernel size = 16
        pad = ks // 2
        self.depth = depth

        # Encoder with fixed kernel size=16, stride=2
        # Filter progression: 64 -> 128 -> 256 -> 512 -> 512 -> 512
        encs = []
        ins = in_channels
        # Define filter progression: [64, 128, 256, 512, 512, 512]
        filter_progression = [64, 128, 256, 512, 512, 512]
        
        for d in range(depth):
            outs = filter_progression[d]
            # First encoder layer doesn't use normalization (as in paper)
            use_norm = False if d == 0 else True
            encs.append(nn.Sequential(
                nn.Conv1d(ins, outs, kernel_size=ks, stride=2, padding=pad),
                LayerNorm1dWrapper(outs) if use_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            ins = outs
        self.encs = nn.ModuleList(encs)

        # Bottleneck with kernel_size=16
        self.bottleneck = nn.Sequential(
            nn.Conv1d(ins, ins * 2, kernel_size=ks, stride=1, padding=pad),
            LayerNorm1dWrapper(ins * 2),
            nn.ReLU(inplace=True)
        )

        # Decoder with kernel_size=16, stride=2
        # After bottleneck, we have ins*2 channels (1024 since ins=512)
        # Decoder mirrors encoder: skip connections from [512, 512, 512, 256, 128, 64]
        decs = []
        att_gates = []
        
        # Track decoder input/output channels
        # First decoder layer takes bottleneck output (ins*2) + skip connection
        dec_in = ins * 2  # bottleneck output channels (1024)
        
        # Decoder filter progression (mirror of encoder): [512, 512, 256, 128, 64, 64]
        decoder_progression = [512, 512, 256, 128, 64, 64]
        
        for d in range(depth):
            # Corresponding skip from encoder (in reverse order)
            # Encoder: [64, 128, 256, 512, 512, 512]
            # Skips:   [512, 512, 512, 256, 128, 64] (reversed)
            skip_ch = filter_progression[depth - 1 - d]
            
            # Decoder output channels
            dec_out = decoder_progression[d]
            
            # Input to decoder layer = current features + gated skip
            dec_layer_in = dec_in + skip_ch
            
            decs.append(nn.Sequential(
                nn.ConvTranspose1d(dec_layer_in, dec_out, kernel_size=ks, stride=2, padding=pad, output_padding=1),
                LayerNorm1dWrapper(dec_out),
                nn.ReLU(inplace=True)
            ))
            
            # Attention gate for the corresponding skip
            att_gates.append(AttentionGate1D(skip_ch, dec_in))
            
            # Next decoder input is current decoder output
            dec_in = dec_out
        
        self.decs = nn.ModuleList(decs)
        self.att_gates = nn.ModuleList(att_gates)

        # Final conv -> tanh (kernel_size=16)
        # dec_in now contains the output channels of the last decoder layer
        self.final = nn.Sequential(
            nn.Conv1d(dec_in, out_channels, kernel_size=ks, padding=pad),
            nn.Tanh()
        )

    def forward(self, x):
        input_size = x.shape[-1]  # Store original input size
        skips = []
        out = x
        # encode
        for enc in self.encs:
            out = enc(out)
            skips.append(out)
        # bottleneck
        out = self.bottleneck(out)
        # decode with attention gated skips (reverse order)
        for idx, dec in enumerate(self.decs):
            # use last skip
            skip = skips[-1 - idx]
            # gating: dec expects the gating signal 'out' (current decoder input). Use gate to modulate skip
            gated = self.att_gates[idx](skip, out)
            # concatenate gated skip with current out along channel dim
            # ensure dec input channels match: many dec layers created expecting specific shapes
            # we will upsample out to skip time length if mismatch
            if out.shape[-1] != gated.shape[-1]:
                out = F.interpolate(out, size=gated.shape[-1], mode='linear', align_corners=False)
            cat = torch.cat([out, gated], dim=1)
            out = dec(cat)
        
        # Apply final convolution
        out = self.final(out)
        
        # Ensure output matches input size exactly
        if out.shape[-1] != input_size:
            out = F.interpolate(out, size=input_size, mode='linear', align_corners=False)
        
        return out


# -------------------------
# Time-domain discriminator (1D)
# -------------------------
class TimeDiscriminator(nn.Module):
    """
    1D Patch-style discriminator: input (B, 1, L) -> output (B, 1, L')
    Uses progressively increasing filters with kernel_size=16, stride=2.
    Filter progression: 64 -> 128 -> 256 -> 512
    """
    def __init__(self, in_channels=1, base_filters=64, n_layers=4, kernel_size=16):
        super().__init__()
        ks = kernel_size  # Fixed kernel size = 16
        pad = ks // 2

        layers = []
        nf = base_filters
        # First layer: no normalization
        layers.append(nn.Conv1d(in_channels, nf, kernel_size=ks, stride=2, padding=pad))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Middle layers with normalization
        for i in range(1, n_layers):
            prev = nf
            nf = min(nf * 2, 512)
            layers.append(nn.Conv1d(prev, nf, kernel_size=ks, stride=2, padding=pad))
            layers.append(LayerNorm1dWrapper(nf))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final layer
        layers.append(nn.Conv1d(nf, 1, kernel_size=ks, stride=1, padding=pad))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# -------------------------
# Spectrogram discriminator (2D)
# -------------------------
class SpecDiscriminator(nn.Module):
    """
    2D conv discriminator for spectrogram images (B, 1, F, T).
    Produces patch output (B, 1, f', t')
    Uses kernel_size=(7, 7), stride=2 for 2D convolutions.
    Filter progression: 64 -> 128 -> 256 -> 512
    """
    def __init__(self, in_channels=1, base_filters=64, n_layers=4, kernel_size=(7, 7)):
        super().__init__()
        ks_h, ks_w = kernel_size
        pad_h, pad_w = ks_h // 2, ks_w // 2

        layers = []
        nf = base_filters
        # First layer: no normalization
        layers.append(nn.Conv2d(in_channels, nf, kernel_size=(ks_h, ks_w), stride=(2, 2), padding=(pad_h, pad_w)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Middle layers with normalization
        for i in range(1, n_layers):
            prev = nf
            nf = min(nf * 2, 512)
            layers.append(nn.Conv2d(prev, nf, kernel_size=(ks_h, ks_w), stride=(2, 2), padding=(pad_h, pad_w)))
            layers.append(nn.InstanceNorm2d(nf))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final layer
        layers.append(nn.Conv2d(nf, 1, kernel_size=(ks_h, ks_w), stride=1, padding=(pad_h, pad_w)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# -------------------------
# Utility: spectrogram function
# -------------------------
def magnitude_spectrogram(x, n_fft=256, hop_length=64, win_length=256, window=torch.hann_window):
    """
    x: (B, 1, L)
    returns log(1 + |STFT|) shape (B, 1, F, T)
    """
    B = x.shape[0]
    device = x.device
    x = x.squeeze(1)  # (B, L)
    win = window(win_length).to(device)
    # torch.stft expects (B, L)
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=win, return_complex=True, normalized=False, center=True, pad_mode='reflect')
    mag = torch.abs(stft)  # (B, F, T)
    mag = torch.log1p(mag)  # log scaling
    mag = mag.unsqueeze(1)  # (B, 1, F, T)
    return mag


# -------------------------
# Instantiate helper factory
# -------------------------
def build_generators(device='cpu'):
    G = AttentionUNet1D(in_channels=1, out_channels=1).to(device)
    F = AttentionUNet1D(in_channels=1, out_channels=1).to(device)
    return G, F


def build_discriminators(device='cpu'):
    # for ECG domain
    D_time_E = TimeDiscriminator(in_channels=1).to(device)
    D_spec_E = SpecDiscriminator(in_channels=1).to(device)
    # for PPG domain
    D_time_P = TimeDiscriminator(in_channels=1).to(device)
    D_spec_P = SpecDiscriminator(in_channels=1).to(device)
    return D_time_E, D_spec_E, D_time_P, D_spec_P
