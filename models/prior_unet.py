import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_dim, out_channels)

    def forward(self, x, t_emb, cond_emb):
        x = self.relu(self.bn1(self.conv1(x)))
        # Add time and condition embeddings
        t_emb_proj = self.time_mlp(t_emb)[:, :, None]
        cond_emb_proj = self.cond_mlp(cond_emb)[:, :, None]
        x = x + t_emb_proj + cond_emb_proj
        
        x = self.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_dim, out_channels)

    def forward(self, x, skip, t_emb, cond_emb):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        
        t_emb_proj = self.time_mlp(t_emb)[:, :, None]
        cond_emb_proj = self.cond_mlp(cond_emb)[:, :, None]
        x = x + t_emb_proj + cond_emb_proj
        
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv1d(in_channels, in_channels, 1)
        self.key = nn.Conv1d(in_channels, in_channels, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width = x.size()
        q = self.query(x).view(batch_size, -1, width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, width)
        v = self.value(x).view(batch_size, -1, width)

        attention = torch.bmm(q, k)
        attention = torch.softmax(attention / (self.in_channels ** 0.5), dim=2)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width)
        
        return self.gamma * out + x # Add residual connection

class TinyUNet(nn.Module):
    def __init__(self, T, K, cond_dim, time_emb_dim=64, n_blocks=4, base_ch=64): # Increased defaults
        super().__init__()
        self.T = T
        self.K = K

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.init_conv = nn.Conv1d(K * 2, base_ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        ch = base_ch
        for i in range(n_blocks):
            self.down_blocks.append(DownBlock(ch, ch*2, time_emb_dim, cond_dim))
            ch *= 2

        self.bottleneck_attn = SelfAttentionBlock(ch) # ADDED ATTENTION

        self.up_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.up_blocks.append(UpBlock(ch, ch//2, time_emb_dim, cond_dim))
            ch //= 2

        self.final_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_ch, K * 2, 1)
        )

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = []
        for block in self.down_blocks:
            x, skip = block(x, t_emb, cond)
            skips.append(skip)

        x = self.bottleneck_attn(x) # APPLY ATTENTION

        for i, block in enumerate(self.up_blocks):
            x = block(x, skips[-(i+1)], t_emb, cond)

        return self.final_conv(x)
