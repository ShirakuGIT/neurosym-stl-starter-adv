# prior_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.dtype not in (torch.float32, torch.float64):
        t = t.float()
    device = t.device
    half = dim // 2
    # frequencies 1 ... 10k on a log scale
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=device))
    args = t.unsqueeze(1) / freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class TinyUNet(nn.Module):
    """
    Checkpoint-compatible MLP denoiser:
      - cond_mlp.0/2/4.*
      - time_mlp.0/2/4.*
      - core.0/2/4/6.*

    Pass `time_in_dim=1` for raw scalar t (matches your checkpoint);
    use `time_in_dim>1` to feed sinusoidal embeddings of that size.
    """
    def __init__(self, in_dim: int, cond_dim: int, hidden: int = 256, time_in_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.hidden = hidden
        self.time_in_dim = time_in_dim

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.core = nn.Sequential(
            nn.Linear(in_dim + hidden + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Build the time feature to match time_in_dim
        if self.time_in_dim == 1:
            t_feat = t.view(-1, 1).float()
        else:
            t_feat = sinusoidal_time_embed(t, self.time_in_dim)

        c = self.cond_mlp(cond)
        tau = self.time_mlp(t_feat)
        h = torch.cat([x, c, tau], dim=1)
        return self.core(h)
