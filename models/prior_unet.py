# models/prior_unet.py
import torch, torch.nn as nn

def mlp(d_in, d_hidden, d_out, n=2):
    layers = [nn.Linear(d_in, d_hidden), nn.SiLU()]
    for _ in range(n-1):
        layers += [nn.Linear(d_hidden, d_hidden), nn.SiLU()]
    layers += [nn.Linear(d_hidden, d_out)]
    return nn.Sequential(*layers)

class KnotPrior(nn.Module):
    def __init__(self, K, cond_dim, hidden=256):
        super().__init__()
        self.K = K
        self.in_dim = 2*K
        self.cond_mlp = mlp(cond_dim, hidden, hidden)
        self.time_mlp = mlp(1, hidden, hidden)
        self.core = mlp(self.in_dim + hidden + hidden, hidden, self.in_dim, n=3)

    def forward(self, x_noisy, t_scalar, cond):
        """
        x_noisy: (B, 2K)
        t_scalar: (B,) or (B,1) in [0,1]
        cond: (B, cond_dim)
        returns: predicted noise eps_hat with shape (B, 2K)
        """
        if t_scalar.dim() == 1:
            t_scalar = t_scalar.unsqueeze(-1)
        h = torch.cat([x_noisy, self.cond_mlp(cond), self.time_mlp(t_scalar)], dim=-1)
        return self.core(h)
