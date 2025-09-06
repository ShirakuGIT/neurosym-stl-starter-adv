import torch, torch.nn as nn
class TinyUNet(nn.Module):
    def __init__(self, d=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d+1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d)
        )
    def forward(self, x, t):
        B,T,D = x.shape
        tfeat = torch.full((B,T,1), float(t), device=x.device)
        inp = torch.cat([x, tfeat], dim=-1)
        return self.net(inp)
class PriorSampler:
    def __init__(self, ckpt_path, device='cpu'):
        self.device = torch.device(device)
        self.model = TinyUNet().to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
    @torch.no_grad()
    def sample(self, B, T, steps=20, sigma=0.7):
        x = torch.randn(B,T,2, device=self.device)*sigma
        for s in range(steps,0,-1):
            t = s/steps
            eps = self.model(x, t)
            x = x - 0.5*eps
        v = torch.norm(x, dim=-1, keepdim=True)+1e-6
        x = x * torch.clamp(2.0/v, max=1.0)
        return x
