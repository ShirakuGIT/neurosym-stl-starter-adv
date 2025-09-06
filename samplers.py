import torch
import torch.nn.functional as F
from pathlib import Path

def sample_smooth_controls(B, T, device, max_speed=1.6, smooth_k=15, noise_scale=0.4):
    """
    Gaussian noise -> depthwise 1D conv smoothing per channel.
    Gentler defaults: lower noise, longer smoothing kernel, and lower max_speed.
    """
    u = torch.randn(B, T, 2, device=device, dtype=torch.float32) * noise_scale
    # conv1d expects (N, C, L); do depthwise (groups=2) for 2 channels
    u_ncL = u.permute(0, 2, 1).contiguous()  # (B,2,T)
    k = torch.ones(2, 1, smooth_k, device=device, dtype=u.dtype) / smooth_k  # (out=2, in=1, k)
    u_sm = F.conv1d(u_ncL, k, padding=smooth_k // 2, groups=2)  # (B,2,T)
    u = u_sm.permute(0, 2, 1).contiguous()  # (B,T,2)
    # clip speed
    speed = torch.norm(u, dim=-1, keepdim=True) + 1e-6
    u = u * torch.clamp(max_speed / speed, max=1.0)
    return u

# Optional DDPM prior (tiny). If no checkpoint, fall back to smoothing sampler.
def sample_with_ddpm_or_fallback(B, T, device, max_speed=1.6):
    try:
        from infer import PriorSampler
        ckpt = Path('outputs/checkpoints/prior.pt')
        if ckpt.exists():
            sampler = PriorSampler(str(ckpt), device)
            return sampler.sample(B, T)
    except Exception:
        pass
    return sample_smooth_controls(B, T, device=device, max_speed=max_speed)
