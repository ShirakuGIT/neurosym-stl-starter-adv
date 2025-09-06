# sampling_prior.py
import torch

def ddpm_sample_knots(net, cond, K, Tdiff=1000, device="cuda"):
    """
    cond: (cond_dim,) on device
    returns knots: (K,2)
    """
    net.eval()
    x = torch.randn(1, 2*K, device=device)
    cond = cond.view(1, -1).to(device)
    betas = torch.linspace(1e-4, 0.02, Tdiff, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    for t in reversed(range(Tdiff)):
        t_f = torch.full((1,), float(t) / (Tdiff-1), device=device)
        eps_hat = net(x, t_f, cond)
        a_t = alphas[t]; ab_t = alpha_bar[t]; b_t = betas[t]
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)
        x = (1.0/torch.sqrt(a_t)) * (x - (b_t/torch.sqrt(1-ab_t)) * eps_hat) + torch.sqrt(b_t) * z
    knots = x.view(1, 2*K).reshape(1, K, 2)
    return knots[0]
