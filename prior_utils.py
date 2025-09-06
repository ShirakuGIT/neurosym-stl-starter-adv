# prior_utils.py
import torch

@torch.no_grad()
def encode_env(x0, obs_xy, obs_r, goal_xy, goal_r, max_obs=8):
    """
    Same layout as generate_dataset.encode_env:
    [x0(2), goal(2), goal_r(1), obs_xy(2*max_obs), obs_r(max_obs), n_obs(1)]
    Returns (D,) tensor on x0.device
    """
    dev = x0.device
    n = obs_xy.shape[0]
    n_use = min(n, max_obs)
    if n > 0:
        d = torch.linalg.norm(obs_xy - x0, dim=-1)
        idx = torch.argsort(d)[:n_use]
    else:
        idx = torch.tensor([], dtype=torch.long, device=dev)

    pad_xy = torch.zeros(max_obs, 2, device=dev)
    pad_r  = torch.zeros(max_obs, device=dev)
    if n_use > 0:
        pad_xy[:n_use] = obs_xy[idx]
        pad_r[:n_use]  = obs_r[idx]

    parts = [
        x0,                             # (2,)
        goal_xy,                        # (2,)
        goal_r.view(1),                 # (1,)
        pad_xy.reshape(-1),             # (2*max_obs,)
        pad_r,                          # (max_obs,)
        torch.tensor([float(n)], device=dev),  # (1,)
    ]
    return torch.cat(parts, dim=0)

def _default_betas(T):
    # simple linear schedule; matches many minimal DDPM trainings
    return torch.linspace(1e-4, 2e-2, T)

@torch.no_grad()
def sample_knots_ddpm(model, cond_vec, K, steps=None, betas=None, device="cuda"):
    """
    DDPM ancestral sampling (epsilon prediction).
    model(x_t, t, cond) -> eps_hat
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * eps_hat) + sigma_t * z
    where sigma_t = sqrt(beta_t)
    """
    model.eval()
    Tdiff = steps if (steps is not None) else getattr(model, "Tdiff", 1000)
    betas = betas if betas is not None else _default_betas(Tdiff).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    x = torch.randn(1, 2*K, device=device)
    cond = cond_vec.view(1, -1).to(device)

    for t in reversed(range(Tdiff)):
        t_idx = torch.full((1,), t, device=device, dtype=torch.long)
        eps_hat = model(x, t_idx, cond)  # (1, 2K)

        a_t = alphas[t]
        ab_t = alpha_bars[t]
        # mean of p(x_{t-1} | x_t)
        mean = (x - (1 - a_t).sqrt() / (1 - ab_t).sqrt() * eps_hat) / a_t.sqrt()
        if t > 0:
            z = torch.randn_like(x)
            x = mean + betas[t].sqrt() * z
        else:
            x = mean

    return x.view(K, 2)
