import torch

@torch.no_grad()
def one_step_repair(x0, u_seq, obs_xy, obs_r, dt=0.1, alpha=1.2, d_safe=0.25):
    """
    Stronger repulsion: bigger alpha and radius. Single cheap pass.
    """
    device = u_seq.device
    B, T, _ = u_seq.shape

    # simulate once
    x = x0.clone()
    traj = []
    for t in range(T):
        x = x + u_seq[:, t, :] * dt
        traj.append(x.clone())
    xy = torch.stack(traj, dim=1)  # (B,T,2)

    if obs_xy.numel() == 0:
        return u_seq

    M = obs_xy.shape[0]
    xy_bt2 = xy.unsqueeze(2)                  # (B,T,1,2)
    obs_m2 = obs_xy.view(1, 1, M, 2)          # (1,1,M,2)
    r_m1   = obs_r.view(1, 1, M, 1)           # (1,1,M,1)
    vec = xy_bt2 - obs_m2                     # (B,T,M,2)
    dist = torch.linalg.norm(vec + 1e-9, dim=-1, keepdim=True)  # (B,T,M,1)
    sd = dist - r_m1                          # (B,T,M,1)

    sd_min, idx = sd.squeeze(-1).min(dim=2)   # (B,T)
    violating = sd_min < d_safe               # (B,T)

    chosen_vec = torch.gather(vec, 2, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 2)).squeeze(2)  # (B,T,2)
    chosen_dist = torch.gather(dist.squeeze(-1), 2, idx.unsqueeze(-1)).squeeze(2)                        # (B,T)

    dir_unit = chosen_vec / (chosen_dist.unsqueeze(-1) + 1e-6)
    mag = torch.clamp(d_safe - sd_min, min=0.0, max=d_safe) * alpha  # (B,T)

    delta_u = dir_unit * mag.unsqueeze(-1)   # (B,T,2)
    mask = violating.unsqueeze(-1).float()
    u_repaired = u_seq + mask * delta_u
    return u_repaired
