# optimize_stl_annealed_batched.py
import torch
from stl_soft import F_eventually_soft, G_always_soft
from ap_defs import ap_goal_inside, ap_outside_obstacle

def simulate_batch(x0, u, dt):
    # x0: (B,2), u: (B,T,2)
    inc = dt * torch.cumsum(u, dim=1)        # (B,T,2)
    return x0.unsqueeze(1) + inc             # (B,T,2)

def robustness_soft_batch(traj, obs_xy, obs_r, goal_xy, goal_r, tau):
    # traj: (B,T,2)
    B, T, _ = traj.shape
    if obs_xy.numel() > 0:
        # (B,T,M)
        diff = traj.unsqueeze(2) - obs_xy.view(1,1,-1,2)
        d = diff.square().sum(-1).sqrt()
        margin = d - obs_r.view(1,1,-1)
        obs_t = margin.min(dim=2).values          # (B,T)
        G_avoid = G_always_soft(obs_t, tau=tau)   # (B,)
    else:
        G_avoid = torch.full((B,), 1e6, device=traj.device)

    rho_goal_t = ap_goal_inside(traj, goal_xy, goal_r)  # implement to accept (B,T,2)
    F_reach = F_eventually_soft(rho_goal_t, tau=tau)    # (B,)
    return torch.minimum(G_avoid, F_reach)              # (B,)

def optimize_controls_annealed_batched(
    x0_B2, obs_xy, obs_r, goal_xy, goal_r,
    T=64, dt=0.1, taus=(0.6,0.3,0.15), iters_per_tau=80, lr=0.25, max_speed=1.6,
    hard_eval_fn=None, init_u=None
):
    device = x0_B2.device
    B = x0_B2.shape[0]
    if init_u is None:
        u = torch.zeros(B, T, 2, device=device, requires_grad=True)
    else:
        u = init_u.detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([u], lr=lr)
    best_u = u.detach().clone()
    best_hard = torch.full((B,), -1e9, device=device)

    for tau in taus:
        for _ in range(iters_per_tau):
            opt.zero_grad(set_to_none=True)
            traj = simulate_batch(x0_B2, u, dt)               # (B,T,2)
            rho_soft = robustness_soft_batch(traj, obs_xy, obs_r, goal_xy, goal_r, tau)  # (B,)
            smooth = (u[:,1:]-u[:,:-1]).pow(2).mean(dim=(1,2))                           # (B,)
            loss = (-rho_soft + 0.01*(u**2).mean(dim=(1,2)) + 0.02*smooth).mean()
            loss.backward()
            opt.step()
            # project speeds
            sp = (u*u).sum(-1, keepdim=True).sqrt() + 1e-6
            u.data = u.data * torch.clamp(max_speed / sp, max=1.0)

        with torch.no_grad():
            traj = simulate_batch(x0_B2, u, dt)
            # evaluate hard robustness independently per sample (exact)
            hard = hard_eval_fn(traj)  # expect (B,) vector
            improved = hard > best_hard + 1e-6
            best_hard[improved] = hard[improved]
            best_u[improved] = u.detach()[improved]

            # rollback where not improved
            u.data[~improved] = best_u[~improved]

    return best_u, best_hard
