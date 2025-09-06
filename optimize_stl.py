import torch
from ap_defs import ap_goal_inside, ap_outside_obstacle
from stl_soft import F_eventually_soft, G_always_soft

@torch.no_grad()
def simulate(x0, u, dt):
    x = x0.clone()
    traj = []
    for t in range(u.shape[0]):
        x = x + u[t]*dt
        traj.append(x.clone())
    return torch.stack(traj, dim=0)  # (T,2)

def robustness_soft(traj, obs_xy, obs_r, goal_xy, goal_r, tau=0.05):
    # rho = min( G avoid, F reach ) using soft min/max
    B = 1
    xy = traj.unsqueeze(0)  # (1,T,2)
    if len(obs_xy) > 0:
        all_obs = []
        for i in range(len(obs_xy)):
            rho_t = ap_outside_obstacle(xy, obs_xy[i], obs_r[i])
            all_obs.append(rho_t)
        obs_t = torch.stack(all_obs, dim=0).min(dim=0).values  # (1,T)
        G_avoid = G_always_soft(obs_t, tau=tau).squeeze(0)
    else:
        G_avoid = torch.tensor(1e6, device=traj.device)
    rho_goal_t = ap_goal_inside(xy, goal_xy, goal_r)
    F_reach = F_eventually_soft(rho_goal_t, tau=tau).squeeze(0)
    return torch.minimum(G_avoid, F_reach)  # scalar per batch elem

def optimize_controls(x0, obs_xy, obs_r, goal_xy, goal_r, T=64, dt=0.1, iters=200, lr=0.2, tau=0.05):
    device = x0.device
    u = torch.zeros(T, 2, device=device, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    best_u = None; best_rho = -1e9
    for k in range(iters):
        opt.zero_grad()
        traj = simulate(x0, u, dt)  # (T,2)
        rho = robustness_soft(traj, obs_xy, obs_r, goal_xy, goal_r, tau=tau)
        loss = -rho + 0.01*(u**2).mean()  # maximize rho + small control penalty
        loss.backward()
        opt.step()
        # clip speed to 1.6
        sp = torch.norm(u.detach(), dim=-1, keepdim=True)+1e-6
        u.data = u.data * torch.clamp(1.6/sp, max=1.0)
        if rho.item() > best_rho:
            best_rho = rho.item(); best_u = u.detach().clone()
    return best_u  # (T,2)
