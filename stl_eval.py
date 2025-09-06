import torch
from stl_ops import G_always, F_eventually, G_bounded, F_bounded, U_until, conj, disj, neg
from ap_defs import ap_goal_inside, ap_outside_obstacle

def spec_G_avoid_and_F_reach(xy, obstacles_xy, obstacles_r, goal_xy, goal_r):
    B,T,_ = xy.shape
    device = xy.device
    if len(obstacles_xy)>0:
        all_obs = []
        for i in range(len(obstacles_xy)):
            rho_t = ap_outside_obstacle(xy, obstacles_xy[i].to(device), obstacles_r[i].to(device))
            all_obs.append(rho_t)
        obs_t = torch.stack(all_obs, dim=0).min(dim=0).values
        G_avoid = G_always(obs_t)
    else:
        G_avoid = torch.full((B,), 1e6, device=device)
    rho_goal_t = ap_goal_inside(xy, goal_xy.to(device), goal_r.to(device))
    F_reach = F_eventually(rho_goal_t)
    return torch.minimum(G_avoid, F_reach)

def spec_bounded_avoid_then_reach(xy, obstacles_xy, obstacles_r, goal_xy, goal_r, a1,b1,a2,b2):
    B,T,_ = xy.shape
    device = xy.device
    if len(obstacles_xy)>0:
        all_obs = []
        for i in range(len(obstacles_xy)):
            rho_t = ap_outside_obstacle(xy, obstacles_xy[i].to(device), obstacles_r[i].to(device))
            all_obs.append(rho_t)
        obs_t = torch.stack(all_obs, dim=0).min(dim=0).values
        G_avoid = G_bounded(obs_t, a1, b1)
    else:
        G_avoid = torch.full((B,), 1e6, device=device)
    rho_goal_t = ap_goal_inside(xy, goal_xy.to(device), goal_r.to(device))
    F_reach = F_bounded(rho_goal_t, a2, b2)
    return torch.minimum(G_avoid, F_reach)

def chunked_eval(eval_fn, *args, chunk=32, **kwargs):
    return eval_fn(*args, **kwargs)
