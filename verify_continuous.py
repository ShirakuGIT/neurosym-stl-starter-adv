# verify_continuous.py (fast, exact)
import torch

def seg_circle_clear(p0, p1, c, R):
    # p0,p1,c: (...,2), R: (...,) or scalar. Returns Bool tensor (...,)
    v = p1 - p0                                  # (...,2)
    w = c - p0                                   # (...,2)
    vv = (v*v).sum(-1) + 1e-12                   # (...,)
    t  = (w*v).sum(-1) / vv                      # (...,)
    t  = t.clamp(0.0, 1.0)
    closest = p0 + t.unsqueeze(-1) * v           # (...,2)
    d2 = ((closest - c)**2).sum(-1)              # (...,)
    return d2 >= (R*R)

def continuous_check_traj(x0, u, dt, obs_xy, obs_r, goal_xy, goal_r, margin=0.0):
    # x0: (2,), u: (T,2), obs_xy: (M,2), obs_r: (M,), goal_xy: (2,)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x0 = torch.tensor(x0, dtype=torch.float32, device=device)
    u  = torch.tensor(u,  dtype=torch.float32, device=device)
    obs_xy = torch.tensor(obs_xy, dtype=torch.float32, device=device)
    obs_r  = torch.tensor(obs_r,  dtype=torch.float32, device=device)
    goal_xy= torch.tensor(goal_xy,dtype=torch.float32, device=device)
    goal_r = float(goal_r)

    # build trajectory (vectorized)
    inc  = dt * torch.cumsum(u, dim=0)           # (T,2)
    traj = x0.unsqueeze(0) + inc                 # (T,2)

    # segment endpoints
    p0 = traj[:-1]                               # (T-1,2)
    p1 = traj[1:]                                # (T-1,2)

    # batch all segs × all obstacles
    S = p0.shape[0]
    M = obs_xy.shape[0]
    p0e = p0.unsqueeze(1).expand(S, M, 2)        # (S,M,2)
    p1e = p1.unsqueeze(1).expand(S, M, 2)        # (S,M,2)
    ce  = obs_xy.unsqueeze(0).expand(S, M, 2)    # (S,M,2)
    Re  = (obs_r + margin).unsqueeze(0).expand(S, M)  # (S,M)

    clear = seg_circle_clear(p0e, p1e, ce, Re)   # (S,M)
    if not clear.all().item():
        return False, "obstacle-intersection"

    # goal: exact continuous check = any position within disk at any t
    # Sufficient + exact for piecewise linear: if any vertex inside OR any segment crosses goal circle.
    inside_vertices = ((traj - goal_xy)**2).sum(-1).sqrt() <= goal_r
    if inside_vertices.any().item():
        return True, "reach-at-vertex"

    # segment-circle intersection test with goal disk
    ge  = goal_xy.unsqueeze(0).expand(S, 2)      # (S,2)
    ReG = torch.full((S,), goal_r, device=device)
    to_goal = seg_circle_clear(p0, p1, ge, ReG)  # True => segment stays outside
    if (~to_goal).any().item():
        return True, "reach-by-crossing"

    return False, "never-reached"
