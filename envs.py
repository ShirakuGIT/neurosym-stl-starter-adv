import torch

class PointMassEnv2D:
    def __init__(self, bounds=(-5, 5, -5, 5), dt=0.1):
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.dt = dt

    def simulate(self, x0, u_seq):
        B, T, _ = u_seq.shape
        x = x0.clone()
        traj = []
        for t in range(T):
            x = x + u_seq[:, t, :] * self.dt
            x[:, 0].clamp_(self.xmin, self.xmax)
            x[:, 1].clamp_(self.ymin, self.ymax)
            traj.append(x.clone())
        return torch.stack(traj, dim=1)  # (B,T,2)

def _min_signed_dist(points, obs_xy, obs_r):
    if obs_xy.numel() == 0:
        return torch.full((points.shape[0],), 1e6, device=points.device)
    vec = points.unsqueeze(1) - obs_xy.unsqueeze(0)  # (B,M,2)
    dist = torch.linalg.norm(vec + 1e-9, dim=-1)     # (B,M)
    sd = dist - obs_r.unsqueeze(0)                   # (B,M)
    return sd.min(dim=1).values                      # (B,)

def _segment_intersects_disks(p0, p1, obs_xy, obs_r, margin=0.0):
    # returns True if segment p0->p1 hits any inflated obstacle
    if obs_xy.numel() == 0:
        return False
    v = p1 - p0  # (2,)
    vv = (v * v).sum()
    t = ((obs_xy - p0) @ v) / (vv + 1e-9)  # (M,)
    t = t.clamp(0.0, 1.0)
    proj = p0.unsqueeze(0) + t.unsqueeze(1) * v.unsqueeze(0)  # (M,2)
    d = torch.linalg.norm(proj - obs_xy, dim=-1)  # (M,)
    return bool((d <= (obs_r + margin)).any())

def random_world(B, num_obstacles=5, bounds=(-5, 5, -5, 5), device='cpu',
                 min_start_goal_dist=2.0, los_block_required=True, los_margin=0.3, max_retries=50):
    """
    Harder-but-fair world sampler:
    - start not inside obstacles (with small margin),
    - start at least min_start_goal_dist from goal,
    - optionally require that straight-line start->goal intersects some inflated obstacle (no trivial LOS).
    """
    xmin, xmax, ymin, ymax = bounds
    # Sample world on CPU for stability
    for _ in range(max_retries):
        obs_xy = torch.empty((num_obstacles, 2)).uniform_(xmin + 0.5, xmax - 0.5)
        obs_r  = torch.empty((num_obstacles,)).uniform_(0.4, 1.0)
        goal_xy = torch.empty((2,)).uniform_(xmin + 1.0, xmax - 1.0)
        goal_r  = torch.tensor(0.7)

        # start: valid wrt obstacles & far enough from goal
        x0 = torch.empty((B, 2)).uniform_(xmin + 1.0, xmax - 1.0)
        # fix invalid starts
        for _k in range(8):
            sd = _min_signed_dist(x0, obs_xy, obs_r)  # (B,)
            near = ((x0 - goal_xy).norm(dim=-1) < min_start_goal_dist)
            bad = (sd < 0.10) | near
            if not bad.any():
                break
            x0[bad] = torch.empty((bad.sum(), 2)).uniform_(xmin + 1.0, xmax - 1.0)

        # Optionally enforce no trivial LOS
        if los_block_required:
            # test LOS only for the first batch sample's start (all starts are similar statistics)
            p0 = x0[0]
            if _segment_intersects_disks(p0, goal_xy, obs_xy, obs_r, margin=los_margin):
                # good: there is at least some blocking obstacle
                return (x0.to(device), obs_xy.to(device), obs_r.to(device),
                        goal_xy.to(device), goal_r.to(device))
        else:
            return (x0.to(device), obs_xy.to(device), obs_r.to(device),
                    goal_xy.to(device), goal_r.to(device))

    # Fallback (if retries exhausted): return last sample without LOS requirement
    return (x0.to(device), obs_xy.to(device), obs_r.to(device),
            goal_xy.to(device), goal_r.to(device))
