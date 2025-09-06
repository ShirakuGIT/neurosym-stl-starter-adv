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
    # points: (B,2), obs_xy: (M,2), obs_r: (M,)
    if obs_xy.numel() == 0:
        return torch.full((points.shape[0],), 1e6, device=points.device)
    vec = points.unsqueeze(1) - obs_xy.unsqueeze(0)      # (B,M,2)
    dist = torch.linalg.norm(vec + 1e-9, dim=-1)         # (B,M)
    sd = dist - obs_r.unsqueeze(0)                        # (B,M)
    return sd.min(dim=1).values                           # (B,)

def random_world(B, num_obstacles=5, bounds=(-5, 5, -5, 5), device='cpu'):
    """
    Create world on CPU then move to device. Ensure x0 is not inside obstacles.
    """
    xmin, xmax, ymin, ymax = bounds
    # world
    obs_xy = torch.empty((num_obstacles, 2)).uniform_(xmin + 0.5, xmax - 0.5)
    obs_r  = torch.empty((num_obstacles,)).uniform_(0.4, 1.0)
    goal_xy = torch.empty((2,)).uniform_(xmin + 1.0, xmax - 1.0)
    goal_r  = torch.tensor(0.7)

    # spawn starts with a small safety margin
    x0 = torch.empty((B, 2)).uniform_(xmin + 1.0, xmax - 1.0)
    margin = 0.10
    for _ in range(6):  # a few retries to fix invalid starts
        sd = _min_signed_dist(x0, obs_xy, obs_r)  # (B,)
        bad = sd < margin
        if not bad.any():
            break
        # resample only the bad ones
        x0[bad] = torch.empty((bad.sum(), 2)).uniform_(xmin + 1.0, xmax - 1.0)
    # move tensors to device
    return (x0.to(device), obs_xy.to(device), obs_r.to(device),
            goal_xy.to(device), goal_r.to(device))
