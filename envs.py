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

def random_world(B, num_obstacles=5, bounds=(-5, 5, -5, 5), device='cpu'):
    """
    Create small world tensors on CPU first to avoid flaky OOM on Windows/WDDM,
    then move them to the target device.
    """
    xmin, xmax, ymin, ymax = bounds
    obs_xy = torch.empty((num_obstacles, 2)).uniform_(xmin + 0.5, xmax - 0.5).to(device)
    obs_r  = torch.empty((num_obstacles,)).uniform_(0.4, 1.0).to(device)
    goal_xy = torch.empty((2,)).uniform_(xmin + 1.0, xmax - 1.0).to(device)
    goal_r  = torch.tensor(0.7).to(device)
    x0      = torch.empty((B, 2)).uniform_(xmin + 1.0, xmax - 1.0).to(device)
    return x0, obs_xy, obs_r, goal_xy, goal_r
