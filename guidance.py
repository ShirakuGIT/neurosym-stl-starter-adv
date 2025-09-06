import torch

@torch.no_grad()
def add_goal_bias(x0, u_seq, goal_xy, dt=0.1, beta=0.35, max_speed=1.6):
    """
    Iteratively propagate state and add a small drift each step toward the goal.
    x0: (B,2)  u_seq: (B,T,2)  goal_xy: (2,)
    """
    B, T, _ = u_seq.shape
    device = u_seq.device
    x = x0.clone()
    u = u_seq.clone()
    g = goal_xy.view(1, 2).to(device).expand(B, 2)

    for t in range(T):
        # direction toward goal
        dir_vec = g - x
        norm = torch.norm(dir_vec, dim=-1, keepdim=True) + 1e-6
        step = (dir_vec / norm) * (beta * max_speed)  # small pull each step

        # blend and clip speed
        u[:, t, :] = u[:, t, :] + step
        sp = torch.norm(u[:, t, :], dim=-1, keepdim=True) + 1e-6
        u[:, t, :] = u[:, t, :] * torch.clamp(max_speed / sp, max=1.0)

        # roll state
        x = x + u[:, t, :] * dt

    return u
