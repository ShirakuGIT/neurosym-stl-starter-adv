import torch

def corridor_world(B, device='cpu'):
    # Two long obstacles forming a corridor from left to right; goal at right end
    obs_xy = torch.tensor([[-1.5, 0.5], [-1.5, -0.5], [1.5, 0.5], [1.5, -0.5]], dtype=torch.float32)
    obs_r  = torch.tensor([1.4, 1.4, 1.4, 1.4], dtype=torch.float32)  # fat bars approximated by discs
    x0     = torch.tensor([[ -4.0, 0.0 ]]).repeat(B,1)
    goal_xy= torch.tensor([ 4.0, 0.0 ])
    goal_r = torch.tensor(0.6)
    return x0.to(device), obs_xy.to(device), obs_r.to(device), goal_xy.to(device), goal_r.to(device)
