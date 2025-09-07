import torch

def ap_goal_inside(xy, goal_xy, goal_radius):
    d = torch.linalg.norm(xy - goal_xy, dim=-1)
    return goal_radius - d

def ap_outside_obstacle(xy, obs_xy, obs_radius):
    d = torch.linalg.norm(xy - obs_xy, dim=-1)
    return d - obs_radius

def ap_pairwise_separation(xyA, xyB, d_min):
    d = torch.linalg.norm(xyA - xyB, dim=-1)
    return d - d_min
