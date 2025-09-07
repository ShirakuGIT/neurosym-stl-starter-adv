import torch
import numpy as np

from rrt_bootstrap import find_knots_rrt
from optimize_basis import optimize_basis

def baseline_rrt(env):
    """Baseline: just RRT*"""
    knots = find_knots_rrt(env, rrt_time=1.0, rrt_iter=5000)
    if knots is None:
        return None, 0.0
    
    controls = knots[1:] - knots[:-1]
    traj = torch.cumsum(controls, dim=0)
    traj = torch.cat([knots[0].unsqueeze(0), traj], dim=0)
    
    robs = env.spec.get_robs(traj.unsqueeze(0))
    return traj, robs.item()

def baseline_opt_from_rrt(env, opt_iters=200):
    """Baseline: RRT* followed by basis optimization"""
    knots0 = find_knots_rrt(env, rrt_time=0.5, rrt_iter=2000)
    if knots0 is None:
        return None, 0.0

    knots_opt, robs = optimize_basis(knots0, env, n_iters=opt_iters)
    
    controls = knots_opt[1:] - knots_opt[:-1]
    traj = torch.cumsum(controls, dim=0)
    traj = torch.cat([knots_opt[0].unsqueeze(0), traj], dim=0)

    return traj, robs