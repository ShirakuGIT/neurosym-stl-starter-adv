# PASTE THIS CODE INTO: src/scenarios.py
import torch
import numpy as np

from envs import Nav2D
from ap_defs import StlFormula

def generate_env(num_obstacles, seed=None, T=64, K=8):
    """
    Generates a standard environment with a simple G(~avoid) & F(goal) spec.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    obstacles = []
    while len(obstacles) < num_obstacles:
        center = torch.rand(2) * 16 - 8
        is_overlapping = any(torch.linalg.norm(center - obs['center']) < 2.5 for obs in obstacles)
        if not is_overlapping:
            obstacles.append({'center': center.to(torch.float32), 'radius': 1.0})

    spec_dict = {
        'goal': {'center': torch.tensor([8.0, 8.0], dtype=torch.float32), 'radius': 1.0},
        'avoid': obstacles,
    }
    spec_string = "always(avoid) & eventually(goal)"
    spec = StlFormula(spec_dict, spec_string)
    start_pos = torch.tensor([-8.0, -8.0], dtype=torch.float32)

    return Nav2D(start_pos, T=T, K=K, dt=0.1, spec=spec)

def generate_hard_env_and_spec(T, K, seed=None, formula_type='default'):
    """
    Generates a significantly harder environment with more obstacles and
    a more constrained STL specification for robust dataset creation.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    num_obstacles = np.random.randint(8, 13)
    obstacles = []
    while len(obstacles) < num_obstacles:
        center = torch.rand(2) * 18 - 9
        is_overlapping = any(torch.linalg.norm(center - obs['center']) < 2.2 for obs in obstacles)
        if not is_overlapping:
            obstacles.append({'center': center.to(torch.float32), 'radius': 1.0 + torch.rand(1).item() * 0.2})

    # Create more complex specifications
    goal_center = torch.tensor([np.random.uniform(7, 9), np.random.uniform(7, 9)], dtype=torch.float32)
    waypoint1_center = torch.tensor([np.random.uniform(-5, 5), np.random.uniform(5, -5)], dtype=torch.float32)

    spec_dict = {'avoid': obstacles}
    
    if formula_type == 'sequence':
        spec_dict['waypoint1'] = {'center': waypoint1_center, 'radius': 1.5}
        spec_dict['goal'] = {'center': goal_center, 'radius': 1.0}
        spec_string = "always(avoid) & eventually(waypoint1 & eventually(goal))"
    elif formula_type == 'until':
        spec_dict['waypoint1'] = {'center': waypoint1_center, 'radius': 1.5}
        spec_dict['goal'] = {'center': goal_center, 'radius': 1.0}
        spec_string = "always(avoid) & until(~goal, waypoint1)"
    else: # default
        spec_dict['goal'] = {'center': goal_center, 'radius': 1.0}
        spec_string = "always(avoid) & eventually(goal)"

    spec = StlFormula(spec_dict, spec_string)
    start_pos = torch.tensor([-8.0, -8.0], dtype=torch.float32)
    
    return Nav2D(start_pos, T=T, K=K, dt=0.1, spec=spec)