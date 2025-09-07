import torch
import numpy as np

from envs import Env, Nav2D
from ap_defs import ap_map_nav2d, GF_Avoid_Reach, AP, StlFormula

def generate_env(num_obstacles, seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    spec_dict = {
        'goal': {'center': torch.tensor([8.0, 8.0]), 'radius': 1.0},
        'avoid': [],
    }
    for i in range(num_obstacles):
        spec_dict['avoid'].append({
            'center': torch.rand(2) * 16 - 8,
            'radius': 1.0,
        })
    spec = GF_Avoid_Reach(spec_dict)
    start_pos = torch.tensor([-8.0, -8.0])
    env = Nav2D(start_pos, T=64, K=8, dt=0.1, ap_map=ap_map_nav2d, spec=spec)
    return env

def generate_hard_env_and_spec(T, K, num_obstacles_range=(8, 15), seed=None, formula_type='default'):
    """
    Generates a significantly harder environment with more obstacles and
    a more constrained STL specification. This is for "hard-case mining"
    to create a robust training dataset.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 1. Generate a more cluttered environment
    num_obstacles = np.random.randint(num_obstacles_range[0], num_obstacles_range[1] + 1)
    
    obstacles = []
    # Generate obstacles in a way that creates corridors and traps
    for i in range(num_obstacles):
        # Cluster some obstacles together
        if i % 3 == 0 and len(obstacles) > 0:
            cluster_center = obstacles[-1]['center']
            center = cluster_center + (torch.rand(2) - 0.5) * 4.0
        else:
            center = torch.rand(2) * 18 - 9
        
        # Clamp to bounds to avoid trivial edge cases
        center = torch.clamp(center, -9.0, 9.0)

        # Ensure no overlap with start
        if torch.linalg.norm(center - torch.tensor([-8.0, -8.0])) < 2.5:
            continue

        obstacles.append({'center': center, 'radius': 1.0 + torch.rand(1).item() * 0.5})

    # 2. Create a more challenging STL specification
    start_pos = torch.tensor([-8.0, -8.0])
    goal_center = torch.tensor([8.0, 8.0])
    waypoint1_center = torch.tensor([np.random.uniform(-7, -2), np.random.uniform(2, 7)])
    waypoint2_center = torch.tensor([np.random.uniform(2, 7), np.random.uniform(-7, -2)])
    
    # Ensure waypoints are not inside obstacles
    for obs in obstacles:
        if torch.linalg.norm(waypoint1_center - obs['center']) < obs['radius'] + 1.0:
            waypoint1_center = torch.rand(2) * 10 - 5 # re-sample
        if torch.linalg.norm(waypoint2_center - obs['center']) < obs['radius'] + 1.0:
            waypoint2_center = torch.rand(2) * 10 - 5 # re-sample

    spec_dict = {
        'goal': {'center': goal_center, 'radius': 1.0},
        'avoid': obstacles,
        'waypoint1': {'center': waypoint1_center, 'radius': 1.5},
        'waypoint2': {'center': waypoint2_center, 'radius': 1.5},
    }

    # Use different complex formulas to train for variety
    if formula_type == 'default':
        # Eventually reach waypoint1, then eventually reach goal, while always avoiding obstacles
        spec_string = "G(~avoid) & F(waypoint1 & F(goal))"
    elif formula_type == 'sequence':
        # Must visit waypoint1 THEN waypoint2 THEN goal
        spec_string = "G(~avoid) & F(waypoint1 & F(waypoint2 & F(goal)))"
    else: # until
        # Stay out of waypoint2 UNTIL you have hit waypoint1
        spec_string = "G(~avoid) & (~waypoint2 U waypoint1) & F(goal)"

    spec = StlFormula(spec_dict, spec_string)

    env = Nav2D(start_pos, T=T, K=K, dt=0.1, ap_map=ap_map_nav2d, spec=spec)
    return env
