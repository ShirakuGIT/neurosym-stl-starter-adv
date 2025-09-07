# PASTE THIS CODE INTO: src/envs.py
import torch
from abc import ABC, abstractmethod

class Env(ABC):
    """
    Abstract base class for all environments.
    """
    def __init__(self, start_pos, T, K, dt, spec):
        self.start_pos = start_pos
        self.T = T
        self.K = K
        self.dt = dt
        self.spec = spec
        # Link the environment to the specification, enabling the spec to call back
        if self.spec is not None:
            self.spec.env = self

    @abstractmethod
    def step(self, x, u):
        """Propagate state x forward by one time step with control u."""
        pass

    @abstractmethod
    def ap_map(self, traj_batch):
        """
        Maps a batch of trajectories to a batch of atomic proposition values.
        Returns a tensor of shape (B, T, num_aps).
        """
        pass

class Nav2D(Env):
    """
    A 2D navigation environment with point obstacles.
    """
    def __init__(self, start_pos, T, K, dt, spec):
        super().__init__(start_pos, T, K, dt, spec)
        self.dim = 2

    def step(self, x, u):
        """Integrator dynamics."""
        return x + u * self.dt

    def ap_map(self, traj_batch):
        """
        Maps a batch of trajectories to atomic proposition values for Nav2D.
        traj_batch shape: (B, T, 2) for position
        """
        B, T, _ = traj_batch.shape
        num_aps = len(self.spec.ap_dict)
        ap_values = torch.zeros((B, T, num_aps), device=traj_batch.device, dtype=torch.float32)

        for ap_name, ap_idx in self.spec.ap_dict.items():
            params = self.spec.spec_dict[ap_name]
            
            if ap_name == 'avoid':
                # 'avoid' is a list of obstacles
                avoid_robs_per_obs = []
                for obs in params:
                    center = obs['center'].to(traj_batch.device)
                    radius = obs['radius']
                    # Robs = distance from center - radius (positive is safe)
                    dist_to_center = torch.linalg.norm(traj_batch - center.view(1, 1, 2), dim=-1)
                    avoid_robs_per_obs.append(dist_to_center - radius)
                # We are safe if we are outside ALL obstacles (min robustness over obstacles)
                ap_values[:, :, ap_idx] = torch.min(torch.stack(avoid_robs_per_obs), dim=0)[0]
            else:
                # Other APs are regions (goal, waypoint, etc.)
                center = params['center'].to(traj_batch.device)
                radius = params['radius']
                # Robs = radius - distance from center (positive is inside)
                dist_to_center = torch.linalg.norm(traj_batch - center.view(1, 1, 2), dim=-1)
                ap_values[:, :, ap_idx] = radius - dist_to_center

        return ap_values