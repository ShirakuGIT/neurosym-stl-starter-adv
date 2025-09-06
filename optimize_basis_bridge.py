# optimize_basis_bridge.py
import torch

import torch
from typing import Optional, Sequence
from optimize_basis import optimize_basis_controls

def optimize_basis_from_knots(
    x0: torch.Tensor,
    obs_xy: torch.Tensor,
    obs_r: torch.Tensor,
    goal_xy: torch.Tensor,
    goal_r: torch.Tensor,
    *,
    K: int = 8,
    T: int = 64,
    dt: float = 0.1,
    init_knots: Optional[torch.Tensor] = None,
    iters: int = 200,
    lr: float = 0.3,
    max_speed: float = 1.6,
    taus: Sequence[float] = (0.6, 0.3, 0.15, 0.08),
    hard_eval_fn=None,
):
    """
    Thin wrapper that calls the basis optimizer with an (optional) warm-start
    set of knots coming from RRT*.

    Returns:
        u_best: (T,2) tensor of controls
        best_hard_rho: float
    """
    # Pass-through to the core basis optimizer (which accepts init_knots)
    u_best, best_hard_rho = optimize_basis_controls(
        x0=x0,
        obs_xy=obs_xy,
        obs_r=obs_r,
        goal_xy=goal_xy,
        goal_r=goal_r,
        T=T,
        dt=dt,
        K=K,
        taus=tuple(taus),
        iters_per_tau=max(1, iters // max(1, len(taus))),
        lr=lr,
        max_speed=max_speed,
        hard_eval_fn=hard_eval_fn,
        init_knots=init_knots,
    )
    return u_best, best_hard_rho