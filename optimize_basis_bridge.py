# optimize_basis_bridge.py
import torch

import torch
from typing import Optional, Sequence
from optimize_basis import optimize_basis_controls

def optimize_basis_from_knots(
    x0, obs_xy, obs_r, goal_xy, goal_r,
    K=8, T=64, dt=0.1, init_knots=None,
    iters=200, lr=0.3, max_speed=1.6,
    hard_eval_fn=None,
    return_final_knots: bool = False  # <-- NEW
):
    # Convert "iters" into annealing schedule (simple, deterministic)
    # You can keep your existing taus/iters_per_tau logic if already set elsewhere.
    taus = (0.6, 0.3, 0.15, 0.08)
    iters_per_tau = max(1, iters // len(taus))

    if return_final_knots:
        u_best, best_hard, best_knots = optimize_basis_controls(
            x0=x0, obs_xy=obs_xy, obs_r=obs_r,
            goal_xy=goal_xy, goal_r=goal_r,
            T=T, dt=dt, K=K,
            taus=taus, iters_per_tau=iters_per_tau,
            lr=lr, max_speed=max_speed,
            hard_eval_fn=hard_eval_fn,
            init_knots=init_knots,
            return_knots=True,              # <-- ask for knots back
        )
        return u_best, best_hard, best_knots
    else:
        u_best, best_hard = optimize_basis_controls(
            x0=x0, obs_xy=obs_xy, obs_r=obs_r,
            goal_xy=goal_xy, goal_r=goal_r,
            T=T, dt=dt, K=K,
            taus=taus, iters_per_tau=iters_per_tau,
            lr=lr, max_speed=max_speed,
            hard_eval_fn=hard_eval_fn,
            init_knots=init_knots,
            return_knots=False,
        )
        return u_best, best_hard