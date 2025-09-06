# optimize_stl_annealed.py
import os
import torch
import torch._dynamo as dynamo
from stl_soft import F_eventually_soft, G_always_soft
from ap_defs import ap_goal_inside, ap_outside_obstacle

# Ensure nothing tries to JIT/compile this; we want eager for stability
os.environ.setdefault("TORCH_COMPILE", "0")

def _make_amp():
    """
    Returns (scaler, autocast_context) that work on both older and newer PyTorch.
    If CUDA is unavailable, returns (None, a no-op context).
    """
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        class _NullCtx:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        return None, _NullCtx()

    # Prefer new torch.amp if present; otherwise fall back to torch.cuda.amp
    try:
        from torch import amp as _amp
        scaler = _amp.GradScaler(enabled=True)           # don't pass device_type for compat
        autocast_ctx = _amp.autocast(enabled=True)
        return scaler, autocast_ctx
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        autocast_ctx = torch.cuda.amp.autocast(enabled=True)
        return scaler, autocast_ctx

def simulate(x0, u, dt):
    """
    Simple point-mass Euler integration.
    x_{t+1} = x_t + u_t * dt
    Returns positions x[1:T] shaped (T,2), with gradients w.r.t. u.
    """
    # cumulative sum of controls -> displacements from x0
    steps = torch.cumsum(u, dim=0) * dt            # (T,2)
    return x0.unsqueeze(0) + steps                 # (T,2)

def robustness_soft(traj, obs_xy, obs_r, goal_xy, goal_r, tau=0.2):
    """
    Soft STL robustness for G(avoid) ∧ F(reach) using your differentiable APs.
    """
    xy = traj.unsqueeze(0)  # (1,T,2)

    if obs_xy.numel() > 0:
        all_obs = [ap_outside_obstacle(xy, obs_xy[i], obs_r[i])   # (1,T) each
                   for i in range(obs_xy.shape[0])]
        obs_t = torch.stack(all_obs, dim=0).min(dim=0).values     # (1,T)
        G_avoid = G_always_soft(obs_t, tau=tau).squeeze(0)        # ()
    else:
        G_avoid = torch.tensor(1e6, device=traj.device, dtype=traj.dtype)

    F_reach = F_eventually_soft(ap_goal_inside(xy, goal_xy, goal_r), tau=tau).squeeze(0)  # ()
    return torch.minimum(G_avoid, F_reach)  # scalar (0-dim tensor)

@dynamo.disable()  # force eager; avoid TorchInductor/Triton path
def optimize_controls_annealed(
    x0, obs_xy, obs_r, goal_xy, goal_r,
    T=64, dt=0.1, taus=(0.4, 0.2, 0.1, 0.05), iters_per_tau=120,
    lr=0.25, max_speed=1.6, hard_eval_fn=None, init_u=None
):
    """
    Annealed gradient ascent on soft robustness, projecting to a speed cap,
    with stage-end checks against a provided hard robustness function.

    Args:
      x0: (2,) tensor
      obs_xy: (nObs,2)
      obs_r:  (nObs,)
      goal_xy:(2,)
      goal_r: scalar tensor
      init_u: optional (T,2) controls to warm-start; else zeros
    Returns:
      (best_u: (T,2) tensor, best_hard_rho: float)
    """
    device = x0.device
    dtype = torch.float32

    if init_u is None:
        u = torch.zeros(T, 2, device=device, dtype=dtype, requires_grad=True)
    else:
        u = init_u.to(device=device, dtype=dtype).detach().clone().requires_grad_(True)
        # enforce speed cap on warm start
        sp = torch.norm(u.detach(), dim=-1, keepdim=True).clamp_min(1e-6)
        u.data = u.data * torch.clamp(max_speed / sp, max=1.0)

    opt = torch.optim.Adam([u], lr=lr)
    scaler, autocast_ctx = _make_amp()

    best_u = u.detach().clone()
    best_hard_rho = -1e9

    for tau in taus:
        for _ in range(iters_per_tau):
            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                traj = simulate(x0, u, dt)
                rho_soft = robustness_soft(traj, obs_xy, obs_r, goal_xy, goal_r, tau=tau)

                # small regularizers
                smooth = (u[1:] - u[:-1]).pow(2).mean() if T > 1 else torch.zeros((), device=device)
                loss = -rho_soft + 0.01*(u**2).mean() + 0.02*smooth

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            # project controls to speed cap
            sp = torch.norm(u.detach(), dim=-1, keepdim=True).clamp_min(1e-6)
            u.data = u.data * torch.clamp(max_speed / sp, max=1.0)

        # stage-end: accept only if hard robustness improves
        with torch.no_grad():
            traj = simulate(x0, u, dt)
            hard_rho = hard_eval_fn(traj) if hard_eval_fn is not None else rho_soft
            hard_val = float(hard_rho.item())
            if hard_val > best_hard_rho + 1e-6:
                best_hard_rho = hard_val
                best_u = u.detach().clone()
            else:
                # rollback if no improvement under hard metric
                u.data.copy_(best_u)

    return best_u, best_hard_rho
