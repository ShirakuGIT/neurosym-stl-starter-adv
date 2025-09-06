# optimize_basis.py
import torch
from torch import amp
from control_basis import synthesize_controls_from_knots, _get_basis
from optimize_stl_annealed import simulate, robustness_soft

def _make_amp():
    # Unified AMP helper; avoids device_type kw that breaks on older torch
    if not torch.cuda.is_available():
        class _Null:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        return None, _Null()
    try:
        scaler = amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        autocast_ctx = amp.autocast("cuda", enabled=torch.cuda.is_available())
    except Exception:
        scaler = amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        autocast_ctx = amp.autocast("cuda", enabled=torch.cuda.is_available())
    return scaler, autocast_ctx

def optimize_basis_controls(
    x0, obs_xy, obs_r, goal_xy, goal_r, T=64, dt=0.1, K=8,
    taus=(0.4, 0.2, 0.1, 0.05), iters_per_tau=150, lr=0.3, max_speed=1.6,
    hard_eval_fn=None, init_knots=None, degree=3,
):
    """
    Basis-space optimizer:
      - Precomputes (and caches) the B-spline basis on CPU; reuses it each step.
      - Keeps basis ops in fp32 outside autocast to avoid CUDA allocator churn.
    """
    device = x0.device
    dtype = torch.float32  # keep basis/control math in fp32 for stability/speed

    # initialize knots
    if init_knots is not None:
        knots = init_knots.detach().to(device=device, dtype=dtype).clone().requires_grad_(True)
    else:
        knots = torch.zeros(K, 2, device=device, dtype=dtype, requires_grad=True)

    opt = torch.optim.Adam([knots], lr=lr)
    scaler, autocast_ctx = _make_amp()

    # Pre-fetch basis once to avoid rebuilding inside the loop
    with torch.no_grad():
        B = _get_basis(T=T, K=K, degree=degree, device=device).to(dtype)  # (T,K)

    def _controls_from_knots(k):
        # Avoid autocast on the tiny matmul; use prebuilt fp32 basis.
        return (B @ k)  # (T,2)

    best_knots = knots.detach().clone()
    best_hard = -1e9

    for tau in taus:
        for _ in range(iters_per_tau):
            opt.zero_grad(set_to_none=True)

            # synthesize controls in fp32 (no autocast), then simulate/evaluate
            u = _controls_from_knots(knots)  # (T,2), fp32

            # simple speed projection (scale down uniformly if needed)
            with torch.no_grad():
                sp_max = torch.norm(u, dim=-1).max()
                if sp_max > max_speed:
                    knots.data.mul_(float(max_speed / (sp_max + 1e-6)))

            # simulate + soft robustness (these can benefit from autocast on CUDA)
            with autocast_ctx:
                traj = simulate(x0, u, dt)  # u is fp32; simulate handles dtype internally
                rho_soft = robustness_soft(traj, obs_xy, obs_r, goal_xy, goal_r, tau=tau)
                smooth = (u[1:] - u[:-1]).pow(2).mean()
                loss = -rho_soft + 0.01 * (u ** 2).mean() + 0.02 * smooth

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # anneal checkpoint
        with torch.no_grad():
            u_proj = _controls_from_knots(knots)
            traj = simulate(x0, u_proj, dt)
            hard_rho = hard_eval_fn(traj) if hard_eval_fn else rho_soft
            if float(hard_rho.item()) > best_hard + 1e-6:
                best_hard = float(hard_rho.item())
                best_knots = knots.detach().clone()
            else:
                knots.data.copy_(best_knots)

    u_best = _controls_from_knots(best_knots)
    return u_best, best_hard
