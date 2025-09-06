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
    taus=(0.4,0.2,0.1,0.05), iters_per_tau=150, lr=0.3, max_speed=1.6,
    hard_eval_fn=None,
    init_knots=None,                 # <-- (if you already added this, keep it)
    return_knots: bool = False       # <-- NEW
):
    device = x0.device

    # init knots
    if init_knots is not None:
        knots = init_knots.detach().clone().requires_grad_(True)
    else:
        knots = torch.zeros(K, 2, device=device, requires_grad=True)

    opt = torch.optim.Adam([knots], lr=lr)

    # (optional) AMP, keep your existing _make_amp() if you have one
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        autocast_ctx = torch.amp.autocast('cuda', enabled=torch.cuda.is_available())
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        autocast_ctx = torch.cuda.amp.autocast(enabled=torch.cuda.is_available())

    best_knots = knots.detach().clone()
    best_hard = -1e9

    for tau in taus:
        for _ in range(iters_per_tau):
            opt.zero_grad()
            with autocast_ctx:
                u = synthesize_controls_from_knots(knots, T)  # (T,2)

                # function-space speed cap (scale knot amplitudes if any u exceeds cap)
                sp_max = torch.norm(u.detach(), dim=-1).max()
                if sp_max.item() > max_speed:
                    scale = max_speed / (sp_max.item() + 1e-6)
                    with torch.no_grad():
                        knots.mul_(scale)

                traj = simulate(x0, u, dt)
                rho_soft = robustness_soft(traj, obs_xy, obs_r, goal_xy, goal_r, tau=tau)
                smooth = (u[1:] - u[:-1]).pow(2).mean()
                loss = -rho_soft + 0.01*(u**2).mean() + 0.02*smooth

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        with torch.no_grad():
            u_tmp = synthesize_controls_from_knots(knots, T)
            traj = simulate(x0, u_tmp, dt)
            hard_rho = hard_eval_fn(traj) if hard_eval_fn is not None else rho_soft
            if float(hard_rho.item()) > best_hard + 1e-6:
                best_hard = float(hard_rho.item())
                best_knots = knots.detach().clone()
            else:
                knots.data.copy_(best_knots)

    u_best = synthesize_controls_from_knots(best_knots, T)
    if return_knots:
        return u_best, best_hard, best_knots
    else:
        return u_best, best_hard