# micro_repair.py
import torch
from control_basis import synthesize_controls_from_knots
from optimize_stl_annealed import simulate, robustness_soft

@torch.no_grad()
def project_trust_region(knots, base, radius=0.15):
    delta = knots - base
    nrm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-9)
    scale = torch.clamp(radius / nrm, max=1.0)
    return base + delta * scale

def micro_repair_knots(
    knots_init: torch.Tensor,
    *,
    x0, obs_xy, obs_r, goal_xy, goal_r,
    T: int, dt: float,
    steps: int = 6, lr: float = 0.08,
    max_speed: float = 1.6, tau: float = 0.15,
    trust_radius: float = 0.15,
):
    """
    Tiny gradient-based nudge on B-spline knots to fix near-misses quickly.
    """
    device = knots_init.device
    knots = knots_init.detach().clone().requires_grad_(True)
    opt   = torch.optim.SGD([knots], lr=lr, momentum=0.0)

    best = knots.detach().clone()
    best_hard = -1e9

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        u = synthesize_controls_from_knots(knots, T)        # (T,2)
        # speed cap in function space (scale if needed)
        sp = torch.norm(u.detach(), dim=-1).max().item()
        if sp > max_speed:
            with torch.no_grad():
                knots.data *= (max_speed / (sp + 1e-6))
        traj = simulate(x0, u, dt)                          # (T,2)
        rho_soft = robustness_soft(traj, obs_xy, obs_r, goal_xy, goal_r, tau=tau)
        smooth   = (u[1:] - u[:-1]).pow(2).mean()
        loss = -rho_soft + 0.01*(u**2).mean() + 0.02*smooth
        loss.backward()
        opt.step()
        # trust region around initialization
        with torch.no_grad():
            knots.data = project_trust_region(knots.data, knots_init, radius=trust_radius)

        # track best
        with torch.no_grad():
            u_proj = synthesize_controls_from_knots(knots, T)
            traj_p = simulate(x0, u_proj, dt)
            # Use soft as proxy for speed; hard check will happen outside
            if float(rho_soft.item()) > best_hard + 1e-6:
                best_hard = float(rho_soft.item())
                best = knots.detach().clone()

    return best.detach()
