# optimize_basis.py
import torch
from control_basis import synthesize_controls_from_knots
from optimize_stl_annealed import simulate, robustness_soft

def optimize_basis_controls(
    x0, obs_xy, obs_r, goal_xy, goal_r, T=64, dt=0.1, K=8,
    taus=(0.4,0.2,0.1,0.05), iters_per_tau=150, lr=0.3, max_speed=1.6, hard_eval_fn=None
):
    device = x0.device
    knots = torch.zeros(K, 2, device=device, requires_grad=True)
    opt = torch.optim.Adam([knots], lr=lr)

    best_knots = knots.detach().clone()
    best_hard = -1e9

    for tau in taus:
        for _ in range(iters_per_tau):
            opt.zero_grad()
            u = synthesize_controls_from_knots(knots, T)
            # speed cap projection in function space: scale knots if needed
            sp = torch.norm(u.detach(), dim=-1).max().item()
            if sp > max_speed:
                knots.data *= (max_speed / (sp + 1e-6))

            traj = simulate(x0, u, dt)
            rho_soft = robustness_soft(traj, obs_xy, obs_r, goal_xy, goal_r, tau=tau)
            smooth = (u[1:] - u[:-1]).pow(2).mean()
            loss = -rho_soft + 0.01*(u**2).mean() + 0.02*smooth
            loss.backward()
            opt.step()

        with torch.no_grad():
            u = synthesize_controls_from_knots(knots, T)
            traj = simulate(x0, u, dt)
            hard_rho = hard_eval_fn(traj) if hard_eval_fn else rho_soft
            if float(hard_rho.item()) > best_hard + 1e-6:
                best_hard = float(hard_rho.item())
                best_knots = knots.detach().clone()
            else:
                knots.data.copy_(best_knots)

    u_best = synthesize_controls_from_knots(best_knots, T)
    return u_best, best_hard
