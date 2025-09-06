import argparse, os, time
import torch
import numpy as np
import matplotlib.pyplot as plt

from envs import PointMassEnv2D, random_world
from samplers import sample_with_ddpm_or_fallback
from stl_eval import spec_G_avoid_and_F_reach, spec_bounded_avoid_then_reach, chunked_eval
from logging_utils import append_record


import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"


def adaptive_N(N_base, num_obstacles, T, depth=1, num_AP=4, cap=None):
    """
    Adaptive sample count with an optional safety cap to prevent OOM on laptops.
    """
    N = int(N_base * (1 + 0.08 * num_AP + 0.15 * depth + 0.002 * T) * (1 + 0.05 * num_obstacles))
    if cap is not None:
        N = min(N, cap)
    return max(1, N)

def plot_world(traj, obs_xy, obs_r, goal_xy, goal_r, title, outpath):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    for i in range(len(obs_xy)):
        c = plt.Circle(obs_xy[i], obs_r[i], fill=False)
        ax.add_artist(c)
    g = plt.Circle(goal_xy, goal_r, color='green', fill=False, linestyle='--')
    ax.add_artist(g)
    ax.scatter(traj[0, 0], traj[0, 1], s=40, label='start')
    ax.plot(traj[:, 0], traj[:, 1], label='best traj')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_title(title); ax.grid(True); ax.legend()
    fig.savefig(outpath, dpi=160); plt.close(fig)

def main(args):
    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    # Global seeds (reproducible world + sampling)
    # Reproducible RNG (must be before random_world)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        import random as _py_random
        _py_random.seed(args.seed)

    world_seed_used = args.world_seed if (args.world_seed is not None and args.world_seed >= 0) \
                      else torch.randint(0, 10_000_000, (1,)).item()
    torch.manual_seed(world_seed_used)
    np.random.seed(world_seed_used)

    N = adaptive_N(args.N_base, args.num_obstacles, args.T,
                   depth=2 if args.spec == 'bounded' else 1,
                   num_AP=2 + args.num_obstacles,
                   cap=args.N_cap)

    env = PointMassEnv2D(dt=args.dt)
    x0, obs_xy, obs_r, goal_xy, goal_r = random_world(N, args.num_obstacles, device=device)

    x0      = x0.contiguous()
    obs_xy  = obs_xy.contiguous()
    obs_r   = obs_r.contiguous()
    goal_xy = goal_xy.contiguous()
    goal_r  = goal_r.contiguous()

        # ---------------- Hard robustness (torch, no specs module) ----------------
    def _hard_rho_of(traj_T2):
        """
        traj_T2: (T,2) positions for a single trajectory on current device.
        Hard STL: G(avoid) ∧ F(reach)
        """
        # G(avoid)
        if obs_xy.numel() > 0:
            diff = traj_T2.unsqueeze(1) - obs_xy.unsqueeze(0)       # (T, nObs, 2)
            dists = torch.linalg.norm(diff, dim=-1)                 # (T, nObs)
            margins = dists - obs_r.unsqueeze(0)                    # (T, nObs)
            rho_avoid_t = margins.min(dim=1).values                 # (T,)
            G_avoid = rho_avoid_t.min()
        else:
            G_avoid = torch.tensor(1e6, device=traj_T2.device)

        # F(reach)
        d_goal = torch.linalg.norm(traj_T2 - goal_xy.unsqueeze(0), dim=-1)  # (T,)
        rho_goal_t = goal_r - d_goal                                        # (T,)
        F_reach = rho_goal_t.max()

        return torch.minimum(G_avoid, F_reach)

    # ---------------- Produce u_seq exactly once, based on baseline ----------------
    N = x0.shape[0]  # batch size
    t0 = time.time()

    if args.baseline == 'opt':
        from optimize_stl_annealed import optimize_controls_annealed

        # how many starts to actually optimize (rest will use sampler fallback)
        opt_count = N if (args.opt_max is None or args.opt_max <= 0) else min(args.opt_max, N)
        u_seq = torch.zeros((N, args.T, 2), device=device)

        # run optimizer for the first opt_count starts
        for i in range(opt_count):
            print(f"[opt] optimizing start {i+1}/{opt_count}…", flush=True)
            u_i, _ = optimize_controls_annealed(
                x0=x0[i],
                obs_xy=obs_xy, obs_r=obs_r,
                goal_xy=goal_xy, goal_r=goal_r,
                T=args.T, dt=args.dt,
                hard_eval_fn=_hard_rho_of
            )
            if not torch.is_tensor(u_i):
                u_i = torch.tensor(u_i, dtype=torch.float32, device=device)
            u_seq[i] = u_i.to(device)

        # fill the remainder with our heuristic sampler
        if opt_count < N:
            from guidance import add_goal_bias
            u_fallback = sample_with_ddpm_or_fallback(N - opt_count, args.T, device=device, max_speed=1.6)
            u_fallback = add_goal_bias(x0[opt_count:], u_fallback, goal_xy, dt=args.dt, beta=0.35, max_speed=1.6)
            u_seq[opt_count:] = u_fallback

    elif args.baseline == 'opt_from_rrt':
        from rrtgeom import plan_rrt_star, path_to_controls
        from optimize_stl_annealed import optimize_controls_annealed

        max_speed = 1.6
        u_seq = torch.zeros((N, args.T, 2), device=device)
        # optionally only optimize a subset (keeps rigor; you’ll report budget)
        opt_count = N if (args.opt_max is None or args.opt_max <= 0) else min(args.opt_max, N)

        for i in range(opt_count):
            if (i % 25) == 0:
                print(f"[opt] optimizing start {i+1}/{opt_count}…", flush=True)

            path = plan_rrt_star(
            x0=x0[i],                         # tensor
            goal_xy=goal_xy,                  # tensor
            obs_xy=obs_xy,                    # tensor
            obs_r=obs_r,                      # tensor
            max_iter=4000, step=0.30, r_rewire=0.60,
            goal_sample_rate=0.10, goal_thresh=float(goal_r.item()*0.9),
            xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0,
            seed=int(args.world_seed if args.world_seed is not None else 0)
        )
            if path is None or len(path) < 2:
                # no warm start; zeros let optimizer try locally
                init_u = None
            else:
                init_u = path_to_controls(path, T=args.T, dt=args.dt, max_speed=1.6, device=device)

            u_i, _ = optimize_controls_annealed(
                x0=x0[i], obs_xy=obs_xy, obs_r=obs_r,
                goal_xy=goal_xy, goal_r=goal_r,
                T=args.T, dt=args.dt, init_u=init_u,
                hard_eval_fn=_hard_rho_of,  # your hard evaluator from earlier
                taus=(0.6, 0.3, 0.15, 0.08), iters_per_tau=80, lr=0.25, max_speed=max_speed
            )
            u_seq[i] = u_i.to(device)

        if opt_count < N:
            from guidance import add_goal_bias
            u_fallback = sample_with_ddpm_or_fallback(N - opt_count, args.T, device=device, max_speed=max_speed)
            u_fallback = add_goal_bias(x0[opt_count:], u_fallback, goal_xy, dt=args.dt, beta=0.35, max_speed=max_speed)
            u_seq[opt_count:] = u_fallback


    elif args.baseline == 'ours':
        from guidance import add_goal_bias
        # from repair import one_step_repair  # optional
        u_seq = sample_with_ddpm_or_fallback(N, args.T, device=device, max_speed=1.6)
        u_seq = add_goal_bias(x0, u_seq, goal_xy, dt=args.dt, beta=0.35, max_speed=1.6)
        # u_seq = one_step_repair(x0, u_seq, obs_xy, obs_r, dt=args.dt, alpha=1.2, d_safe=0.25)

    else:
        raise NotImplementedError(f"baseline {args.baseline} not wired yet")

    # Simulate once
    traj = env.simulate(x0, u_seq)
    t1 = time.time()


    if args.spec == 'gf_avoid_reach':
        rho = chunked_eval(spec_G_avoid_and_F_reach, traj, obs_xy, obs_r, goal_xy, goal_r, chunk=args.chunk)
    else:
        # bounded windows: convert seconds to indices with dt spacing (here we just use fractions of T)
        a1, b1 = int(0), int(args.T - 1)
        a2, b2 = int(0.3 * args.T), int(0.7 * args.T)
        rho = chunked_eval(spec_bounded_avoid_then_reach, traj, obs_xy, obs_r, goal_xy, goal_r, a1, b1, a2, b2, chunk=args.chunk)

    passrate = (rho > 0).float().mean().item()

    # Choose top-K for δ-SMT
    mask = rho > args.rho_min
    idx_sorted = torch.argsort(rho, descending=True)
    top_idx = [i.item() for i in idx_sorted if mask[i]][:args.K]



    # δ-SMT stub stats
    # before: from smt_stub import check_delta_sat
    use_dreal = getattr(args, "dreal", False)
    if use_dreal:
        from verify_dreal import delta_sat_check

    acc = rej = near = 0
    backend = args.cert

    for i in top_idx:
        r = float(rho[i].item())
        if r < 0:
            rej += 1
            continue

        x0_i    = x0[i].detach().cpu().numpy().tolist()
        u_i     = u_seq[i].detach().cpu().numpy().tolist()
        obs_xy_i= obs_xy.detach().cpu().numpy().tolist()
        obs_r_i = obs_r.detach().cpu().numpy().tolist()
        goal_xy_i = goal_xy.detach().cpu().numpy().tolist()
        goal_r_i  = float(goal_r.item())

        ok = True
        if backend == 'continuous':
            from verify_continuous import continuous_check_traj
            ok, _why = continuous_check_traj(x0_i, u_i, args.dt, obs_xy_i, obs_r_i, goal_xy_i, goal_r_i)
        elif backend == 'z3':
            from verify_z3 import z3_check_traj
            ok = z3_check_traj(x0_i, u_i, args.dt, obs_xy_i, obs_r_i, goal_xy_i, goal_r_i)
        elif backend == 'dreal':
            from verify_dreal import delta_sat_check
            ok = delta_sat_check(x0_i, u_i, args.dt, obs_xy_i, obs_r_i, goal_xy_i, goal_r_i,
                                delta=args.delta, dreal_path="dreal")
        else:
            ok = True  # no extra certification

        if ok: acc += 1
        else:  near += 1



    # Best traj
    best_i = int(idx_sorted[0].item())
    best_traj = traj[best_i].detach().cpu().numpy()
    best_rho = float(rho[best_i].item())

    # Throughput
    gpu_traj_s = N / max(1e-6, (t1 - t0))

    # Log row
    rec = dict(task=args.task, spec=args.spec, device=str(device), N=N, T=args.T, dt=args.dt,
               num_obstacles=args.num_obstacles, passrate=passrate, success=passrate,
               best_rho=best_rho, gpu_traj_s=gpu_traj_s, rho_min=args.rho_min, K=args.K, delta=args.delta,
               accept_rate=acc / max(1, len(top_idx)), reject_rate=rej / max(1, N), near_feasible=near / max(1, len(top_idx)),
               seed=args.seed, world_seed=world_seed_used,
               goal_r=float(goal_r.item()),min_start_goal_dist=2.0,
               los_block_required=True)
    append_record("outputs/logs/records.csv", rec)
    
    # Plot
    title = f"Success={passrate*100:.1f}% | Best ρ={best_rho:.3f} | N={N}"
    plot_world(best_traj,
               obs_xy.detach().cpu().numpy(),
               obs_r.detach().cpu().numpy(),
               goal_xy.detach().cpu().numpy(),
               float(goal_r.item()),
               title,
               f"outputs/plots/{args.task}_{args.spec}.png")

    print(f"Device: {device}")
    print(f"N(adaptive)={N}, T={args.T}, dt={args.dt}, obstacles={args.num_obstacles}")
    print(f"GPU eval passrate: {passrate*100:.2f}% | top-K (ρ>{args.rho_min})={len(top_idx)} | δ accept rate ~ {acc/max(1,len(top_idx)):.2f}")
    print(f"Throughput (traj/s): {gpu_traj_s:.1f}")
    print("Plot saved to outputs/plots/")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--task', type=str, default='nav2d')
    p.add_argument('--spec', type=str, default='gf_avoid_reach', choices=['gf_avoid_reach', 'bounded'])
    p.add_argument('--N_base', type=int, default=512)
    p.add_argument('--N_cap', type=int, default=2048, help='cap on adaptive N to avoid OOM on laptops')
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--num_obstacles', type=int, default=5)
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--rho_min', type=float, default=0.03)
    p.add_argument('--K', type=int, default=16)
    p.add_argument('--delta', type=float, default=1e-2)
    p.add_argument('--seed', type=int, default=123, help='global RNG seed')
    p.add_argument('--world_seed', type=int, default=0, help='seed for world generation; -1 = random each run')
    p.add_argument('--dreal', action='store_true', help='use dReal δ-SMT gate instead of stub')
    p.add_argument('--smt', type=str, default='none', choices=['none','z3','dreal'],
               help='choose SMT backend: none | z3 | dreal (dReal requires Linux/WSL)')
    p.add_argument('--cert', type=str, default='continuous',
               choices=['none','continuous','z3','dreal'],
               help='final certification backend')
    p.add_argument('--baseline', type=str, default='ours',
    choices=['ours','opt','rrt','opt_from_rrt','opt_from_ddpm'],
    help='pipeline to run')
    p.add_argument('--opt_max', type=int, default=None,
                help='If set, only optimize this many starts; others use sampler.')
    p.add_argument('--batch_opt', type=int, default=32,
                help='Batch width for the (optional) batched optimizer path.')








    args = p.parse_args()
    main(args)
