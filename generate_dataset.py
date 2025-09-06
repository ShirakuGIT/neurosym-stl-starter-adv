# generate_dataset.py
import os, time, argparse
import torch
import numpy as np
from envs import PointMassEnv2D, random_world
from rrtgeom import plan_rrt_star
from rrt_bootstrap import path_to_knots
from optimize_basis_bridge import optimize_basis_from_knots
from logging_utils import now_timestamp


@torch.no_grad()
def encode_env(x0, obs_xy, obs_r, goal_xy, goal_r, max_obs=8):
    """
    Flatten env into fixed-size vector for conditioning:
    [x0(2), goal(2), goal_r(1), obs_xy(2*max_obs), obs_r(max_obs), n_obs(1)]
    """
    dev = x0.device
    n = obs_xy.shape[0]
    n_use = min(int(n), int(max_obs))

    # sort obstacles by distance to start
    if n > 0:
        d = torch.linalg.norm(obs_xy - x0, dim=-1)
        idx = torch.argsort(d)[:n_use]
    else:
        idx = torch.tensor([], dtype=torch.long, device=dev)

    pad_xy = torch.zeros(max_obs, 2, device=dev, dtype=x0.dtype)
    pad_r  = torch.zeros(max_obs, device=dev, dtype=x0.dtype)
    if n_use > 0:
        pad_xy[:n_use] = obs_xy[idx]
        pad_r[:n_use]  = obs_r[idx]

    parts = [
        x0,                             # (2,)
        goal_xy,                        # (2,)
        goal_r.view(1),                 # (1,)
        pad_xy.reshape(-1),             # (2*max_obs,)
        pad_r,                          # (max_obs,)
        torch.tensor([float(n)], device=dev, dtype=x0.dtype),  # (1,)
    ]
    return torch.cat(parts, dim=0)  # (2+2+1+2M+M+1) = 6+3M with M=max_obs


def run_one_env(
    *,
    K: int,
    T: int,
    dt: float,
    device: str,
    num_obstacles: int,
    rho_min: float,
    rrt_time: float,
    rrt_iter: int,
    rrt_step: float,
    rrt_goal_rate: float,
    rewire_k: int,
    ncheck: int,
    opt_iters: int,
    world_seed: int = -1,
):
    """
    Generates a random world, plans with RRT*, converts to knots, runs basis optimizer.
    Returns (x0, obs_xy, obs_r, goal_xy, goal_r, final_knots, best_rho) or None on fail.
    """
    torch_device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    env = PointMassEnv2D(dt=dt)

    # Random world; world_seed < 0 → new RNG each call
    if world_seed >= 0:
        torch.manual_seed(world_seed)
        np.random.seed(world_seed)

    x0, obs_xy, obs_r, goal_xy, goal_r = random_world(1, num_obstacles=num_obstacles, device=torch_device)
    x0, obs_xy, obs_r, goal_xy, goal_r = x0[0], obs_xy.contiguous(), obs_r.contiguous(), goal_xy, goal_r  # un-batch x0

    # --- RRT* with time cap ---
    t0p = time.time()
    path = plan_rrt_star(
        x0=x0, goal_xy=goal_xy, obs_xy=obs_xy, obs_r=obs_r,
        max_iter=int(rrt_iter), step=float(rrt_step),
        max_time=float(rrt_time),
        rewire_k=int(rewire_k), ncheck=int(ncheck),
        r_rewire=0.0,
        goal_sample_rate=float(rrt_goal_rate),
        xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0,
        goal_thresh=float(goal_r.item()),
        seed=None,  # randomized per call
    )
    plan_dt = time.time() - t0p

    # Optional: if no path, skip quickly
    init_knots = None
    if path is not None and len(path) >= 2:
        init_knots = path_to_knots(path, K=K, device=torch_device)

    # --- Optimize in basis space (returns controls + final knots) ---
    u_best, best_rho, final_knots = optimize_basis_from_knots(
        x0=x0, obs_xy=obs_xy, obs_r=obs_r,
        goal_xy=goal_xy, goal_r=goal_r,
        K=K, T=T, dt=dt,
        init_knots=init_knots,
        iters=int(opt_iters),
        lr=0.3, max_speed=1.6,
        hard_eval_fn=None,
        return_final_knots=True,
    )

    if float(best_rho) <= float(rho_min):
        return None  # discard non-robust solutions

    return (x0, obs_xy, obs_r, goal_xy, goal_r, final_knots, float(best_rho), plan_dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="datasets/expert_knots.pth")
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_obstacles", type=int, default=5)
    ap.add_argument("--rho_min", type=float, default=0.05)
    ap.add_argument("--max_obs", type=int, default=8)

    # RRT* knobs
    ap.add_argument("--rrt_time", type=float, default=0.25, help="time cap per world (s)")
    ap.add_argument("--rrt_iter", type=int, default=4000, help="max iterations")
    ap.add_argument("--rrt_step", type=float, default=0.30)
    ap.add_argument("--rrt_goal_rate", type=float, default=0.20)
    ap.add_argument("--rewire_k", type=int, default=24)
    ap.add_argument("--ncheck", type=int, default=8)

    # Optimizer knob
    ap.add_argument("--opt_iters", type=int, default=120)

    # Checkpointing
    ap.add_argument("--save_every", type=int, default=200, help="save temp every N samples")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    records = []
    kept = 0
    plan_times = []

    for i in range(args.num):
        out = run_one_env(
            K=args.K, T=args.T, dt=args.dt, device=args.device,
            num_obstacles=args.num_obstacles, rho_min=args.rho_min,
            rrt_time=args.rrt_time, rrt_iter=args.rrt_iter, rrt_step=args.rrt_step,
            rrt_goal_rate=args.rrt_goal_rate, rewire_k=args.rewire_k, ncheck=args.ncheck,
            opt_iters=args.opt_iters, world_seed=-1,
        )

        if out is not None:
            x0, obs_xy, obs_r, goal_xy, goal_r, knots, rho, plan_dt = out
            cond = encode_env(x0, obs_xy, obs_r, goal_xy, goal_r, max_obs=args.max_obs)
            records.append(dict(cond=cond.cpu(), knots=knots.detach().cpu(), rho=float(rho)))
            kept += 1
            plan_times.append(plan_dt)

        # light progress log
        if (i + 1) % 25 == 0:
            med_plan = np.median(plan_times) if plan_times else 0.0
            print(f"[data] {i+1}/{args.num} | kept={kept} | med RRT* {med_plan:.3f}s | keep {100.0*kept/max(1,i+1):.1f}%", flush=True)

        # periodic checkpoint
        if args.save_every and ((i + 1) % args.save_every == 0):
            tmp_out = args.out + ".tmp"
            torch.save(dict(
                K=args.K, T=args.T, dt=args.dt, max_obs=args.max_obs, ts=now_timestamp(), data=records
            ), tmp_out)
            print(f"[data] checkpoint saved: {tmp_out} ({kept} samples)", flush=True)

    # final save
    torch.save(dict(
        K=args.K, T=args.T, dt=args.dt, max_obs=args.max_obs, ts=now_timestamp(), data=records
    ), args.out)
    print(f"[data] saved {kept} / {args.num} to {args.out}")


if __name__ == "__main__":
    # Avoid any torch.compile surprises
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    main()
