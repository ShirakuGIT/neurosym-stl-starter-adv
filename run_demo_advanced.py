# run_demo_advanced.py
# ----------------------------------------------------------------------
# MUST come before importing torch to actually disable compilation stacks.
import os as _os
_os.environ["TORCH_COMPILE_DISABLE"] = "1"
_os.environ["TORCHINDUCTOR_DISABLE"] = "1"
# ----------------------------------------------------------------------

import argparse, os, time
import numpy as np
import matplotlib.pyplot as plt
import torch
# add these imports
from prior_utils import encode_env, sample_knots_ddpm
from micro_repair import micro_repair_knots

from envs import PointMassEnv2D, random_world
from samplers import sample_with_ddpm_or_fallback
from stl_eval import (
    spec_G_avoid_and_F_reach,
    spec_bounded_avoid_then_reach,
    chunked_eval,
)
from logging_utils import append_record


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

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def main(args):
    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    # ---------- Reproducible RNG ----------
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        import random as _py_random
        _py_random.seed(args.seed)

    world_seed_used = (
        args.world_seed
        if (args.world_seed is not None and args.world_seed >= 0)
        else torch.randint(0, 10_000_000, (1,)).item()
    )
    torch.manual_seed(world_seed_used)
    np.random.seed(world_seed_used)

    # ---------- Problem size ----------
    N = adaptive_N(
        args.N_base,
        args.num_obstacles,
        args.T,
        depth=2 if args.spec == 'bounded' else 1,
        num_AP=2 + args.num_obstacles,
        cap=args.N_cap,
    )

    env = PointMassEnv2D(dt=args.dt)
    x0, obs_xy, obs_r, goal_xy, goal_r = random_world(N, args.num_obstacles, device=device)

    # normalize layout/dtype
    x0      = x0.contiguous().to(device=device, dtype=torch.float32)
    obs_xy  = obs_xy.contiguous().to(device=device, dtype=torch.float32)
    obs_r   = obs_r.contiguous().to(device=device, dtype=torch.float32)
    goal_xy = goal_xy.contiguous().to(device=device, dtype=torch.float32)
    goal_r  = goal_r.contiguous().to(device=device, dtype=torch.float32)

    # ---------- Hard robustness (torch, no specs module) ----------
    def _hard_rho_of(traj_T2: torch.Tensor):
        """
        traj_T2: (T,2) positions for a single trajectory on current device.
        STL: G(avoid) ∧ F(reach)
        """
        # G(avoid)
        if obs_xy.numel() > 0:
            diff = traj_T2.unsqueeze(1) - obs_xy.unsqueeze(0)     # (T, nObs, 2)
            dists = torch.linalg.norm(diff, dim=-1)               # (T, nObs)
            margins = dists - obs_r.unsqueeze(0)                  # (T, nObs)
            rho_avoid_t = margins.min(dim=1).values               # (T,)
            G_avoid = rho_avoid_t.min()
        else:
            G_avoid = torch.tensor(1e6, device=traj_T2.device)

        # F(reach)
        d_goal = torch.linalg.norm(traj_T2 - goal_xy.unsqueeze(0), dim=-1)  # (T,)
        rho_goal_t = goal_r - d_goal                                        # (T,)
        F_reach = rho_goal_t.max()

        return torch.minimum(G_avoid, F_reach)

    # ---------- Produce u_seq exactly once, based on baseline ----------
    N = x0.shape[0]  # batch size
    t0 = time.time()

    if args.baseline == 'opt':
        from optimize_stl_annealed import optimize_controls_annealed

        opt_count = N if (args.opt_max is None or args.opt_max <= 0) else min(args.opt_max, N)
        u_seq = torch.zeros((N, args.T, 2), device=device)

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

        if opt_count < N:
            from guidance import add_goal_bias
            u_fallback = sample_with_ddpm_or_fallback(N - opt_count, args.T, device=device, max_speed=1.6)
            u_fallback = add_goal_bias(x0[opt_count:], u_fallback, goal_xy, dt=args.dt, beta=0.35, max_speed=1.6)
            u_seq[opt_count:] = u_fallback

    elif args.baseline == 'opt_from_rrt':
        from rrtgeom import plan_rrt_star
        # safer helpers we control
        try:
            from rrt_bootstrap import path_to_controls as _bootstrap_path_to_controls
        except Exception:
            _bootstrap_path_to_controls = None

        from optimize_stl_annealed import optimize_controls_annealed
        max_speed = 1.6
        u_seq = torch.zeros((N, args.T, 2), device=device)
        opt_count = N if (args.opt_max is None or args.opt_max <= 0) else min(args.opt_max, N)

        # call wrapper so we tolerate different signatures
        def _path_to_controls_wrapper(path, T, dt, max_speed, device):
            if _bootstrap_path_to_controls is not None:
                return _bootstrap_path_to_controls(path, T=T, dt=dt, max_speed=max_speed, device=device)
            # fall back to rrtgeom if available
            try:
                from rrtgeom import path_to_controls as _rrt_path_to_controls
                try:
                    return _rrt_path_to_controls(path, T=T, dt=dt, max_speed=max_speed, device=device)
                except TypeError:
                    return _rrt_path_to_controls(path, T=T, dt=dt, max_speed=max_speed)
            except Exception:
                return None

        for i in range(opt_count):
            if (i % 25) == 0:
                print(f"[opt] optimizing start {i+1}/{opt_count}…", flush=True)

            # Pass numpy to avoid torch RNG/generator edge cases inside rrtgeom
            path = plan_rrt_star(
                x0=x0[i].detach().cpu().numpy(),
                goal_xy=goal_xy.detach().cpu().numpy(),
                obs_xy=obs_xy.detach().cpu().numpy(),
                obs_r=obs_r.detach().cpu().numpy(),
                max_iter=4000, step=0.30, r_rewire=0.60,
                goal_sample_rate=0.10, goal_thresh=float(goal_r.item()*0.9),
                xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0,
                seed=int(args.world_seed if args.world_seed is not None else 0),
            )

            init_u = None
            if path is not None and len(path) >= 2:
                init_u = _path_to_controls_wrapper(path, T=args.T, dt=args.dt, max_speed=max_speed, device=device)

            u_i, _ = optimize_controls_annealed(
                x0=x0[i], obs_xy=obs_xy, obs_r=obs_r,
                goal_xy=goal_xy, goal_r=goal_r,
                T=args.T, dt=args.dt, init_u=init_u,
                hard_eval_fn=_hard_rho_of,
                taus=(0.6, 0.3, 0.15, 0.08), iters_per_tau=80, lr=0.25, max_speed=max_speed
            )
            u_seq[i] = u_i.to(device)

        if opt_count < N:
            from guidance import add_goal_bias
            u_fallback = sample_with_ddpm_or_fallback(N - opt_count, args.T, device=device, max_speed=max_speed)
            u_fallback = add_goal_bias(x0[opt_count:], u_fallback, goal_xy, dt=args.dt, beta=0.35, max_speed=max_speed)
            u_seq[opt_count:] = u_fallback

    elif args.baseline == 'opt_basis_from_rrt':
        from rrtgeom import plan_rrt_star
        from rrt_bootstrap import path_to_knots
        from optimize_basis_bridge import optimize_basis_from_knots

        max_speed = 1.6
        u_seq = torch.zeros((N, args.T, 2), device=device)

        opt_count = N if (args.opt_max is None or args.opt_max <= 0) else min(args.opt_max, N)
        warm_hits, plan_times = 0, []

        K_knots = int(args.K_basis)

        for i in range(opt_count):
            if (i % 25) == 0:
                print(f"[opt_basis] optimizing start {i+1}/{opt_count}…", flush=True)

            t0p = time.time()
            # Pass CPU tensors to RRT* (safer with CPU-only ops inside rrtgeom.py)
            path = plan_rrt_star(
                x0=x0[i].cpu(),
                goal_xy=goal_xy.cpu(),
                obs_xy=obs_xy.cpu(),
                obs_r=obs_r.cpu(),
                max_iter=8000,
                step=0.25,
                # remove r_rewire if your plan_rrt_star does not support it
                # r_rewire=0.60,
                goal_sample_rate=0.20,
                xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0,
                goal_thresh=float(goal_r.item()),
                seed=int(args.world_seed if args.world_seed is not None else 0)
            )
            plan_times.append(time.time() - t0p)

            init_knots = None
            if path is not None and len(path) >= 2:
                init_knots = path_to_knots(path, K=K_knots, device=device)
                warm_hits += 1

            u_i, _ = optimize_basis_from_knots(
                x0=x0[i], obs_xy=obs_xy, obs_r=obs_r,
                goal_xy=goal_xy, goal_r=goal_r,
                K=K_knots, T=args.T, dt=args.dt,
                init_knots=init_knots,
                iters=int(args.iters_basis),
                lr=0.3, max_speed=max_speed,
                hard_eval_fn=_hard_rho_of
            )
            u_seq[i] = u_i.to(device)

        if opt_count > 0:
            import numpy as _np
            med_plan = float(_np.median(_np.array(plan_times))) if plan_times else 0.0
            print(
                f"[opt_basis_from_rrt] warm-starts used: {warm_hits}/{opt_count} "
                f"({100.0*warm_hits/max(1,opt_count):.1f}%) | median RRT* plan {med_plan:.3f}s",
                flush=True,
            )

        if opt_count < N:
            from guidance import add_goal_bias
            u_fallback = sample_with_ddpm_or_fallback(N - opt_count, args.T, device=device, max_speed=max_speed)
            u_fallback = add_goal_bias(x0[opt_count:], u_fallback, goal_xy, dt=args.dt, beta=0.35, max_speed=max_speed)
            u_seq[opt_count:] = u_fallback

    elif args.baseline == 'opt_basis_from_ddpm_hybrid':
        assert args.prior_ckpt is not None, "Provide --prior_ckpt"

        from prior_model import TinyUNet
        from prior_utils import encode_env, sample_knots_ddpm
        from optimize_basis_bridge import optimize_basis_from_knots
        from rrtgeom import plan_rrt_star
        from rrt_bootstrap import path_to_knots

        # --- load prior ---
        ckpt = torch.load(args.prior_ckpt, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)  # supports {'model':..., 'meta':...} or raw sd
        meta = ckpt.get("meta", {})

        # infer meta/params
        K_ckpt      = int(meta.get("K", args.K_basis))
        cond_dim    = int(meta.get("cond_dim", args.prior_cond_dim))
        Tdiff       = int(meta.get("Tdiff", 1000))
        hidden      = int(getattr(args, "prior_hidden", 256))
        time_in_dim = state_dict["time_mlp.0.weight"].shape[1] if "time_mlp.0.weight" in state_dict else 1

        # build exactly like training
        model = TinyUNet(in_dim=2*K_ckpt, cond_dim=cond_dim, hidden=hidden, time_in_dim=time_in_dim).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.Tdiff = Tdiff 

        # --- derive max_obs for encode_env exactly like training ---
        if "max_obs" in meta:
            max_obs = int(meta["max_obs"])
        else:
            # encode_env layout: 2 (x0) + 2 (goal) + 1 (goal_r) + 2*max_obs + max_obs + 1 (n_obs)
            max_obs = int((cond_dim - 6) // 3)
        max_obs = max(0, max_obs)

        from prior_utils import encode_env
        probe = encode_env(x0[0], obs_xy, obs_r, goal_xy, goal_r, max_obs=max_obs)
        assert probe.shape[-1] == cond_dim, f"encode_env produced {probe.shape[-1]} but checkpoint expects {cond_dim}. Fix max_obs/encode_env."
        print(f"[sanity] cond_dim ok: {probe.shape[-1]} == {cond_dim}")

        # --- runtime constants used below ---
        max_speed = 1.6
        u_seq = torch.zeros((N, args.T, 2), device=device)


        # logging counters
        ddpm_draws_total = 0
        ddpm_success     = 0
        fallback_calls   = 0
        fallback_success = 0
        fallback_times   = []

        opt_count = N if (args.opt_max is None or args.opt_max <= 0) else min(args.opt_max, N)
        
        steps = min(args.prior_steps, model.Tdiff) if args.prior_steps is not None else None


        for i in range(opt_count):
            if (i % 25) == 0:
                print(f"[hybrid] start {i+1}/{opt_count}…", flush=True)

            found = False
            cond_vec = encode_env(x0[i], obs_xy, obs_r, goal_xy, goal_r, max_obs=max_obs)

            for s in range(int(args.hybrid_ddpm_samples)):
                ddpm_draws_total += 1
                K_use = K_ckpt
                knots0 = sample_knots_ddpm(model, cond_vec, K=K_use, steps=steps, device=device)
                u_i, hard_rho = optimize_basis_from_knots(
                    x0=x0[i], obs_xy=obs_xy, obs_r=obs_r,
                    goal_xy=goal_xy, goal_r=goal_r,
                    K=K_use, T=args.T, dt=args.dt,
                    init_knots=knots0,
                    iters=int(args.hybrid_refine_iters), lr=0.3, max_speed=max_speed,
                    hard_eval_fn=_hard_rho_of
                )

                # 2) accept if good enough
                if float(hard_rho) > args.rho_min:
                    u_seq[i] = u_i
                    ddpm_success += 1
                    found = True
                    break

                # 3) optional micro-repair only for near-miss
                if getattr(args, "use_micro_repair", False) and (-0.03 <= float(hard_rho) <= args.rho_min):
                    repaired = micro_repair_knots(
                        knots0, x0=x0[i], obs_xy=obs_xy, obs_r=obs_r,
                        goal_xy=goal_xy, goal_r=goal_r, T=args.T, dt=args.dt,
                        steps=6, lr=0.08, trust_radius=0.15
                    )
                    u_i2, hard_rho2 = optimize_basis_from_knots(
                        x0=x0[i], obs_xy=obs_xy, obs_r=obs_r,
                        goal_xy=goal_xy, goal_r=goal_r,
                        K=int(args.K_basis), T=args.T, dt=args.dt,
                        init_knots=repaired,
                        iters=20, lr=0.3, max_speed=max_speed,
                        hard_eval_fn=_hard_rho_of
                    )
                    if float(hard_rho2) > args.rho_min:
                        u_seq[i] = u_i2
                        ddpm_success += 1
                        found = True
                        break






            if found:
                continue

            # ---- fallback: one RRT* with time/iter cap + refine ----
            fallback_calls += 1
            import time as _t
            t0p = _t.time()
            path = plan_rrt_star(
                x0=x0[i].cpu(), goal_xy=goal_xy.cpu(), obs_xy=obs_xy.cpu(), obs_r=obs_r.cpu(),
                max_iter=int(args.hybrid_rrt_iter), step=0.30, goal_sample_rate=0.20,
                xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0,
                goal_thresh=float(goal_r.item()),
                seed=int(args.world_seed if args.world_seed is not None else 0),
                max_time=float(args.hybrid_rrt_time)
            )
            fallback_times.append(_t.time() - t0p)

            init_knots = path_to_knots(path, K=int(args.K_basis), device=device) if (path is not None and len(path) >= 2) else None
            u_i, hard_rho = optimize_basis_from_knots(
                x0=x0[i], obs_xy=obs_xy, obs_r=obs_r,
                goal_xy=goal_xy, goal_r=goal_r,
                K=int(args.K_basis), T=args.T, dt=args.dt,
                init_knots=init_knots,
                iters=int(args.hybrid_rrt_refine_iters),
                lr=0.3, max_speed=max_speed,
                hard_eval_fn=_hard_rho_of
            )
            u_seq[i] = u_i
            if float(hard_rho) > args.rho_min:
                fallback_success += 1

        # fill remainder with cheap sampler if needed
        if opt_count < N:
            from guidance import add_goal_bias
            u_fallback = sample_with_ddpm_or_fallback(N - opt_count, args.T, device=device, max_speed=max_speed)
            u_fallback = add_goal_bias(x0[opt_count:], u_fallback, goal_xy, dt=args.dt, beta=0.35, max_speed=max_speed)
            u_seq[opt_count:] = u_fallback

        # stash hybrid stats for the logger below
        import numpy as _np
        locals().update(dict(
            _hy_ddpm_draws=ddpm_draws_total,
            _hy_ddpm_succ=ddpm_success,
            _hy_fb_calls=fallback_calls,
            _hy_fb_succ=fallback_success,
            _hy_fb_med=float(_np.median(_np.array(fallback_times))) if fallback_times else None,
        ))



    elif args.baseline == 'ours':
        from guidance import add_goal_bias
        u_seq = sample_with_ddpm_or_fallback(N, args.T, device=device, max_speed=1.6)
        u_seq = add_goal_bias(x0, u_seq, goal_xy, dt=args.dt, beta=0.35, max_speed=1.6)

    elif args.baseline == 'opt_basis_from_ddpm':
        assert args.prior_ckpt is not None, "Provide --prior_ckpt"

        # Build the same model class used in training
        from prior_model import TinyUNet

        # --- load checkpoint (wrapped with meta) ---
        ckpt = torch.load(args.prior_ckpt, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)  # handle {'model': ..., 'meta': ...} or raw sd
        meta = ckpt.get("meta", {})
        # fallbacks if meta was saved top-level (older script)
        for k in ["K", "cond_dim", "Tdiff", "max_obs", "time_in_dim"]:
            if k in ckpt and k not in meta:
                meta[k] = ckpt[k]

        # --- infer shapes/params from ckpt ---
        # time MLP input dim from first Linear in time_mlp
        time_in_dim = state_dict["time_mlp.0.weight"].shape[1] if "time_mlp.0.weight" in state_dict else 1
        K_ckpt      = int(meta.get("K", args.K_basis))
        cond_dim    = int(meta.get("cond_dim", args.prior_cond_dim))
        Tdiff       = int(meta.get("Tdiff", 1000))
        # --- derive max_obs for encode_env exactly like training ---
        if "max_obs" in meta:
            max_obs = int(meta["max_obs"])
        else:
            # encode_env layout: 2 (x0) + 2 (goal) + 1 (goal_r) + 2*max_obs + max_obs + 1 (n_obs)
            max_obs = int((cond_dim - 6) // 3)
        max_obs = max(0, max_obs)
        
        from prior_utils import encode_env
        probe = encode_env(x0[0], obs_xy, obs_r, goal_xy, goal_r, max_obs=max_obs)
        assert probe.shape[-1] == cond_dim, f"encode_env produced {probe.shape[-1]} but checkpoint expects {cond_dim}. Fix max_obs/encode_env."
        print(f"[sanity] cond_dim ok: {probe.shape[-1]} == {cond_dim}")

        # --- runtime constants *before* any use ---
        max_speed = 1.6
        u_seq = torch.zeros((N, args.T, 2), device=device)

        hidden = int(getattr(args, "prior_hidden", 256))

        # --- instantiate model to match ckpt ---
        in_dim = 2 * K_ckpt
        model = TinyUNet(in_dim=in_dim, cond_dim=cond_dim, hidden=hidden, time_in_dim=time_in_dim).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.Tdiff = Tdiff  # sampler uses this

        

        # --- runtime constants used below ---
        max_speed = 1.6
        u_seq = torch.zeros((N, args.T, 2), device=device)


        from optimize_basis_bridge import optimize_basis_from_knots
        from prior_utils import encode_env, sample_knots_ddpm

        max_speed = 1.6
        u_seq = torch.zeros((N, args.T, 2), device=device)

        opt_count = N if (args.opt_max is None or args.opt_max <= 0) else min(args.opt_max, N)
        steps = min(args.prior_steps, model.Tdiff) if args.prior_steps is not None else None

        for i in range(opt_count):
            if (i % 25) == 0:
                print(f"[opt_basis(ddpm)] optimizing start {i+1}/{opt_count}…", flush=True)

            # condition vector must match how the dataset was encoded
            cond_vec = encode_env(x0[i], obs_xy, obs_r, goal_xy, goal_r, max_obs=max_obs)

            # Sample K knots using the trained prior (use K from ckpt)
            K_use = K_ckpt
            knots0 = sample_knots_ddpm(model, cond_vec, K=K_use, steps=steps, device=device)
            u_i, hard_rho = optimize_basis_from_knots(
                x0=x0[i], obs_xy=obs_xy, obs_r=obs_r,
                goal_xy=goal_xy, goal_r=goal_r,
                K=K_use, T=args.T, dt=args.dt,
                init_knots=knots0,
                iters=int(args.hybrid_refine_iters), lr=0.3, max_speed=max_speed,
                hard_eval_fn=_hard_rho_of
            )
        

        # optional micro-repair if below threshold
        if float(hard_rho) <= args.rho_min and args.use_micro_repair:
            repaired = micro_repair_knots(
                knots0, x0=x0[i], obs_xy=obs_xy, obs_r=obs_r,
                goal_xy=goal_xy, goal_r=goal_r, T=args.T, dt=args.dt,
                steps=6, lr=0.08, trust_radius=0.15
            )
            u_i2, hard_rho2 = optimize_basis_from_knots(
                x0=x0[i], obs_xy=obs_xy, obs_r=obs_r,
                goal_xy=goal_xy, goal_r=goal_r,
                K=K_use, T=args.T, dt=args.dt,
                init_knots=repaired,
                iters=20, lr=0.3, max_speed=max_speed,
                hard_eval_fn=_hard_rho_of
            )
            if float(hard_rho2) > args.rho_min:
                u_i = u_i2  # adopt repaired solution


        u_seq[i] = u_i.to(device)


        if opt_count < N:
            from guidance import add_goal_bias
            u_fallback = sample_with_ddpm_or_fallback(N - opt_count, args.T, device=device, max_speed=max_speed)
            u_fallback = add_goal_bias(x0[opt_count:], u_fallback, goal_xy, dt=args.dt, beta=0.35, max_speed=max_speed)
            u_seq[opt_count:] = u_fallback



    else:
        raise NotImplementedError(f"baseline {args.baseline} not wired yet")

    # ---------- Simulate & Evaluate ----------
    traj = env.simulate(x0, u_seq)
    t1 = time.time()

    if args.spec == 'gf_avoid_reach':
        rho = chunked_eval(
            spec_G_avoid_and_F_reach, traj, obs_xy, obs_r, goal_xy, goal_r, chunk=args.chunk
        )
    else:
        a1, b1 = int(0), int(args.T - 1)
        a2, b2 = int(0.3 * args.T), int(0.7 * args.T)
        rho = chunked_eval(
            spec_bounded_avoid_then_reach, traj, obs_xy, obs_r, goal_xy, goal_r, a1, b1, a2, b2, chunk=args.chunk
        )

    passrate = (rho > 0).float().mean().item()

    # ---------- δ-SMT gate on top-K ----------
    mask = rho > args.rho_min
    idx_sorted = torch.argsort(rho, descending=True)
    top_idx = [i.item() for i in idx_sorted if mask[i]][: args.K]

    acc = rej = near = 0
    backend = args.cert

    for i in top_idx:
        r = float(rho[i].item())
        if r < 0:
            rej += 1
            continue

        x0_i      = x0[i].detach().cpu().numpy().tolist()
        u_i       = u_seq[i].detach().cpu().numpy().tolist()
        obs_xy_i  = obs_xy.detach().cpu().numpy().tolist()
        obs_r_i   = obs_r.detach().cpu().numpy().tolist()
        goal_xy_i = goal_xy.detach().cpu().numpy().tolist()
        goal_r_i  = float(goal_r.item())

        ok = True
        try:
            if backend == 'continuous':
                from verify_continuous import continuous_check_traj
                ok, _why = continuous_check_traj(x0_i, u_i, args.dt, obs_xy_i, obs_r_i, goal_xy_i, goal_r_i)
            elif backend == 'z3':
                from verify_z3 import z3_check_traj
                ok = z3_check_traj(x0_i, u_i, args.dt, obs_xy_i, obs_r_i, goal_xy_i, goal_r_i)
            elif backend == 'dreal':
                from verify_dreal import delta_sat_check
                ok = delta_sat_check(
                    x0_i, u_i, args.dt, obs_xy_i, obs_r_i, goal_xy_i, goal_r_i, delta=args.delta, dreal_path="dreal"
                )
            else:
                ok = True
        except Exception:
            # Treat as "near" if the solver isn't available or crashes
            ok = False

        if ok:
            acc += 1
        else:
            near += 1

    # --- best traj + throughput ---
    best_i   = int(idx_sorted[0].item())
    best_traj = traj[best_i].detach().cpu().numpy()
    best_rho = float(rho[best_i].item())
    gpu_traj_s = float(N) / max(1e-6, (t1 - t0))   # <-- now it's defined

    # --- logging (after we have acc/rej/near + gpu_traj_s) ---
    import numpy as _np
    _warm_hits   = locals().get("warm_hits", None)
    _plan_times  = locals().get("plan_times", None)
    _median_rrt  = float(_np.median(_np.array(_plan_times))) if _plan_times else None
    _warm_hits_pct = (100.0 * _warm_hits / max(1, locals().get("opt_count", 0))) if _warm_hits is not None else None

    rec = dict(
        task=args.task, spec=args.spec, baseline=args.baseline, device=str(device),
        N=N, T=args.T, dt=args.dt, num_obstacles=args.num_obstacles,
        passrate=passrate, success=passrate, best_rho=best_rho,
        gpu_traj_s=gpu_traj_s, rho_min=args.rho_min, K=args.K, delta=args.delta,
        accept_rate=acc / max(1, len(top_idx)),
        reject_rate=rej / max(1, N),
        near_feasible=near / max(1, len(top_idx)),
        seed=args.seed, world_seed=world_seed_used,
        goal_r=float(goal_r.item()), min_start_goal_dist=2.0, los_block_required=True,
        opt_max=getattr(args, "opt_max", None),
        K_basis=getattr(args, "K_basis", None),
        iters_basis=getattr(args, "iters_basis", None),
        warm_hits=_warm_hits, warm_hits_pct=_warm_hits_pct,
        median_rrt_plan_s=_median_rrt,
    )

    rec.update(dict(
        hy_ddpm_draws = locals().get('_hy_ddpm_draws', None),
        hy_ddpm_succ  = locals().get('_hy_ddpm_succ', None),
        hy_fb_calls   = locals().get('_hy_fb_calls', None),
        hy_fb_succ    = locals().get('_hy_fb_succ', None),
        hy_fb_med_s   = locals().get('_hy_fb_med', None),
    ))

    rec.update(dict(
        ddpm_win_rate = (locals().get('_hy_ddpm_succ', 0) / max(1, locals().get('_hy_ddpm_draws', 0))),
        fb_call_rate  = (locals().get('_hy_fb_calls', 0) / max(1, locals().get('opt_count', 1))),
        fb_success_rate = (locals().get('_hy_fb_succ', 0) / max(1, locals().get('_hy_fb_calls', 0))),
        ))
    append_record("outputs/logs/records.csv", rec)

    # --- plot + prints (unchanged) ---
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
    p.add_argument('--dreal', action='store_true', help='use dReal δ-SMT gate instead of stub')  # kept for CLI parity
    p.add_argument('--smt', type=str, default='none', choices=['none', 'z3', 'dreal'],
                   help='choose SMT backend: none | z3 | dreal (dReal requires Linux/WSL)')
    p.add_argument('--cert', type=str, default='continuous',
                   choices=['none', 'continuous', 'z3', 'dreal'],
                   help='final certification backend')
    p.add_argument('--use_micro_repair', action='store_true')
    p.add_argument('--baseline', type=str, default='ours',
                    choices=['ours','opt','rrt','opt_from_rrt','opt_from_ddpm','opt_basis_from_rrt','opt_basis_from_ddpm', 'opt_basis_from_ddpm_hybrid'],
                    help='pipeline to run')

    p.add_argument('--opt_max', type=int, default=None,
                   help='If set, only optimize this many starts; others use sampler.')
    p.add_argument('--batch_opt', type=int, default=32,
                   help='Batch width for the (optional) batched optimizer path.')

    # basis controls
    p.add_argument('--K_basis', type=int, default=8,
                   help='Number of B-spline knots for basis optimization.')
    p.add_argument('--iters_basis', type=int, default=200,
                   help='Iterations per tau for basis optimizer.')
    # in __main__ argparse section
    p.add_argument('--prior_ckpt', type=str, default=None, help='Path to trained prior .pt')
    p.add_argument('--prior_steps', type=int, default=None, help='Override diffusion steps for sampling')
    p.add_argument('--prior_cond_dim', type=int, default=30)
    p.add_argument('--prior_hidden', type=int, default=256)
    p.add_argument('--hybrid_ddpm_samples', type=int, default=4, help='DDPM knot draws per start before fallback')
    p.add_argument('--hybrid_rrt_time', type=float, default=0.6, help='Seconds cap for RRT* in hybrid fallback')
    p.add_argument('--hybrid_rrt_iter', type=int, default=3000, help='Iter cap for RRT* in hybrid fallback')
    p.add_argument('--hybrid_refine_iters', type=int, default=60, help='Basis iters per DDPM draw in hybrid')
    p.add_argument('--hybrid_rrt_refine_iters', type=int, default=80, help='Basis iters after fallback RRT*')


    args = p.parse_args()
    main(args)
