# rrt_bootstrap.py
import torch

def resample_polyline(points_xy, T, dt, max_speed=1.6, device="cpu"):
    """
    points_xy: (M,2) tensor or list of (x,y) waypoints from RRT* (start->goal).
    Returns u_init: (T,2) control sequence that roughly tracks the polyline
    with piecewise-constant velocity, speed-limited by max_speed.
    """
    if not torch.is_tensor(points_xy):
        points_xy = torch.tensor(points_xy, dtype=torch.float32)
    points_xy = points_xy.to(device)

    seg_vecs = points_xy[1:] - points_xy[:-1]        # (M-1,2)
    seg_lens = torch.linalg.norm(seg_vecs, dim=-1)   # (M-1,)
    total_len = torch.clamp(seg_lens.sum(), min=1e-6)

    # Constant-speed allocation of T steps across segments
    steps_total = T
    steps_per_seg = torch.clamp((seg_lens / total_len * steps_total).round().long(), min=1)
    # adjust to match exactly T
    diff = steps_total - int(steps_per_seg.sum().item())
    if diff != 0:
        # tweak largest segments first
        order = torch.argsort(-seg_lens)
        for k in order:
            if diff == 0: break
            steps_per_seg[k] = steps_per_seg[k] + (1 if diff > 0 else -1)
            diff += (-1 if diff > 0 else 1)

    # Build piecewise-constant velocity controls
    u = torch.zeros((T, 2), device=device)
    t = 0
    for v, n in zip(seg_vecs, steps_per_seg):
        if t >= T: break
        dir_v = v / (torch.linalg.norm(v) + 1e-9)
        # desired speed: move segment in n steps (distance ≈ |v|, time ≈ n*dt)
        speed = (torch.linalg.norm(v) / (n * dt + 1e-9)).clamp(max=max_speed)
        u_seg = dir_v * speed
        n_int = int(n.item()) if torch.is_tensor(n) else int(n)
        end = min(T, t + n_int)
        u[t:end] = u_seg
        t = end

    # If we have leftover steps, keep last control zero (already zero)
    return u

def path_to_controls(path_xy, T, dt, max_speed=1.6, device="cpu"):
    """
    path_xy: list/array of (x,y) nodes from RRT* (including start and goal).
    Wraps resample_polyline() for clarity.
    """
    return resample_polyline(path_xy, T=T, dt=dt, max_speed=max_speed, device=device)
