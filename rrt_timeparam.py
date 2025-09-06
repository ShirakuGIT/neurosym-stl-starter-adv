# rrt_timeparam.py
import numpy as np

def _arc_length_resample(path_xy, num_pts):
    pts = np.asarray(path_xy, dtype=float)
    if len(pts) < 2:
        return np.repeat(pts[:1], num_pts, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1] if s[-1] > 1e-6 else 1.0
    s_target = np.linspace(0.0, total, num_pts)
    px = np.interp(s_target, s, pts[:,0])
    py = np.interp(s_target, s, pts[:,1])
    return np.stack([px,py], axis=1)

def _smooth_path(p, k=3):
    # simple moving-average smoothing to limit sharp corners
    if k <= 1: return p
    padl = np.repeat(p[:1], k//2, axis=0)
    padr = np.repeat(p[-1:], k - k//2 - 1, axis=0)
    q = np.concatenate([padl, p, padr], axis=0)
    w = np.ones(k)/k
    sm = np.stack([np.convolve(q[:,0], w, mode='valid'), np.convolve(q[:,1], w, mode='valid')], axis=1)
    return sm

def rrt_path_to_controls_feasible(path_xy, T, dt, max_speed=1.6, smooth_k=5):
    """
    Convert a geometric path into a speed-limited, corner-smoothed trajectory with T waypoints.
    Then return controls u (T,2) by finite-difference.
    """
    dense = _arc_length_resample(path_xy, num_pts=max(8*T, 256))
    dense = _smooth_path(dense, k=smooth_k)

    # target constant speed along arc length
    total_len = np.sum(np.linalg.norm(np.diff(dense, axis=0), axis=1))
    target_speed = min(max_speed*0.9, total_len / (T*dt) + 1e-6)

    # time-parameterize by arc length at target_speed
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    t_total = max(total_len / target_speed, T*dt)
    t_dense = s / max(total_len,1e-6) * t_total
    t_target = np.linspace(0.0, t_total, T)
    px = np.interp(t_target, t_dense, dense[:,0])
    py = np.interp(t_target, t_dense, dense[:,1])
    traj = np.stack([px,py], axis=1)

    # controls
    u = np.zeros_like(traj)
    u[1:] = (traj[1:] - traj[:-1]) / dt

    # speed cap projection
    sp = np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    u = u * np.minimum(1.0, max_speed / sp)

    return u.astype(np.float32)  # (T,2)
