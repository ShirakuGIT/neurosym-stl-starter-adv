import numpy as np, torch
from rrtstar import RRTStar2D

def rrtstar_trajectory(x0, goal_xy, goal_r, obs_xy, obs_r, T=64, dt=0.1, bounds=(-5,5,-5,5)):
    obstacles = [(float(cx), float(cy), float(r)) for (cx,cy), r in zip(obs_xy.detach().cpu().numpy(), obs_r.detach().cpu().numpy())]
    rrt = RRTStar2D(bounds=bounds, obstacles=obstacles, step=0.25, radius=0.75, iters=5000)
    path = rrt.plan(tuple(x0.detach().cpu().tolist()), tuple(goal_xy.detach().cpu().tolist()), float(goal_r.item()))
    if path is None: return None
    # resample to T waypoints (linear interpolation), then infer controls
    pts = np.array(path)
    # arc-length parameterize
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1] if s[-1] > 1e-6 else 1.0
    s_target = np.linspace(0, total, T)
    px = np.interp(s_target, s, pts[:,0]); py = np.interp(s_target, s, pts[:,1])
    traj = np.stack([px,py], axis=1)  # (T,2)
    # build control by finite diff
    u = np.zeros_like(traj)
    u[1:] = (traj[1:] - traj[:-1]) / dt
    return torch.from_numpy(u).float()  # (T,2)
