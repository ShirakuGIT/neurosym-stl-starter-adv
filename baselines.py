from rrt_timeparam import rrt_path_to_controls_feasible

def rrtstar_trajectory_feasible(x0, goal_xy, goal_r, obs_xy, obs_r, T=64, dt=0.1, bounds=(-5,5,-5,5)):
    obstacles = [(float(cx), float(cy), float(r)) for (cx,cy), r in zip(obs_xy.cpu().numpy(), obs_r.cpu().numpy())]
    rrt = RRTStar2D(bounds=bounds, obstacles=obstacles, step=0.25, radius=0.75, iters=5000)
    path = rrt.plan(tuple(x0.cpu().tolist()), tuple(goal_xy.cpu().tolist()), float(goal_r.item()))
    if path is None: return None
    return torch.from_numpy(rrt_path_to_controls_feasible(path, T=T, dt=dt)).float()
