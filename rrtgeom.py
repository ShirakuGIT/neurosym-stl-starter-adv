# rrtgeom.py
import torch
from typing import List, Tuple, Optional

def collides_point(p: torch.Tensor, obs_xy: torch.Tensor, obs_r: torch.Tensor, margin: float = 0.0) -> bool:
    d = torch.linalg.norm(obs_xy - p.unsqueeze(0), dim=-1)  # (nObs,)
    return bool((d <= (obs_r + margin)).any().item())

def collides_segment(p: torch.Tensor, q: torch.Tensor, obs_xy: torch.Tensor, obs_r: torch.Tensor,
                     margin: float = 0.02, ncheck: int = 16) -> bool:
    ts = torch.linspace(0, 1, ncheck, device=p.device, dtype=p.dtype).unsqueeze(-1)
    pts = p.unsqueeze(0) * (1 - ts) + q.unsqueeze(0) * ts  # (ncheck,2)
    for i in range(ncheck):
        if collides_point(pts[i], obs_xy, obs_r, margin):
            return True
    return False

class _Node:
    __slots__ = ("pt", "parent", "cost")
    def __init__(self, pt: torch.Tensor, parent: Optional[int], cost: float):
        self.pt = pt
        self.parent = parent  # int or None
        self.cost = cost

def plan_rrt_star(
    x0: torch.Tensor,
    goal_xy: torch.Tensor,
    obs_xy: torch.Tensor,
    obs_r: torch.Tensor,
    *,
    max_iter: int = 4000,
    step: float = 0.30,
    r_rewire: float = 0.60,
    goal_thresh: float = 0.5,
    xmin: float = -5.0, xmax: float = 5.0,
    ymin: float = -5.0, ymax: float = 5.0,
    goal_sample_rate: float = 0.10,
    seed: int = 0,
) -> Optional[List[Tuple[float, float]]]:
    """
    Minimal, robust RRT*. Returns a list of (x,y) or None.
    All tensors should be on same device (CPU or CUDA) and dtype.
    """
    assert isinstance(x0, torch.Tensor) and isinstance(goal_xy, torch.Tensor)
    assert isinstance(obs_xy, torch.Tensor) and isinstance(obs_r, torch.Tensor)
    device = x0.device
    dtype = x0.dtype

    # RNG lives on CPU for portability
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    nodes: List[_Node] = [_Node(x0.detach().clone().to(device=device, dtype=dtype), parent=None, cost=0.0)]

    def _sample() -> torch.Tensor:
        try:
            p = torch.rand(1, generator=gen).item()
            xr = torch.empty(1, generator=gen).uniform_(xmin, xmax).item()
            yr = torch.empty(1, generator=gen).uniform_(ymin, ymax).item()
        except TypeError:
            # Fallback if generator kwarg isn’t supported on this op
            p  = torch.rand(1).item()
            xr = torch.empty(1).uniform_(xmin, xmax).item()
            yr = torch.empty(1).uniform_(ymin, ymax).item()
        if p < goal_sample_rate:
            return goal_xy
        return torch.tensor([xr, yr], device=device, dtype=dtype)



    def _nearest(q: torch.Tensor) -> int:
        pts = torch.stack([n.pt for n in nodes], dim=0)  # (N,2)
        d2  = torch.sum((pts - q.unsqueeze(0))**2, dim=-1)
        return int(torch.argmin(d2).item())

    for _ in range(max_iter):
        q_rand = _sample()
        j = _nearest(q_rand)                  # int
        q_near = nodes[j].pt
        v = q_rand - q_near
        dist = torch.linalg.norm(v).clamp_min(1e-6)
        q_new = q_near + v * (step / dist)

        if collides_segment(q_near, q_new, obs_xy, obs_r, margin=0.02, ncheck=16):
            continue

        # choose parent in neighborhood
        pts = torch.stack([n.pt for n in nodes], dim=0)
        dists = torch.linalg.norm(pts - q_new.unsqueeze(0), dim=-1)
        nbrs_idx = torch.where(dists < r_rewire)[0].tolist()
        if len(nbrs_idx) == 0:
            nbrs_idx = [j]

        best_parent = j
        best_cost = nodes[j].cost + float(torch.linalg.norm(q_new - nodes[j].pt).item())
        for k in nbrs_idx:
            # k is an int index
            if collides_segment(nodes[k].pt, q_new, obs_xy, obs_r, margin=0.02, ncheck=16):
                continue
            cost_k = nodes[k].cost + float(torch.linalg.norm(q_new - nodes[k].pt).item())
            if cost_k < best_cost:
                best_cost = cost_k
                best_parent = k

        idx_new = len(nodes)
        nodes.append(_Node(q_new, parent=best_parent, cost=best_cost))

        # rewire
        for k in nbrs_idx:
            if k == best_parent:
                continue
            if collides_segment(nodes[k].pt, q_new, obs_xy, obs_r, margin=0.02, ncheck=16):
                continue
            new_cost = best_cost + float(torch.linalg.norm(nodes[k].pt - q_new).item())
            if new_cost + 1e-9 < nodes[k].cost:
                nodes[k].parent = idx_new
                nodes[k].cost = new_cost

        # goal check
        if torch.linalg.norm(q_new - goal_xy) <= goal_thresh:
            # backtrack parents (guard against None)
            path: List[Tuple[float, float]] = []
            cur: Optional[int] = idx_new
            visited = 0
            while cur is not None and visited <= len(nodes):
                p = nodes[cur].pt
                path.append((float(p[0].item()), float(p[1].item())))
                cur = nodes[cur].parent
                visited += 1
            path.reverse()
            if len(path) >= 2:
                return path

    return None

def path_to_controls(
    path: Optional[List[Tuple[float, float]]],
    *,
    T: int,
    dt: float,
    max_speed: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    u = torch.zeros(T, 2, device=device, dtype=torch.float32)
    if path is None or len(path) < 2:
        return u

    pts = torch.tensor(path, device=device, dtype=torch.float32)  # (M,2)
    segs = pts[1:] - pts[:-1]                                     # (M-1,2)
    seg_d = torch.linalg.norm(segs, dim=-1).clamp_min(1e-6)       # (M-1,)
    seg_v = segs / seg_d.unsqueeze(-1) * max_speed                # (M-1,2)

    steps_per_seg = torch.clamp((seg_d / (max_speed * dt)).round().to(torch.int64), min=1)
    t = 0
    for v, n in zip(seg_v, steps_per_seg):
        n = int(n.item())
        if t >= T:
            break
        n_write = min(n, T - t)
        u[t:t+n_write] = v
        t += n_write
    return u
