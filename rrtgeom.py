# rrtgeom.py
import time
from typing import List, Tuple, Optional, Sequence

import torch


# ----------------------------- Collision helpers -----------------------------

def _normalize_obstacles(
    obs_xy: torch.Tensor, obs_r: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make sure obstacles are (nObs, 2) and radii are (nObs,).
    If empty or higher-rank, coerce safely.
    """
    if obs_xy is None or obs_r is None:
        return torch.empty(0, 2, device=device, dtype=dtype), torch.empty(0, device=device, dtype=dtype)

    if not isinstance(obs_xy, torch.Tensor):
        obs_xy = torch.as_tensor(obs_xy, device=device, dtype=dtype)
    if not isinstance(obs_r, torch.Tensor):
        obs_r = torch.as_tensor(obs_r, device=device, dtype=dtype)

    # If higher rank (e.g., batched), flatten the leading dims
    if obs_xy.ndim > 2:
        obs_xy = obs_xy.reshape(-1, obs_xy.shape[-1])
    if obs_r.ndim > 1:
        obs_r = obs_r.reshape(-1)

    # Coerce to correct shapes
    if obs_xy.numel() == 0:
        obs_xy = torch.empty(0, 2, device=device, dtype=dtype)
        obs_r = torch.empty(0, device=device, dtype=dtype)
    else:
        obs_xy = obs_xy.to(device=device, dtype=dtype)
        obs_r = obs_r.to(device=device, dtype=dtype)
        if obs_xy.shape[-1] != 2:
            obs_xy = obs_xy.reshape(-1, 2)
        if obs_r.shape[0] != obs_xy.shape[0]:
            # If mismatch, best-effort trim/pad radii
            n = obs_xy.shape[0]
            if obs_r.numel() >= n:
                obs_r = obs_r[:n]
            else:
                pad = torch.zeros(n - obs_r.numel(), device=device, dtype=dtype)
                obs_r = torch.cat([obs_r, pad], dim=0)

    return obs_xy, obs_r


def collides_point(
    p: torch.Tensor,
    obs_xy: torch.Tensor,
    obs_r: torch.Tensor,
    margin: float = 0.0,
) -> bool:
    """
    Return True if point p is inside any (expanded) obstacle.
    """
    if obs_xy.numel() == 0:
        return False
    d = torch.linalg.norm(obs_xy - p.unsqueeze(0), dim=-1)  # (nObs,)
    return bool((d <= (obs_r + margin)).any().item())


def collides_segment(
    p: torch.Tensor,
    q: torch.Tensor,
    obs_xy: torch.Tensor,
    obs_r: torch.Tensor,
    margin: float = 0.02,
    ncheck: int = 16,
) -> bool:
    """
    Sampled segment collision check (ncheck points, inclusive of endpoints).
    """
    if obs_xy.numel() == 0:
        return False
    ncheck = max(2, int(ncheck))
    ts = torch.linspace(0, 1, ncheck, device=p.device, dtype=p.dtype).unsqueeze(-1)  # (ncheck,1)
    pts = p.unsqueeze(0) * (1 - ts) + q.unsqueeze(0) * ts  # (ncheck,2)
    # Vectorized early-out: check all samples
    # (Use loop if you prefer true short-circuit; this is usually fast enough.)
    for i in range(ncheck):
        if collides_point(pts[i], obs_xy, obs_r, margin):
            return True
    return False


# ----------------------------- Path utilities --------------------------------

def path_to_knots(path: List[Sequence[float]], K: int) -> torch.Tensor:
    """
    Choose K roughly equally spaced points along a polyline path [(x,y), ...].
    Returns (K,2) tensor of knot positions (CPU; move to device as needed).
    """
    K = max(2, int(K))
    pts = torch.as_tensor(path, dtype=torch.float32)  # (M,2)
    if pts.ndim == 1:
        pts = pts.view(-1, 2)
    M = pts.shape[0]
    if M <= 0:
        return torch.zeros(K, 2, dtype=torch.float32)
    if M == 1:
        return pts.repeat(K, 1)

    # Cumulative arc-length
    seg = pts[1:] - pts[:-1]                 # (M-1,2)
    seg_len = torch.linalg.norm(seg, dim=-1) # (M-1,)
    s = torch.cat([torch.zeros(1), torch.cumsum(seg_len, dim=0)], dim=0)  # (M,)
    total = float(s[-1].item()) if s[-1] > 0 else 1.0

    # Target arc-lengths
    target = torch.linspace(0.0, total, steps=K)
    knots = torch.zeros(K, 2, dtype=torch.float32)

    j = 0
    for i in range(K):
        si = target[i]
        while (j + 1) < s.numel() and s[j + 1] < si:
            j += 1
        if (j + 1) >= s.numel():
            knots[i] = pts[-1]
        else:
            denom = max((s[j + 1] - s[j]).item(), 1e-9)
            t = float((si - s[j]).item() / denom)
            knots[i] = pts[j] + t * (pts[j + 1] - pts[j])

    return knots


def path_to_controls(
    path: Optional[List[Tuple[float, float]]],
    *,
    T: int,
    dt: float,
    max_speed: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert a geometric path to a constant-speed control sequence (T,2).
    """
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
        u[t:t + n_write] = v
        t += n_write
    return u


# --------------------------------- RRT* --------------------------------------

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
    max_time: Optional[float] = None,   # seconds – stop early if exceeded
    rewire_k: int = 24,                 # cap rewire neighborhood by K nearest
    ncheck: int = 8,                    # segment collision samples
    r_rewire: float = 0.0,              # optional radius filter; 0 disables
    goal_thresh: float = 0.5,
    xmin: float = -5.0, xmax: float = 5.0,
    ymin: float = -5.0, ymax: float = 5.0,
    goal_sample_rate: float = 0.10,
    seed: Optional[int] = 0,
) -> Optional[List[Tuple[float, float]]]:
    """
    Minimal, robust RRT*. Returns a list of (x,y) or None.
    - Honors time cap (max_time)
    - Uses K-nearest rewiring (rewire_k), optionally filtered by r_rewire > 0
    - All tensors must be on the same device/dtype; obstacles are normalized
    """
    assert isinstance(x0, torch.Tensor) and isinstance(goal_xy, torch.Tensor)
    assert isinstance(obs_xy, torch.Tensor) and isinstance(obs_r, torch.Tensor)

    device = x0.device
    dtype = x0.dtype
    obs_xy, obs_r = _normalize_obstacles(obs_xy, obs_r, device, dtype)

    # RNG on CPU for portability
    gen = torch.Generator(device="cpu")
    if seed is None:
        seed = int(time.time() * 1e6) & 0x7FFFFFFF
    gen.manual_seed(int(seed))

    nodes: List[_Node] = [
        _Node(x0.detach().clone().to(device=device, dtype=dtype), parent=None, cost=0.0)
    ]

    def _sample() -> torch.Tensor:
        # Uniform sample with occasional goal bias
        try:
            p = torch.rand((), generator=gen).item()
            xr = torch.empty((), generator=gen).uniform_(xmin, xmax).item()
            yr = torch.empty((), generator=gen).uniform_(ymin, ymax).item()
        except TypeError:
            # Fallback if generator kwarg isn’t supported by this PyTorch build
            p = torch.rand(()).item()
            xr = torch.empty(()).uniform_(xmin, xmax).item()
            yr = torch.empty(()).uniform_(ymin, ymax).item()
        if p < goal_sample_rate:
            return goal_xy
        return torch.tensor([xr, yr], device=device, dtype=dtype)

    def _nearest(q: torch.Tensor) -> int:
        pts = torch.stack([n.pt for n in nodes], dim=0)  # (N,2)
        d2 = torch.sum((pts - q.unsqueeze(0)) ** 2, dim=-1)
        return int(torch.argmin(d2).item())

    t_start = time.time()
    rewire_k = max(1, int(rewire_k))
    ncheck = max(2, int(ncheck))

    for _ in range(int(max_iter)):
        if (max_time is not None) and (time.time() - t_start > max_time):
            break

        q_rand = _sample()
        j = _nearest(q_rand)              # index of nearest
        q_near = nodes[j].pt
        v = q_rand - q_near
        dist = torch.linalg.norm(v).clamp_min(1e-9)
        q_new = q_near + v * (float(step) / dist)

        if collides_segment(q_near, q_new, obs_xy, obs_r, margin=0.02, ncheck=ncheck):
            continue

        # --- Choose parent (K-nearest + optional radius filter) ---
        pts_all = torch.stack([n.pt for n in nodes], dim=0)  # (N,2)
        dists = torch.linalg.norm(pts_all - q_new.unsqueeze(0), dim=-1)  # (N,)
        k = min(rewire_k, dists.numel())
        nbr_idx = torch.topk(dists, k=k, largest=False).indices.tolist()
        if r_rewire > 0.0:
            nbr_idx = [ii for ii in nbr_idx if float(dists[ii].item()) <= r_rewire]
            if not nbr_idx:
                nbr_idx = [j]  # at least nearest

        best_parent = j
        best_cost = nodes[j].cost + float(torch.linalg.norm(q_new - nodes[j].pt).item())
        for k_idx in nbr_idx:
            if collides_segment(nodes[k_idx].pt, q_new, obs_xy, obs_r, margin=0.02, ncheck=ncheck):
                continue
            cand_cost = nodes[k_idx].cost + float(torch.linalg.norm(q_new - nodes[k_idx].pt).item())
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_parent = k_idx

        idx_new = len(nodes)
        nodes.append(_Node(q_new, parent=best_parent, cost=best_cost))

        # --- Rewire neighbors to new node if beneficial ---
        for k_idx in nbr_idx:
            if k_idx == best_parent:
                continue
            if collides_segment(nodes[k_idx].pt, q_new, obs_xy, obs_r, margin=0.02, ncheck=ncheck):
                continue
            new_cost = best_cost + float(torch.linalg.norm(nodes[k_idx].pt - q_new).item())
            if new_cost + 1e-9 < nodes[k_idx].cost:
                nodes[k_idx].parent = idx_new
                nodes[k_idx].cost = new_cost

        # --- Goal check & backtrack path ---
        if torch.linalg.norm(q_new - goal_xy) <= goal_thresh:
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
