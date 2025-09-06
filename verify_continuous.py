# verify_continuous.py
import math

def _min_sqdist_segment_to_circle(p0, p1, c):
    """
    Returns the minimum squared distance from the segment [p0,p1] to circle center c.
    p0,p1,c: 2-tuples or lists (floats).
    """
    x0,y0 = p0; x1,y1 = p1; cx,cy = c
    vx, vy = (x1 - x0), (y1 - y0)
    wx, wy = (cx - x0), (cy - y0)
    vv = vx*vx + vy*vy
    if vv <= 1e-18:
        # Degenerate segment
        dx, dy = (x0 - cx), (y0 - cy)
        return dx*dx + dy*dy, 0.0
    t = (wx*vx + wy*vy) / vv
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    px = x0 + t * vx
    py = y0 + t * vy
    dx, dy = (px - cx), (py - cy)
    return dx*dx + dy*dy, t

def continuous_check_traj(x0, u, dt, obs_xy, obs_r, goal_xy, goal_r):
    """
    Certify the WHOLE polyline path (straight segments per step) for:
      - Always-avoid: for every segment and obstacle, min distance >= r
      - Eventually reach: exists waypoint or segment where distance <= goal_r
    Returns: (ok: bool, reason: str)
    """
    # Reconstruct waypoints
    T = len(u)
    xs = [float(x0[0])]; ys = [float(x0[1])]
    for t in range(T):
        ux, uy = u[t]
        xs.append(xs[-1] + float(ux) * float(dt))
        ys.append(ys[-1] + float(uy) * float(dt))

    # 1) Always-avoid on every segment
    for t in range(T):
        p0 = (xs[t], ys[t]); p1 = (xs[t+1], ys[t+1])
        for (cx,cy), r in zip(obs_xy, obs_r):
            d2, _ = _min_sqdist_segment_to_circle(p0, p1, (float(cx), float(cy)))
            if d2 < float(r)*float(r) - 1e-12:  # tiny epsilon for FP safety
                return False, f"collision on seg {t} with obstacle"

    # 2) Eventually reach: check waypoints AND interior of segments
    #    For the segment case, if min distance point lies inside (t in (0,1))
    #    and d2 <= goal_r^2, we count it as reached.
    gr2 = float(goal_r)*float(goal_r)
    # Waypoints
    for t in range(T+1):
        dx = xs[t] - float(goal_xy[0]); dy = ys[t] - float(goal_xy[1])
        if (dx*dx + dy*dy) <= gr2 + 1e-12:
            return True, "reached at waypoint"

    # Interior of segments
    for t in range(T):
        p0 = (xs[t], ys[t]); p1 = (xs[t+1], ys[t+1])
        d2, lam = _min_sqdist_segment_to_circle(p0, p1, (float(goal_xy[0]), float(goal_xy[1])))
        if 0.0 < lam < 1.0 and d2 <= gr2 + 1e-12:
            return True, "reached along segment"

    # If we get here: safe but never reached
    return False, "safe but did not reach"
