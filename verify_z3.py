# verify_z3.py
from z3 import Solver, RealVal, sat

def _R(x):
    # Make exact rationals (strings avoid FP rounding)
    if isinstance(x, float):
        return RealVal(repr(x))
    return RealVal(x)

def z3_check_traj(x0, u, dt, obs_xy, obs_r, goal_xy, goal_r):
    """
    Exact SMT check over the discretized trajectory.
    No symbolic vars: we rebuild x_t from constants and assert inequalities.
    Returns True iff all avoid constraints hold for all t and goal holds for some t.
    """
    T = len(u)
    # Reconstruct states as exact reals
    x = [_R(x0[0])]
    y = [_R(x0[1])]
    dtR = _R(dt)

    for t in range(T):
        ux, uy = u[t]
        x.append(x[-1] + _R(ux) * dtR)
        y.append(y[-1] + _R(uy) * dtR)

    s = Solver()

    # Always-avoid:  (x_t - cx)^2 + (y_t - cy)^2 >= r^2  for all t
    avoid = True
    for t in range(T + 1):
        for (cx, cy), r in zip(obs_xy, obs_r):
            dx = x[t] - _R(cx)
            dy = y[t] - _R(cy)
            avoid = avoid & ((dx*dx + dy*dy) >= (_R(r) * _R(r)))
    s.add(avoid)

    # Eventually reach: exists t s.t. distance to goal <= goal_r
    reach_disj = False
    for t in range(T + 1):
        gx = x[t] - _R(goal_xy[0])
        gy = y[t] - _R(goal_xy[1])
        reach_disj = reach_disj | ((gx*gx + gy*gy) <= (_R(goal_r) * _R(goal_r)))
    s.add(reach_disj)

    return s.check() == sat
