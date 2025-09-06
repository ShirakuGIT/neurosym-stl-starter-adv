# verify_z3.py
from z3 import Solver, RealVal, BoolVal, And, Or, sat

def _R(x):
    # Build exact rationals; repr(x) avoids binary FP rounding.
    return RealVal(repr(x)) if isinstance(x, float) else RealVal(x)

def z3_check_traj(x0, u, dt, obs_xy, obs_r, goal_xy, goal_r):
    """
    Exact SMT check over the discretized trajectory (no variables—only constants).
    Returns True iff:
      - for all t:  dist(x_t, c_i)^2 >= r_i^2  for every obstacle i
      - exists t:   dist(x_t, goal)^2 <= goal_r^2
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

    # Always-avoid constraints
    avoid_conds = []
    for t in range(T + 1):
        for (cx, cy), r in zip(obs_xy, obs_r):
            dx = x[t] - _R(cx)
            dy = y[t] - _R(cy)
            avoid_conds.append((dx*dx + dy*dy) >= (_R(r) * _R(r)))
    s.add(And(*avoid_conds) if avoid_conds else BoolVal(True))

    # Eventually reach constraints (disjunction over time)
    reach_conds = []
    for t in range(T + 1):
        gx = x[t] - _R(goal_xy[0])
        gy = y[t] - _R(goal_xy[1])
        reach_conds.append((gx*gx + gy*gy) <= (_R(goal_r) * _R(goal_r)))
    s.add(Or(*reach_conds) if reach_conds else BoolVal(False))

    return s.check() == sat
