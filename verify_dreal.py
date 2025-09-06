import tempfile, subprocess, os, textwrap

def _dr_vec(name, dim):
    return " ".join([f"{name}{i}" for i in range(dim)])

def build_dreal_model(x0, u, dt, obs_xy, obs_r, goal_xy, goal_r, delta=1e-2):
    """
    x0: (2,) list/tuple
    u:  (T,2) list of controls
    obs_xy: list of (2,) obstacle centers; obs_r: list of radii
    goal_xy: (2,), goal_r: float
    Returns: dreal .dr string
    """
    T = len(u)
    lines = []
    # Declare vars for all x_t
    for t in range(T+1):
        lines.append(f"(declare-fun x{t}0 () Real)")
        lines.append(f"(declare-fun x{t}1 () Real)")
    # Dynamics & box constraints (optional)
    for t in range(T):
        ux, uy = u[t]
        lines.append(f"(assert (= x{t+1}0 (+ x{t}0 (* {ux} {dt}))))")
        lines.append(f"(assert (= x{t+1}1 (+ x{t}1 (* {uy} {dt}))))")
    # Initial state
    lines.append(f"(assert (= x00 {x0[0]}))")
    lines.append(f"(assert (= x01 {x0[1]}))")
    # Always-avoid obstacles: for all t, distance >= r + 0 (δ handles slack)
    for t in range(T+1):
        for (cx, cy), r in zip(obs_xy, obs_r):
            lines.append(f"(assert (>= (+ (^ (- x{t}0 {cx}) 2) (^ (- x{t}1 {cy}) 2)) (^ {r} 2)))")
    # Eventually reach goal in window [0,T] (pick any t; disjunction)
    reach_disj = []
    for t in range(T+1):
        reach_disj.append(f"(<= (+ (^ (- x{t}0 {goal_xy[0]}) 2) (^ (- x{t}1 {goal_xy[1]}) 2)) (^ {goal_r} 2))")
    if reach_disj:
        lines.append("(assert (or " + " ".join(reach_disj) + "))")
    # Check-sat
    lines.append("(check-sat)")
    return "(set-logic QF_NRA)\n" + "\n".join(lines) + "\n"

def delta_sat_check(x0, u, dt, obs_xy, obs_r, goal_xy, goal_r, delta=1e-3, dreal_path="dreal"):
    model = build_dreal_model(x0, u, dt, obs_xy, obs_r, goal_xy, goal_r, delta=delta)
    with tempfile.TemporaryDirectory() as td:
        dr = os.path.join(td, "model.dr")
        with open(dr, "w") as f:
            f.write(model)
        # dReal uses --model for witness; --precision is δ
        cmd = [dreal_path, dr, f"--precision={delta}"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except FileNotFoundError:
            raise RuntimeError("dReal binary not found on PATH. Install dReal and ensure `dreal` is callable.")
        if "delta-sat" in out.stdout.lower():
            return True
        elif "unsat" in out.stdout.lower():
            return False
        else:
            # Treat unknown as failure; print stderr for debugging
            raise RuntimeError(f"dReal returned inconclusive.\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}")
