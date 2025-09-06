# control_basis.py
import torch

def bspline_basis_matrix(T, K, degree=3, device='cpu'):
    """
    Build a (T x K) basis that maps K control knots -> T controls via uniform cubic B-spline.
    Implementation: recursive Cox–de Boor with a uniform knot vector and clamping at ends.
    """
    assert degree == 3, "Only degree=3 implemented for brevity"
    # uniform open knot vector
    n = K - 1
    m = n + degree + 1
    # parameter t in [0,1]
    t = torch.linspace(0, 1, T, device=device)
    # knot vector U: 0 repeated (degree+1) times, then uniform, then 1 repeated
    U = torch.zeros(m+1, device=device)
    U[degree:n+1] = torch.linspace(0, 1, n - degree + 1, device=device)
    U[n+1:] = 1.0

    # basis N_{i,0}
    N = []
    for i in range(0, n+1):
        left = (U[i] <= t).float()
        right = (t < U[i+1]).float()
        if i == n:  # include right end
            right = (t <= U[i+1]).float()
        N.append(left * right)
    N = torch.stack(N, dim=1)  # (T, n+1)

    # elevate to degree p
    for p in range(1, degree+1):
        N_next = torch.zeros_like(N)
        for i in range(0, n - p + 1):
            denom1 = U[i+p] - U[i]
            denom2 = U[i+p+1] - U[i+1]
            term1 = 0.0 if denom1 == 0 else ((t - U[i]) / denom1) * N[:, i]
            term2 = 0.0 if denom2 == 0 else ((U[i+p+1] - t) / denom2) * N[:, i+1]
            N_next[:, i] = term1 + term2
        N = N_next[:, :n - p + 1]
    # N is (T, K) with K = n - degree + 1 = K
    return N  # (T,K)

def synthesize_controls_from_knots(knots_u, T):
    """
    knots_u: (K,2) learnable control knot vectors
    returns u_seq: (T,2)
    """
    device = knots_u.device
    K = knots_u.shape[0]
    B = bspline_basis_matrix(T, K, degree=3, device=device)  # (T,K)
    u_seq = B @ knots_u  # (T,2)
    return u_seq
