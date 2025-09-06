# control_basis.py
import numpy as np
import torch

# ----------------------- cached basis (CPU) -----------------------
# We cache the CPU basis (float32) by (T, K, degree); and lazily
# move to device when requested.
_BASIS_CPU = {}
_BASIS_DEV = {}

def _uniform_clamped_knots(n_ctrl: int, degree: int):
    """[0,...,0, u1,...,uM, 1,...,1] with clamping at ends."""
    n_internal = max(0, n_ctrl - degree - 1)
    if n_internal > 0:
        internal = np.linspace(0.0, 1.0, n_internal + 2, dtype=np.float32)[1:-1]
    else:
        internal = np.array([], dtype=np.float32)
    knots = np.concatenate([
        np.zeros(degree + 1, dtype=np.float32),
        internal,
        np.ones(degree + 1, dtype=np.float32)
    ])
    return knots  # shape: (n_ctrl + degree + 1,)

def _cox_de_boor_basis(T: int, K: int, degree: int) -> np.ndarray:
    """Return B \in R^{T x K}, uniform clamped B-spline basis."""
    # param points in [0,1]
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    U = _uniform_clamped_knots(K, degree)  # len = K+degree+1

    # N_{i,0}(t)
    N = np.zeros((K, degree + 1, T), dtype=np.float32)
    for i in range(K):
        # N_{i,0}(t) = 1 if U[i] <= t < U[i+1], with special case at t=1
        left, right = U[i], U[i + 1]
        if right > left:
            mask = (t >= left) & (t < right)
            # ensure last point t=1 belongs to the last span
            if i == K - 1:
                mask |= (t == 1.0)
            N[i, 0, mask] = 1.0

    # elevate degree
    for p in range(1, degree + 1):
        for i in range(K):
            denom1 = U[i + p] - U[i]
            denom2 = U[i + p + 1] - U[i + 1]

            a = 0.0
            b = 0.0
            if denom1 > 0:
                a = ((t - U[i]) / denom1)[None, :]
            if denom2 > 0 and (i + 1) < K:
                b = ((U[i + p + 1] - t) / denom2)[None, :]

            left_term = 0.0 if denom1 == 0 else a * N[i, p - 1, :][None, :]
            right_term = 0.0
            if (i + 1) < K and denom2 != 0:
                right_term = b * N[i + 1, p - 1, :][None, :]

            N[i, p, :] = left_term + right_term

    # gather degree-p basis for all i => (T,K)
    B = np.transpose(N[:, degree, :], (1, 0))  # (T,K)
    return B.astype(np.float32)

def _get_basis(T: int, K: int, degree: int, device: torch.device) -> torch.Tensor:
    key = (T, K, degree)
    if key not in _BASIS_CPU:
        B_cpu = _cox_de_boor_basis(T, K, degree)  # np.float32 (T,K)
        _BASIS_CPU[key] = torch.from_numpy(B_cpu)  # CPU tensor, float32
    B = _BASIS_CPU[key]
    if device.type == "cpu":
        return B
    dev_key = (key, device.type)
    if dev_key not in _BASIS_DEV:
        _BASIS_DEV[dev_key] = B.to(device)
    return _BASIS_DEV[dev_key]

# ----------------------- public API -----------------------

def synthesize_controls_from_knots(knots: torch.Tensor, T: int, degree: int = 3) -> torch.Tensor:
    """
    knots: (K, 2) control 'knots' (interpreted as control points for a B-spline in u-space).
    Returns u_seq: (T, 2) = B(T,K) @ knots(K,2)
    """
    assert knots.dim() == 2 and knots.shape[1] == 2, "knots must be (K,2)"
    K = knots.shape[0]
    device = knots.device

    # basis is tiny; precompute on CPU fp32 and reuse; move to device once.
    with torch.no_grad():
        B = _get_basis(T=T, K=K, degree=degree, device=device)  # (T,K), float32

    # ensure dtype matches knots for matmul (promote B if needed)
    if B.dtype != knots.dtype:
        B = B.to(knots.dtype)

    # (T,K) @ (K,2) -> (T,2)
    return B @ knots
