import torch

# ############################################################################ #
# Hard STL Operators (Boolean-like Robustness using min/max)
# ############################################################################ #

def _min_max_op(op, x, dim, keepdim=False):
    """Helper function for min/max operations."""
    if op == 'min':
        y, _ = torch.min(x, dim=dim, keepdim=keepdim)
    elif op == 'max':
        y, _ = torch.max(x, dim=dim, keepdim=keepdim)
    else:
        raise ValueError(f"Unknown min/max operation: {op}")
    return y

def always(x, dim=1, keepdim=False):
    """Hard Always: min over time"""
    return _min_max_op('min', x, dim=dim, keepdim=keepdim)

def eventually(x, dim=1, keepdim=False):
    """Hard Eventually: max over time"""
    return _min_max_op('max', x, dim=dim, keepdim=keepdim)

def until(x1, x2, dim=1):
    """Hard Until: eventually(x2) and x1 must hold until then."""
    if dim != 1:
        raise NotImplementedError("Until is only implemented for dim=1")

    T = x1.shape[dim]
    # robustness of eventually(x2) at each time t
    e_x2 = torch.stack([_min_max_op('max', x2[:, t:], dim=1) for t in range(T)], dim=1)
    
    # robustness of always(x1) up to each time t
    a_x1 = torch.stack([_min_max_op('min', x1[:, :t], dim=1) for t in range(1, T + 1)], dim=1)
    
    # robustness is max over time t of (e_x2[t] and a_x1[t])
    robs = _min_max_op('max', torch.min(e_x2, a_x1), dim=1)
    return robs

# ############################################################################ #
# Soft STL Operators (Differentiable Approximations using softmin/max)
# ############################################################################ #

def softmin(x, dim, T=1.0):
    """Differentiable soft-minimum operator."""
    return -T * torch.logsumexp(-x/T, dim=dim)

def softmax(x, dim, T=1.0):
    """Differentiable soft-maximum operator."""
    return T * torch.logsumexp(x/T, dim=dim)

def soft_always(x, dim=1, T=1.0):
    """Soft Always"""
    return softmin(x, dim=dim, T=T)

def soft_eventually(x, dim=1, T=1.0):
    """Soft Eventually"""
    return softmax(x, dim=dim, T=T)

