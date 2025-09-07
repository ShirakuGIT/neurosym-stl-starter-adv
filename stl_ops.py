import torch

def t_min(x, dim=-1): return x.min(dim=dim).values
def t_max(x, dim=-1): return x.max(dim=dim).values

def window(x, t0, t1):
    T = x.shape[-1]; a = max(0, int(t0)); b = min(T-1, int(t1))
    if a>b: a,b=b,a
    return x[..., a:b+1]

def G_always(rho_t):      return t_min(rho_t)
def F_eventually(rho_t):  return t_max(rho_t)
def G_bounded(rho_t, a, b):   return t_min(window(rho_t, a, b))
def F_bounded(rho_t, a, b):   return t_max(window(rho_t, a, b))
def U_until(rho_phi_t, rho_psi_t):
    prefix_min = torch.cummin(rho_phi_t, dim=-1).values
    return t_max(torch.minimum(rho_psi_t, prefix_min))

def conj(a,b): return torch.minimum(a,b)
def disj(a,b): return torch.maximum(a,b)
def neg(a):    return -a
