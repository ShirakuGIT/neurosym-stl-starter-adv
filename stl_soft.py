import torch

def softmax_tau(x, tau=0.05, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    z = torch.exp((x - m)/tau).sum(dim=dim, keepdim=True)
    return (m + tau*torch.log(z)).squeeze(dim)

def softmin_tau(x, tau=0.05, dim=-1):
    return -softmax_tau(-x, tau=tau, dim=dim)

def F_eventually_soft(rho_t, tau=0.05):  return softmax_tau(rho_t, tau=tau, dim=-1)
def G_always_soft(rho_t, tau=0.05):      return softmin_tau(rho_t, tau=tau, dim=-1)
