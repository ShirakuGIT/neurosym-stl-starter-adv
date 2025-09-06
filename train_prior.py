# train_prior.py
import torch, argparse, os
from torch.utils.data import Dataset, DataLoader
from models.prior_unet import KnotPrior
from prior_model import TinyUNet

class ExpertKnotsDS(Dataset):
    def __init__(self, path):
        blob = torch.load(path, map_location="cpu")
        self.K = blob["K"]; self.data = blob["data"]
        self.cond_dim = self.data[0]["cond"].numel()
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        item = self.data[i]
        cond  = item["cond"].float()
        knots = item["knots"].float().reshape(-1)  # (2K,)
        return cond, knots

def linear_beta_schedule(T=1000, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="datasets/expert_knots_k8.pth")
    ap.add_argument("--out", type=str, default="checkpoints/prior_k8.pt")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=1000)  # if you prefer steps over epochs
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--Tdiff", type=int, default=1000)
    args = ap.parse_args()

    ds = ExpertKnotsDS(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    cond_dim = ds.cond_dim
    K = ds.K
    net = KnotPrior(K=K, cond_dim=cond_dim).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    betas = linear_beta_schedule(args.Tdiff).to(args.device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    global_step = 0
    for epoch in range(args.epochs):
        for cond, x0 in dl:
            net.train()
            cond = cond.to(args.device)
            x0   = x0.to(args.device)  # (B, 2K)
            B = x0.shape[0]

            # sample t
            t = torch.randint(0, args.Tdiff, (B,), device=args.device)
            a_bar_t = alpha_bar[t].view(B, 1)
            eps = torch.randn_like(x0)
            x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * eps

            eps_hat = net(x_t, (t.float() / (args.Tdiff-1)), cond)
            loss = torch.mean((eps - eps_hat) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
        print(f"[train] epoch={epoch+1} loss={float(loss):.6f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(model=net.state_dict(), K=K, cond_dim=cond_dim, Tdiff=args.Tdiff), args.out)
    print(f"[train] saved to {args.out}")

if __name__ == "__main__":
    main()
