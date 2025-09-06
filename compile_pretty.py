import pandas as pd
df = pd.read_csv("outputs/logs/records.csv")
cols = ["task","spec","baseline","world_seed","passrate","best_rho","gpu_traj_s","accept_rate","rho_min","K","K_basis","iters_basis"]
for c in cols:
    if c not in df.columns: df[c]=None
df[cols].to_csv("outputs/logs/summary_clean.csv", index=False)
print("Wrote outputs/logs/summary_clean.csv")