import pandas as pd, numpy as np

df = pd.read_csv("outputs/logs/records.csv")

# Choose columns that do exist
cols = ["spec","world_seed","passrate","best_rho","gpu_traj_s","accept_rate"]
df = df[cols].copy()

agg = (df
   .groupby("spec", as_index=False)
   .agg(pass_mean=("passrate","mean"),
        pass_std =("passrate",lambda x: x.std(ddof=1)),
        rho_mean =("best_rho","mean"),
        thr_mean =("gpu_traj_s","mean"),
        acc_mean =("accept_rate","mean"),
   )
   .sort_values("pass_mean", ascending=False)
)

print("\n=== Aggregate over seeds ===")
print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n=== Per-seed (sanity) ===")
print(df.sort_values(["spec","world_seed"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
