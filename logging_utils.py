import csv, os
from datetime import datetime
FIELDS = ["time","task","spec","device","N","T","dt","num_obstacles","passrate","success","best_rho",
          "gpu_traj_s","rho_min","K","delta","accept_rate","reject_rate","near_feasible"]
def append_record(path, rec: dict):
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists: w.writeheader()
        rec2 = {"time": datetime.now().isoformat()}
        rec2.update({k: rec.get(k,"") for k in FIELDS if k!="time"})
        w.writerow(rec2)
