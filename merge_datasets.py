# merge_datasets.py
import torch, sys
outs = []
meta = None
for p in sys.argv[1:-1]:
    d = torch.load(p, map_location="cpu")
    if meta is None:
        meta = {k:d[k] for k in ["K","T","dt","max_obs"] if k in d}
    outs.extend(d["data"])
torch.save({"data": outs, **(meta or {})}, sys.argv[-1])
print(f"Merged {len(outs)} samples -> {sys.argv[-1]}")
