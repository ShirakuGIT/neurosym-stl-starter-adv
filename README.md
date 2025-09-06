# Neuro-Symbolic STL (Advanced) — GPU pre-check + adaptive-N + δ-SMT stub + logging

This extends the basic starter with:
- **Bounded STL operators**: F_[a,b], G_[a,b], and Until
- **Chunked evaluation** for long horizons (lower peak memory)
- **Adaptive-N controller** tied to spec complexity
- **Top-K δ-SMT stub** with ρ-gating (replace with dReal later)
- **Tiny DDPM prior skeleton** (optional; falls back if no checkpoint)
- **Run logger & leaderboard CSV**

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_demo_advanced.py --device=cuda --task=nav2d --spec=gf_avoid_reach --N_base=512 --T=64
```

Outputs:
- `outputs/plots/*.png`
- `outputs/logs/records.csv` (aggregated run logs)
- `outputs/checkpoints/` (if you train the tiny DDPM prior)

## Notes
- δ-SMT here is a **placeholder gate** (`smt_stub.py`) so you can run end-to-end without heavy solvers. Swap with dReal/Z3 later.
- The DDPM is a minimal UNet and trainer in `prior_ddpm/`. If no checkpoint is found, we **fallback** to the smooth sampler.
