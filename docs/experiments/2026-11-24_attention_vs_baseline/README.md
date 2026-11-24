# Snapshot 2026-11-24 — attention global transition (A, B2) vs Shan's MLP baseline

Frozen record of the three 200k-iteration runs reported on 2026-11-24. Everything needed to reproduce or
re-read them is in this folder or pinned by commit hash below.

## Code state

| run | code | notes |
|---|---|---|
| baseline (job 11864320) | `a692d24` on `ilija_thesis` (= Shan's `sz_forcing_dev` 8e51d97 + two FaIR commits) | Shan's `tsdcd_latent_hierarchy_raw.py` + `train_model_multi.py`, untouched |
| attention A (11875143), B2 (11875142) | working tree that became **`ce65ca6`** | committed right after the runs finished; pre-commit's docformatter rewrapped docstrings between run and commit, no functional change (black/isort/flake8 passed unchanged) |

Environment: Mila cluster, `env_emulator_climatem` venv, Python 3.10.11, torch 2.5.1+cu124, numpy 1.26.4,
`PYTHONPATH=$HOME/projects/climatem/thesis`. GPUs: baseline L40S (cn-l007, partition long), A L40S (cn-l001, main),
B2 L40S (cn-l005, unkillable). Wall time 4 h 52 / 4 h 56 / 4 h 59 min.

## Data

NorESM2-LM, variables ts + co2mass + ch4global (+ mmrbc, so2 in the file name but unused by the transition),
train hist + ssp126 + ssp245, test ssp370, tau 5, d_z 48, d_z_global 1, lr 1e-4, batch 128, 200k iterations.
Exact values: `params/*_params.json` (the `params.json` each run wrote) and `configs/*.json` (what was submitted).

## What differs between the runs

`diff configs/..._baseline_mila.json configs/..._attention_B_mila.json`:

```
model_params.global_transition   "attention"   (baseline: absent = "mlp")
model_params.attention_dim       16
model_params.attention_gate      true          (A: false)
model_params.attention_gate_init 3.0
model_params.attention_gate_coeff 1.0          (A: 0.0)
model_params.attention_gate_anchor true
optim_params.sparsity_mu_max     10000.0       (baseline: absent = uncapped)
```

## Results

Autoregressive validation RMSE (normalised anomalies; target spread 0.962):

| | step 1 | step 9 | mean |
|---|---|---|---|
| baseline | 0.910 | 0.973 | 0.943 |
| A | 0.912 | 0.992 | 0.955 |
| B2 | 0.912 | 0.970 | 0.944 |

Attention shares at 180k (`readouts/*.txt`): A 0.48 z2 / 0.28 CO2 / 0.25 CH4 (flat 0.04 over every forcing token);
B2 0.81 / 0.17 / 0.02 with gates z2 t-1..t-4 ≈ 1, z2 t-5 0.07, CO2 lag60 0.88, lag1 0.88, lag2-4 0.58, lag5-11 0.36,
lag0 0.28, lag12-59 0.24, CH4 0.06–0.35. Plots: `plots/attention_{A,B2}_180000.png`.

Fix verification: sparsity mu plateaued at 1e4 in A and B2 (no NaN; old B 11867039 went NaN at 118k with mu 1e30);
the gate anchor stopped the uniform gate collapse seen in old B.

Caveat (inference, not measured): all three barely beat climatology on monthly anomalies, so B2's "CO2 slow, CH4 off"
ordering may reflect the persistence-dominated target rather than recovered sensitivity. The 4xCO2 rollout is the
test that would separate the two.

## Files

- `configs/` — the three submitted configs (`$SCRATCH` left unexpanded, personal exp_path).
- `params/` — `params.json` as written by each run, with the user's $SCRATCH/$HOME prefixes put back as variables.
- `readouts/` — 180k attention matrices (`.npy`, n_out × 17 tokens), text tables, B2 gates.
- `plots/` — 180k attention plots.
- `sbatch/` — the two job scripts used.
- `metrics.json` — the numbers above, machine-readable.
- Full run folders (model.pth, all checkpoints, every plot): `$SCRATCH/results/{shan_baseline,attention_A,attention_B}/`
  — $SCRATCH is purged after 90 days unused. Final `model.pth` of each run (2.9 MB each) is copied to `models/`.
- Job logs: `../../../logs/attention.11875142.*`, `attention.11875143.*`, `shan_baseline.11864320.*`.
