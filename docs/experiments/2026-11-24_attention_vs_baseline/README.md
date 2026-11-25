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

## Rollout tests (added 2026-11-24 evening)

Particle-filter rollouts (`scripts/rollout_bf.py`, 1027 months, 20×50 particles, `pz2mu` selection, final checkpoint)
evaluated in kelvin with `scripts/evaluate_4xco2_rollout.py` (global-mean anomaly vs the piControl climatology; HEALPix
pixels are equal-area). The truth enters only as the initial condition at step 0 and through the per-step forcings
(CO2, CH4, mmrbc, so2 from the test loader); the target maps are never used after step 0.

**abrupt-4xCO2 (NorESM2-LM r1i1p1f1, held out, never trained on)** — `rollouts_4xco2/`:

| decade mean [K] | years 1-10 | 11-20 | 41-50 | 76-85 |
|---|---|---|---|---|
| NorESM2-LM | 2.68 | 3.26 | 3.05 | 3.59 |
| baseline | 1.62 | 1.78 | 1.77 | 1.80 |
| A | 1.47 | 1.45 | 1.43 | 1.47 |
| B2 | 1.28 | 1.34 | 1.28 | 1.35 |

All three stay at the initial level (year-1 truth 1.43 K). The global latent moves only while the 60-month CO2 window
fills (baseline z2 0.58 → 1.49 by month 60, then 1.54 flat; A 1.90 → 3.18 then flat; B2 3.89 flat from month 1), i.e.
CO2 acts as an instantaneous offset with no accumulating response. 4×CO2 is 27.8 normalised units, ~3× the training max.

**ssp370 (NorESM2-LM, the test scenario, not trained on)** — `rollouts_ssp370/`:

| decade mean [K] | years 1-10 | 11-20 | 41-50 | 76-85 | trend [K/decade] | annual RMSE [K] |
|---|---|---|---|---|---|---|
| NorESM2-LM | 0.82 | 1.03 | 1.68 | 2.92 | 0.29 | |
| baseline | 0.87 | 1.03 | 1.79 | 2.65 | 0.25 | 0.19 |
| A | 0.84 | 0.99 | 1.83 | 2.97 | 0.29 | 0.14 |
| B2 | 0.78 | 0.97 | 1.78 | 2.87 | 0.28 | 0.14 |

Inside the training range all three track the warming (annual corr 0.98). They also follow part of the year-to-year
variability (detrended annual corr 0.63 for the baseline); the global means of the four forcings do not (detrended
corr ≤ 0.2), so if that signal is real it must come through the spatial aerosol fields, which are NorESM's own
simulated mmrbc/so2 — unverified inference.

Jobs: 4xCO2 11884312-4; ssp370 11886342 (unkillable), 11886371-2 (main). Plots: `gmst_*_rollouts.png` (kelvin),
`shan_plot_*.png` (Shan's normalised GMST / z2 / forcing panel). The `gmst_*.npz` arrays are local only.
