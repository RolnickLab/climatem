# 2026-11-26 — forcing-memory inputs ("a1"): abrupt-4xCO2 rollout

**Question.** Baseline, A and B2 (24 Nov) all step once in abrupt-4xCO2 and then stay flat, because CO2 reaches the
global transition only through six values from the last 60 months. Does giving the model inputs that summarise the
whole forcing history let it keep warming?

**Change (opt-in, `data_params.global_memory_years = [1, 10, 30, 100]`).** For CO2 and CH4, four extra inputs per
month: exponentially weighted running means with time constants 1, 10, 30 and 100 years
(m_t = m_{t-1} + (x_t - m_{t-1}) / (12 T), started at the first month of the record; SSP samples carry all of
historical, abrupt-4xCO2 the first 60 months of historical as in Shan's setup). They never exceed the largest CO2 seen,
so 4xCO2 extrapolates no further than CO2 itself. Appended after the six lag values; the gated attention gets 25
tokens instead of 17 (`CO2 mem30y` ...), the MLP baseline's co2_net/ch4_net take 10 values instead of 6.
Code: `climatem/data_loader/{cmip6_dataset,causal_datamodule}.py`, `climatem/model/{tsdcd_latent_hierarchy_raw,
attention_global}.py`, `scripts/{main_picabu,rollout_bf}.py`. Defaults reproduce the old behaviour.

**Run.** B2 config + memory (`configs/attention_B_mem.json`), 120k iterations (the 24 Nov runs: 180-200k), job
11915690, 5 h 08 on an RTX 8000 (~0.16 s/it, twice the 24 Nov rate). Final readout `readouts/`: CO2 mem30y is the
most-attended CO2 token (0.034), lag0 the least (0.021); gates hardly moved (CO2 0.94-0.95, CH4 mem30y 0.84 lowest).

**abrupt-4xCO2 rollout (NorESM2-LM r1i1p1f1, held out; same filter and forcing files as 24 Nov)** —
`rollouts_4xco2/`, decade means of the global-mean anomaly vs piControl [K]:

| | years 1-10 | 11-20 | 41-50 | annual RMSE |
|---|---|---|---|---|
| NorESM2-LM | 2.68 | 3.26 | 3.05 | |
| baseline (24 Nov) | 1.62 | 1.78 | 1.77 | 1.53 |
| A (24 Nov) | 1.47 | 1.45 | 1.43 | 1.83 |
| B2 (24 Nov) | 1.28 | 1.34 | 1.28 | 1.95 |
| **B2 + memory** | **2.40** | **2.47** | **2.48** | **0.84** |

Year by year, B2+memory: 2.15 (y1), 2.41 (y2), then 2.4-2.5 for the remaining 83 years; truth 1.43, 2.03, 2.37, 2.9
(y4), 3.1 (y10), 3.5-3.65 (y60-85). So the immediate response is now about the right size (the first 3 years match),
but the slow rise is still missing: from year 4 on the prediction is flat 0.6-1.2 K below the truth.

**What the memory inputs did in the rollout** (`shan_plot_B2mem.png`): the fed CO2 100-year average climbs from
-1.1 to 15.6 normalised units over the 85 years (correct: the memory starts at pre-industrial and relaxes toward the
4x level); the global latent jumps 3.1 -> 3.7 in the first five years and then drifts 3.7 -> 3.9 over 85 years. The
drift is real but far too small to show in the decoded temperature.

**Reading (mine).** The model used the memory inputs to size the step, not to continue the warming: in the training
data the memories rise together with CO2, so they carry no separate signal about delayed warming (the caveat stated
before the run). An in-range check (ssp370, job 11920700) and the baseline+memory run (job 11915691, rollout 11920701)
are pending; one run per variant, no seeds, so the size of the step gain (1.3 -> 2.4 K) is not yet separated from
run-to-run variation.

**ssp370 in-range rollout (job 11920700, same setup as 24 Nov)** — `rollouts_ssp370/`, decade means [K]:

| | y1-10 | y21-30 | y41-50 | y61-70 | y76-85 | trend [K/decade] | annual RMSE |
|---|---|---|---|---|---|---|---|
| NorESM2-LM | 0.81 | 1.18 | 1.68 | 2.41 | 2.92 | 0.29 | |
| B2 (24 Nov) | 0.78 | 1.26 | 1.78 | 2.36 | 2.87 | 0.28 | 0.14 |
| **B2 + memory** | 0.80 | 1.16 | 1.71 | 2.63 | **3.41** | 0.35 | 0.23 |

In range the memory run tracks the truth for the first 50 years and then warms too fast: +0.5 K by 2091-2100, trend
0.35 vs 0.29 K/decade (B2 0.28). Same direction as the 4xCO2 result: the run is more sensitive to CO2 at high
concentrations, without a delayed component. One run, no seeds.
