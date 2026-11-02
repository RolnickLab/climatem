"""
How strong is CH4's temperature signal in FaIR, compared with the noise a model would have to see it through?

For each CH4 multiplier (a step from the pre-industrial baseline at year 0, CO2 and N2O held at baseline) and
each climate (the NorESM2-LM-like variants by default), runs FaIR with and without CH4's indirect forcing
(ozone, stratospheric water vapour) and reports the forcing and the warming after 50 and 100 years.

Noise references:
- the model's piControl global-mean temperature: year-to-year sd and sd of 10-year means (fair-calibrate data);
- FaIR's own stochastic internal variability for the same climates (control run, sigma_eta/sigma_xi at the
  calibrated medians because the fit does not estimate them).

Usage
-----
    python scripts/measure_fair_ch4_signal.py --climate-params <params.csv> [<params.csv> ...]

Output: a new folder data/fair/analysis/<YYYYmmdd-HHMMSS>_ch4_signal/ with ch4_signal.csv, summary.txt and a figure.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_fair_to_cmip_4xco2 import (  # noqa: E402
    C_ALT,
    C_DATA,
    C_FIT,
    git_state,
    load_cmip,
    stamp,
    style,
)
from generate_fair_scenarios import load_climate_params, run_fair  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NOISE_YEARS = 500


def step_scenarios(multipliers, n_years):
    n_tb = n_years + 1
    scens = []
    for m in multipliers:
        ch4 = np.full(n_tb, np.log(m))
        ch4[0] = 0.0  # equilibrium at the start, step after year 0
        scens.append(
            {
                "co2": np.zeros(n_tb),
                "ch4": ch4,
                "family": "step",
                "co2_shape": "flat",
                "ch4_shape": f"x{m}",
                "corr": np.nan,
            }
        )
    return scens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--climate-params", type=Path, nargs="+", required=True)
    ap.add_argument("--model", default="NorESM2-LM", help="whose piControl gives the noise reference")
    ap.add_argument("--multipliers", default="1.25,1.5,2,3,4,6,8")
    ap.add_argument("--years", type=int, default=100)
    args = ap.parse_args()

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_ch4_signal"
    run_dir = REPO / "data" / "fair" / "analysis" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    git = git_state()
    provenance = f"run {run_id} · scripts/{Path(__file__).name} @ {git}"
    lines = [f"run       {run_id}", f"command   python {' '.join(sys.argv)}", f"git       {git}", ""]

    def say(msg=""):
        print(msg)
        lines.append(msg)

    climate = load_climate_params(args.climate_params)
    mults = [float(x) for x in args.multipliers.split(",")]
    scens = step_scenarios(mults, args.years)
    rows = []
    for indirect in (False, True):
        f, _ = run_fair(scens, args.years, None, stochastic=False, seed=0, climate=climate, ch4_indirect=indirect)
        for j, cfg in enumerate(climate.index):
            for i, m in enumerate(mults):
                fo = f.forcing.isel(scenario=i, config=j)
                t = f.temperature.isel(scenario=i, config=j, layer=0).values
                rows.append(
                    {
                        "climate": cfg,
                        "ch4_multiplier": m,
                        "indirect": indirect,
                        "erf_ch4": float(fo.sel(specie="CH4").isel(timebounds=-1)),
                        "erf_total": float(fo.sum("specie", skipna=True).isel(timebounds=-1)),
                        "dT_50": float(t[50]),
                        "dT_100": float(t[min(100, args.years)]),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "ch4_signal.csv", index=False)

    # noise references
    t_pi, _ = load_cmip(args.model, "piControl", REPO / "data" / "fair" / "cmip6" / args.model)
    t_pi = t_pi - np.polyval(np.polyfit(np.arange(len(t_pi)), t_pi, 1), np.arange(len(t_pi)))
    sd_annual = float(np.std(t_pi))
    sd_decadal = float(np.std(pd.Series(t_pi).rolling(10).mean().dropna().to_numpy()[::10]))
    ctrl = step_scenarios([1.0], NOISE_YEARS)
    f_noise, _ = run_fair(ctrl, NOISE_YEARS, None, stochastic=True, seed=0, climate=climate, ch4_indirect=True)
    sd_fair = {
        cfg: float(f_noise.temperature.isel(scenario=0, config=j, layer=0)[50:].std())
        for j, cfg in enumerate(climate.index)
    }
    say(
        f"noise: {args.model} piControl global-mean T, detrended: year-to-year sd {sd_annual:.3f} K, "
        f"sd of non-overlapping 10-yr means {sd_decadal:.3f} K"
    )
    for cfg, v in sd_fair.items():
        say(
            f"noise: FaIR stochastic control, {cfg} (sigma_eta/sigma_xi = calibrated medians): "
            f"year-to-year sd {v:.3f} K"
        )

    say("\nCH4 step from baseline (CO2, N2O at baseline). Forcing at the end; warming after 50 and 100 years.")
    for cfg in climate.index:
        say(f"\n{cfg}")
        say(
            f"  {'CH4 x':>6} | {'ERF CH4 only':>12} {'dT100':>6} {'/noise':>6} | {'ERF +indirect':>13} "
            f"{'dT100':>6} {'/noise':>6}"
        )
        for m in mults:
            a = df[(df.climate == cfg) & (df.ch4_multiplier == m) & ~df.indirect].iloc[0]
            b = df[(df.climate == cfg) & (df.ch4_multiplier == m) & df.indirect].iloc[0]
            say(
                f"  {m:>6.2f} | {a.erf_total:>12.2f} {a.dT_100:>6.2f} {a.dT_100 / sd_annual:>6.1f} | "
                f"{b.erf_total:>13.2f} {b.dT_100:>6.2f} {b.dT_100 / sd_annual:>6.1f}"
            )
    say("\n'/noise' = warming after 100 years divided by the year-to-year sd of the piControl.")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for j, cfg in enumerate(climate.index):
        for indirect, color, ls in ((False, C_ALT, "--"), (True, C_FIT, "-")):
            d = df[(df.climate == cfg) & (df.indirect == indirect)]
            ax.plot(
                d.ch4_multiplier,
                d.dT_100,
                color=color,
                ls=ls,
                lw=2 if j == 0 else 1.2,
                marker="o",
                ms=4,
                alpha=1 if j == 0 else 0.6,
                label=f"{'with' if indirect else 'without'} indirect forcing, {cfg}",
            )
    for v, lab in ((sd_annual, "piControl year-to-year sd"), (2 * sd_annual, "2 x that")):
        ax.axhline(v, color=C_DATA, lw=1, ls=":")
        ax.text(mults[-1], v, f" {lab}", color=C_DATA, fontsize=8, va="bottom", ha="right")
    ax.set(
        xscale="log",
        xlabel="CH4 multiplier of pre-industrial (log scale)",
        ylabel="warming after 100 years (K)",
        title="CH4-only step: how much warming, against the noise",
    )
    ax.set_xticks(mults)
    ax.set_xticklabels([f"{m:g}" for m in mults])
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp(fig, provenance)
    fig.savefig(run_dir / "ch4_signal.png", dpi=150)
    plt.close(fig)
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {run_dir}/")


if __name__ == "__main__":
    main()
