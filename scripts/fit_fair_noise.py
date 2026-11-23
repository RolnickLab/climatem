"""
Fit FaIR's internal variability (sigma_eta, sigma_xi) to a CMIP6 model's monthly global-mean temperature.

FaIR's noise only means something at the time step it is set for: the same sigma_eta/sigma_xi give about a
third of the year-to-year variability at a monthly step as at an annual one (FAIR_STATE.md, question 5). So the
two numbers are fitted at the step the data will be generated with: monthly.

Target: the model's internal variability, taken from the differences between ensemble members of the same SSP
(x_i - x_j) / sqrt(2). The forced response and the seasonal cycle are the same in every member and cancel; what
is left has the variance of one member's internal variability. Only historical/SSP runs are read — never
abrupt-4xCO2, the thesis's held-out test.

Two statistics are matched, one per knob: the variance of monthly values (fast) and of annual means (slow).
FaIR is linear in its noise, so each variance is a*sigma_eta^2 + b*sigma_xi^2; two unit runs give a and b,
and a non-negative least-squares solve gives the sigmas. A third run checks the result, a fourth shows the
calibrated-median noise for comparison. FaIR statistics use windows as long as the model's series, each
centred on its own mean, so both sides use the same estimator.

Usage
-----
    python scripts/fit_fair_noise.py --climate-params <params.csv> [<params.csv> ...]

Output: a new folder data/fair/cmip6/<model>/noise_fits/<YYYYmmdd-HHMMSS>_monthly/ with noise_params.csv
(one row per climate), summary.txt and noise_fit.png.
"""

import argparse
import itertools
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_fair_to_cmip_4xco2 import (  # noqa: E402
    C_ALT,
    C_DATA,
    C_FIT,
    detrended_sd,
    git_state,
    load_cmip,
    stamp,
    style,
)
from generate_fair_scenarios import PARAMS_FILE, load_climate_params, run_fair  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CLIMATESET = REPO / "Climateset_DATA" / "outputs" / "CMIP6"
STEP = 1 / 12
PER = 12  # steps per year
SPINUP_YEARS = 200  # the deep-ocean layer needs centuries to reach its stationary noise level


# ---------------------------------------------------------------------------------------------
# model data
# ---------------------------------------------------------------------------------------------
def monthly_global_means(mon_dir):
    """{year: 12 area-weighted global-mean tas values} from one-file-per-year ClimateSet output."""
    out = {}
    for year_dir in sorted(mon_dir.iterdir(), key=lambda p: int(p.name)):
        (fp,) = year_dir.glob("*.nc")
        with xr.open_dataset(fp, decode_times=False) as ds:
            monthly = ds["tas"].weighted(np.cos(np.deg2rad(ds["y"]))).mean(("y", "x")).values
        if len(monthly) != PER:
            raise ValueError(f"{fp} has {len(monthly)} months")
        out[int(year_dir.name)] = monthly
    return out


def internal_variability(model, scenarios):
    """[(label, monthly series)]: (member_i - member_j) / sqrt(2) over the years all members share."""
    out = []
    for scen in scenarios:
        members = {}
        for member_dir in sorted((CLIMATESET / model).iterdir()):
            d = member_dir / scen / "tas" / "250_km" / "mon"
            if d.is_dir():
                members[member_dir.name] = monthly_global_means(d)
        if len(members) < 2:
            continue
        years = sorted(set.intersection(*(set(m) for m in members.values())))
        series = {k: np.concatenate([v[y] for y in years]) for k, v in members.items()}
        for a, b in itertools.combinations(sorted(series), 2):
            out.append((f"{scen} {a} - {b}, {years[0]}-{years[-1]}", (series[a] - series[b]) / np.sqrt(2)))
    return out


# ---------------------------------------------------------------------------------------------
# statistics (same estimator for model and FaIR)
# ---------------------------------------------------------------------------------------------
def pooled_variances(series):
    """Variance of monthly values and of annual means, pooled over series that are each centred on their mean."""
    ss_m = n_m = ss_a = n_a = 0.0
    for x in series:
        x = x[: len(x) // PER * PER]
        ss_m += np.sum((x - x.mean()) ** 2)
        n_m += len(x)
        a = x.reshape(-1, PER).mean(axis=1)
        ss_a += np.sum((a - a.mean()) ** 2)
        n_a += len(a)
    return np.array([ss_m / n_m, ss_a / n_a])


def acf(series, max_lag):
    """Autocorrelation at lags 0..max_lag months, pooled over centred series."""
    num, den = np.zeros(max_lag + 1), 0.0
    for x in series:
        x = x - x.mean()
        den += np.sum(x * x)
        for k in range(max_lag + 1):
            num[k] += np.sum(x[: len(x) - k] * x[k:])
    return num / den


def windows(T, lengths):
    """Cut FaIR runs (steps x seeds) into windows with the model's series lengths, in the same proportions."""
    out, weights = [], []
    for L in sorted(set(lengths)):
        chunks = [col[i : i + L] for col in T.T for i in range(0, len(col) - L + 1, L)]
        out.append(chunks)
        weights.append(lengths.count(L) * L / sum(lengths))
    return out, weights


def fair_variances(T, lengths):
    groups, weights = windows(T, lengths)
    return sum(w * pooled_variances(g) for g, w in zip(groups, weights))


# ---------------------------------------------------------------------------------------------
# FaIR
# ---------------------------------------------------------------------------------------------
def control(climate_row, n_years, n_seeds, sigma_eta, sigma_xi, seed):
    """Monthly FaIR temperature (steps x seeds) at constant pre-industrial concentrations, after spin-up."""
    climate = pd.DataFrame([climate_row] * n_seeds, index=[f"c{i}" for i in range(n_seeds)])
    noise = pd.DataFrame({"sigma_eta": sigma_eta, "sigma_xi": sigma_xi}, index=climate.index)
    n_tb = int(round(n_years / STEP)) + 1
    flat = [{"co2": np.zeros(n_tb), "ch4": np.zeros(n_tb)}]
    f, names = run_fair(flat, n_years, None, True, seed, climate=climate, step=STEP, noise=noise)
    T = f.temperature.sel(scenario=names[0], layer=0).values
    mid = 0.5 * (T[1:] + T[:-1])  # one value per month: mean of its two timebounds
    return mid[SPINUP_YEARS * PER :]


# ---------------------------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------------------------
def plot(model, data, results, provenance, path):
    max_lag = 36
    lags = np.arange(max_lag + 1)
    fig, axes = plt.subplots(len(results), 2, figsize=(13, 3.6 * len(results)), squeeze=False)
    sample = data[0]
    for row, (name, r) in zip(axes, results.items()):
        ax = row[0]
        ax.plot(lags, acf([x for _, x in data], max_lag), color=C_DATA, lw=2, label=f"{model} (member differences)")
        ax.plot(lags, acf(list(r["T_fit"].T), max_lag), color=C_FIT, lw=1.8, label="FaIR, fitted noise")
        ax.plot(
            lags,
            acf(list(r["T_default"].T), max_lag),
            color=C_ALT,
            lw=1.5,
            ls="--",
            label="FaIR, calibrated-median noise",
        )
        ax.axhline(0, color="#b9b9b2", lw=0.8)
        ax.set(xlabel="lag (months)", ylabel="autocorrelation", title=f"{name}: how long anomalies persist")
        ax.legend(frameon=False, fontsize=8)
        style(ax)

        ax = row[1]
        n = 20 * PER
        t = np.arange(n) / PER
        ax.plot(t, sample[1][:n] + 0.6, color=C_DATA, lw=0.9)
        ax.plot(t, r["T_fit"][:n, 0], color=C_FIT, lw=0.9)
        ax.text(0.2, 0.6 + 0.35, f"{model}: {sample[0]}", color=C_DATA, fontsize=8)
        ax.text(0.2, 0.35, "FaIR, fitted noise (one seed)", color=C_FIT, fontsize=8)
        sd = np.sqrt(r["target"])
        ax.set(
            xlabel="years",
            ylabel="K (model shifted +0.6)",
            title=f"20-year sample · target sd monthly {sd[0]:.3f} K, annual {sd[1]:.3f} K",
        )
        style(ax)
    stamp(fig, provenance)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--climate-params", type=Path, nargs="+", required=True)
    ap.add_argument("--model", default="NorESM2-LM")
    ap.add_argument("--scenarios", default="ssp245,ssp370", help="SSPs whose members are differenced")
    ap.add_argument("--years", type=int, default=1200, help="FaIR control length per seed, incl. spin-up")
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_monthly"
    run_dir = REPO / "data" / "fair" / "cmip6" / args.model / "noise_fits" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    git = git_state()
    provenance = f"run {run_id} · {args.model} · scripts/{Path(__file__).name} @ {git}"
    lines = [f"run       {run_id}", f"command   python {' '.join(sys.argv)}", f"git       {git}", ""]

    def say(msg=""):
        print(msg)
        lines.append(msg)

    data = internal_variability(args.model, args.scenarios.split(","))
    lengths = [len(x) for _, x in data]
    target = pooled_variances([x for _, x in data])
    say(f"target: {args.model} internal variability from {len(data)} member differences (historical/SSP only):")
    for label, x in data:
        say(f"  {label}  ({len(x) // PER} y)")
    say(f"  sd of monthly values {np.sqrt(target[0]):.3f} K, sd of annual means {np.sqrt(target[1]):.3f} K")
    t_pi, _ = load_cmip(args.model, "piControl", REPO / "data" / "fair" / "cmip6" / args.model)
    say(f"  cross-check: {args.model} piControl annual sd (detrended, {len(t_pi)} y) {detrended_sd(t_pi):.3f} K")
    say("")

    med = pd.read_csv(PARAMS_FILE, index_col=0)[["sigma_eta", "sigma_xi"]].median()
    climate = load_climate_params(args.climate_params)
    rows, results = [], {}
    for name, row in climate.iterrows():
        unit_eta = fair_variances(control(row, args.years, args.seeds, 1.0, 0.0, 1000), lengths)
        unit_xi = fair_variances(control(row, args.years, args.seeds, 0.0, 1.0, 2000), lengths)
        A = np.column_stack([unit_eta, unit_xi]) / target[:, None]  # relative misfit per statistic
        squares, _ = nnls(A, np.ones(2))
        sigma_eta, sigma_xi = np.sqrt(squares)
        T_fit = control(row, args.years, args.seeds, sigma_eta, sigma_xi, 3000)
        T_default = control(row, args.years, args.seeds, med["sigma_eta"], med["sigma_xi"], 4000)
        got, default = fair_variances(T_fit, lengths), fair_variances(T_default, lengths)
        results[name] = dict(target=target, T_fit=T_fit, T_default=T_default)
        lag1 = [acf(s, 1)[1] for s in ([x for _, x in data], list(T_fit.T), list(T_default.T))]
        say(f"climate {name}")
        say(f"  fitted     sigma_eta {sigma_eta:.3f} W m-2, sigma_xi {sigma_xi:.3f} W m-2 yr^0.5 (monthly step)")
        say(f"  calibrated sigma_eta {med['sigma_eta']:.3f}, sigma_xi {med['sigma_xi']:.3f} (median, for comparison)")
        say("                        target   fitted   calibrated-median noise")
        for i, what in enumerate(("sd monthly values (K)", "sd annual means (K)")):
            say(f"  {what:22s} {np.sqrt(target[i]):.3f}    {np.sqrt(got[i]):.3f}    {np.sqrt(default[i]):.3f}")
        say(f"  {'lag-1 autocorr (unfit)':22s} {lag1[0]:.3f}    {lag1[1]:.3f}    {lag1[2]:.3f}")
        say("")
        rows.append(
            {
                "climate": name,
                "climate_params": str(next(p for p in args.climate_params if Path(p).resolve().parent.name == name)),
                "timestep_years": STEP,
                "sigma_eta": sigma_eta,
                "sigma_xi": sigma_xi,
                "target_sd_monthly": np.sqrt(target[0]),
                "target_sd_annual": np.sqrt(target[1]),
                "fitted_sd_monthly": np.sqrt(got[0]),
                "fitted_sd_annual": np.sqrt(got[1]),
            }
        )
    pd.DataFrame(rows).set_index("climate").to_csv(run_dir / "noise_params.csv")
    plot(args.model, data, results, provenance, run_dir / "noise_fit.png")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {run_dir}/ (noise_params.csv, summary.txt, noise_fit.png)")


if __name__ == "__main__":
    main()
