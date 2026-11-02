"""
Fit FaIR's three-layer energy balance model to one CMIP6 model's abrupt-4xCO2 run.

!! For NorESM2-LM the abrupt-4xCO2 run is the thesis's HELD-OUT TEST (decided 2026-10-26, FAIR_STATE.md).
!! Fits of NorESM2-LM made here are test-only references: never generate training data from them.
!! FaIR for training data is calibrated on NorESM2-LM historical + SSP runs instead.

The script stays useful for other CMIP6 models (whose 4xCO2 runs are not a test) and as a reference for
what the held-out NorESM test contains.

Data
----
Global-mean annual anomalies of surface temperature and top-of-atmosphere (TOA) fluxes for abrupt-4xCO2
and piControl, as published in the fair-calibrate repository (data/cmip6-hbf/cmip_data/<model>/).
Downloaded once into data/fair/cmip6/<model>/.

Method
------
Weighted least squares of the deterministic three-layer model, stepped exactly as FaIR steps it
(fair.energy_balance_model), against surface temperature T and the TOA imbalance N = rsdt - rsut - rlut.
Each residual is divided by the year-to-year standard deviation of the same quantity in the model's own
piControl, so T and N count according to how noisy they are. Several starting points; the best fit wins.
This is simpler than fair-calibrate's own method (Kalman-filter maximum likelihood,
input/calibration/02_calibrate_cummins_3layer.r), so the numbers will not match theirs exactly. It does not
estimate the internal-variability parameters (sigma_eta, sigma_xi).

FaIR does not use forcing_4co2 as the CO2 forcing; it computes CO2 forcing from concentration. The script
therefore also works out the CO2 ``forcing_scale`` that makes FaIR's 4xCO2 forcing equal the fitted one, and
checks the result by running the full FaIR model under abrupt-4xCO2.

With ``--check-historical`` it also runs 1750-2014 with all forcings (RCMIP emissions, SSP2-4.5 branch;
non-climate parameters from the 841 calibrated members) twice: once with the calibrated climate parameters
and once with the fitted ones. Both are compared with the CMIP model's own historical global-mean warming.
Reading the historical netCDF files needs the h5netcdf or netCDF4 package.

Usage
-----
    python scripts/fit_fair_to_cmip_4xco2.py --model NorESM2-LM --check-historical
    python scripts/fit_fair_to_cmip_4xco2.py --model NorESM2-LM --fit-years 150   # standard CMIP length

Outputs
-------
Every run writes a new folder data/fair/cmip6/<model>/fits/<YYYYmmdd-HHMMSS>_fit<N>y/ (never overwritten):
    params.csv          one row, columns named as in the calibrated CSV
    summary.txt         command, git state, and every number printed during the run
    fit_4xco2.png       data vs fit; the run id is printed in the corner of each figure
    historical.png      with --check-historical
"""

import argparse
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from fair import FAIR  # noqa: E402
from fair.energy_balance_model import EnergyBalanceModel  # noqa: E402
from fair.interface import fill, initialise  # noqa: E402
from fair.io import read_properties  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CAL = REPO / "data" / "fair" / "calibrated_constrained_ensemble"
PARAMS_FILE = CAL / "calibrated_constrained_parameters_calibration1.4.1.csv"
SPECIES_FILE = CAL / "species_configs_properties_calibration1.4.1.csv"
GHG = ["CO2", "CH4", "N2O"]
CMIP_URL = (
    "https://raw.githubusercontent.com/chrisroadmap/fair-calibrate/main/data/cmip6-hbf/cmip_data/"
    "{m}/{e}/{m}_{e}_r1i1p1f1_anomalies.txt"
)
HIST_DIR = REPO / "Climateset_DATA" / "outputs" / "CMIP6" / "{m}" / "r1i1p1f1" / "historical" / "tas" / "250_km" / "mon"

# name, lower bound, upper bound, first guess. Units: heat capacity W m-2 yr K-1, transfer W m-2 K-1,
# forcing W m-2, gamma yr-1. All positive, so the fit works on their logarithms.
PARAMS = [
    ("ocean_heat_capacity[0]", 1.0, 30.0, 5.0),
    ("ocean_heat_capacity[1]", 2.0, 200.0, 20.0),
    ("ocean_heat_capacity[2]", 10.0, 2000.0, 100.0),
    ("ocean_heat_transfer[0]", 0.2, 4.0, 1.2),
    ("ocean_heat_transfer[1]", 0.2, 8.0, 2.0),
    ("ocean_heat_transfer[2]", 0.05, 4.0, 0.8),
    ("deep_ocean_efficacy", 0.3, 4.0, 1.3),
    ("forcing_4co2", 3.0, 14.0, 8.0),
    ("gamma_autocorrelation", 0.5, 50.0, 2.0),
]
NAMES = [p[0] for p in PARAMS]

# validated categorical pair (dataviz reference palette, slots 1-2) + neutral ink for the CMIP data
C_FIT, C_ALT, C_DATA = "#2a78d6", "#eb6834", "#55554f"


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def load_cmip(model, experiment, out_dir):
    """Annual global-mean anomalies: surface temperature T and net downward TOA flux N."""
    path = out_dir / f"{model}_{experiment}_r1i1p1f1_anomalies.txt"
    if not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CMIP_URL.format(m=model, e=experiment), path)
    df = pd.read_csv(path, index_col=0)
    return df["tas"].to_numpy(), (df["rsdt"] - df["rsut"] - df["rlut"]).to_numpy()


def detrended_sd(x):
    t = np.arange(len(x))
    return float(np.std(x - np.polyval(np.polyfit(t, x, 1), t), ddof=2))


def load_cmip_historical(hist_dir):
    """Area-weighted global-mean annual tas from one-file-per-year monthly ClimateSet output."""
    days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])  # 365_day calendar
    years, gmst = [], []
    for year_dir in sorted(hist_dir.iterdir(), key=lambda p: int(p.name)):
        (fp,) = year_dir.glob("*.nc")
        with xr.open_dataset(fp, decode_times=False) as ds:
            monthly = ds["tas"].weighted(np.cos(np.deg2rad(ds["y"]))).mean(("y", "x")).values
        if len(monthly) != 12:
            raise ValueError(f"{fp} has {len(monthly)} months")
        years.append(int(year_dir.name))
        gmst.append(np.average(monthly, weights=days))
    return pd.Series(gmst, index=years)


# ---------------------------------------------------------------------------------------------
# energy balance model
# ---------------------------------------------------------------------------------------------
def make_ebm(p):
    return EnergyBalanceModel(
        ocean_heat_capacity=[p[f"ocean_heat_capacity[{i}]"] for i in range(3)],
        ocean_heat_transfer=[p[f"ocean_heat_transfer[{i}]"] for i in range(3)],
        deep_ocean_efficacy=p["deep_ocean_efficacy"],
        forcing_4co2=p["forcing_4co2"],
        gamma_autocorrelation=p["gamma_autocorrelation"],
    )


def simulate(p, n_years):
    """Deterministic abrupt-4xCO2 response as annual means (average of the bracketing timebounds)."""
    ebm = make_ebm(p)
    a, b = ebm.eb_matrix_d, ebm.forcing_vector_d
    f4 = p["forcing_4co2"]
    state = np.zeros((n_years + 1, a.shape[0]))  # timebounds x [forcing box, T1, T2, T3]
    for t in range(n_years):
        state[t + 1] = a @ state[t] + b * f4
    # TOA imbalance as FaIR computes it (calculate_toa_imbalance_postrun)
    n_bounds = (
        f4
        - p["ocean_heat_transfer[0]"] * state[:, 1]
        + (1 - p["deep_ocean_efficacy"]) * p["ocean_heat_transfer[2]"] * (state[:, 2] - state[:, 3])
    )
    return 0.5 * (state[:-1, 1] + state[1:, 1]), 0.5 * (n_bounds[:-1] + n_bounds[1:])


def fit(t_obs, n_obs, sd_t, sd_n, n_starts, seed):
    lo, hi = np.log([p[1] for p in PARAMS]), np.log([p[2] for p in PARAMS])
    x0 = np.log([p[3] for p in PARAMS])

    def resid(x):
        t, n = simulate(dict(zip(NAMES, np.exp(x))), len(t_obs))
        return np.concatenate([(t - t_obs) / sd_t, (n - n_obs) / sd_n])

    rng = np.random.default_rng(seed)
    best = None
    for i, start in enumerate([x0] + [rng.uniform(lo, hi) for _ in range(n_starts - 1)]):
        try:
            res = least_squares(resid, start, bounds=(lo, hi))
        except (ValueError, np.linalg.LinAlgError):
            continue
        print(f"  start {i}: cost {res.cost:10.1f}")
        if best is None or res.cost < best.cost:
            best = res
    return dict(zip(NAMES, np.exp(best.x))), best


def emergent(p):
    ebm = make_ebm(p)
    ebm.emergent_parameters()
    return float(ebm.ecs), float(ebm.tcr)


def gregory(t, n, years=150):
    """
    Gregory et al.

    (2004) regression of N on T over the first ``years`` years: (F_4x, lambda).
    """
    slope, intercept = np.polyfit(t[:years], n[:years], 1)
    return intercept, -slope


# ---------------------------------------------------------------------------------------------
# full FaIR runs
# ---------------------------------------------------------------------------------------------
def apply_climate(f, p, co2_scale):
    fill(f.climate_configs["ocean_heat_capacity"], [p[f"ocean_heat_capacity[{i}]"] for i in range(3)])
    fill(f.climate_configs["ocean_heat_transfer"], [p[f"ocean_heat_transfer[{i}]"] for i in range(3)])
    for k in ("deep_ocean_efficacy", "forcing_4co2", "gamma_autocorrelation"):
        fill(f.climate_configs[k], p[k])
    fill(f.climate_configs["stochastic_run"], False)
    fill(f.species_configs["forcing_scale"], co2_scale, specie="CO2")


def fair_abrupt4x(p, co2_scale, n_years):
    """Concentration-driven abrupt-4xCO2 in FaIR itself (CH4 and N2O at baseline)."""
    f = FAIR()
    f.define_time(0, n_years, 1)
    f.define_scenarios(["abrupt-4xCO2"])
    f.define_configs(["fit"])
    _, props = read_properties(filename=SPECIES_FILE)
    f.define_species(GHG, {s: {**props[s], "input_mode": "concentration"} for s in GHG})
    f.allocate()
    f.fill_species_configs(SPECIES_FILE)
    apply_climate(f, p, co2_scale)
    for s in GHG:
        base = float(f.species_configs["baseline_concentration"].sel(specie=s).isel(config=0))
        fill(f.concentration, base * (4 if s == "CO2" else 1), specie=s)
    for var in (f.forcing, f.temperature, f.cumulative_emissions, f.airborne_emissions, f.ocean_heat_content_change):
        initialise(var, 0)
    f.run(progress=False)
    erf = float(f.forcing.sel(specie="CO2").isel(timebounds=1, scenario=0, config=0))
    tb = f.temperature.sel(layer=0).isel(scenario=0, config=0).values
    return erf, 0.5 * (tb[:-1] + tb[1:])


def fair_historical(p=None, co2_scale=None):
    """1750-2014, all forcings, 841 calibrated members; climate replaced by ``p`` if given."""
    df_configs = pd.read_csv(PARAMS_FILE, index_col=0)
    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(1750, 2015, 1)
    f.define_scenarios(["ssp245"])
    f.define_configs(df_configs.index)
    species, properties = read_properties(filename=SPECIES_FILE)
    f.define_species(species, properties)
    f.allocate()
    f.fill_from_rcmip()
    f.fill_species_configs(SPECIES_FILE)
    f.override_defaults(PARAMS_FILE)
    for sp in ("Volcanic", "Solar"):
        fill(f.forcing, f.forcing.sel(specie=sp) * df_configs[f"forcing_scale[{sp}]"].values.squeeze(), specie=sp)
    if p is not None:
        apply_climate(f, p, co2_scale)
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    for var in (f.forcing, f.temperature, f.cumulative_emissions, f.airborne_emissions, f.ocean_heat_content_change):
        initialise(var, 0)
    f.run(progress=False)
    tb = f.temperature.sel(layer=0).isel(scenario=0).values  # (timebounds, config)
    years = np.arange(1750, 2015)
    annual = pd.DataFrame(0.5 * (tb[:-1] + tb[1:]), index=years)
    return annual - annual.loc[1850:1900].mean()


# ---------------------------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------------------------
def style(ax):
    ax.grid(color="#e4e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def stamp(fig, provenance):
    """Print where a figure came from in its bottom-left corner, so a copied PNG stays identifiable."""
    fig.text(0.005, 0.005, provenance, fontsize=7, color="#77776f", ha="left", va="bottom")


def plot_fit(model, t_obs, n_obs, t_fit, n_fit, t_fair, fit_years, provenance, path):
    yrs = np.arange(len(t_obs))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    axes[0].plot(yrs, t_obs, color=C_DATA, lw=0.8, label=f"{model}")
    axes[0].plot(yrs, t_fit, color=C_FIT, lw=2, label="fitted energy balance model")
    axes[0].plot(np.arange(len(t_fair)), t_fair, color=C_ALT, lw=2, ls="--", label="full FaIR with fitted parameters")
    axes[0].set(title="Surface temperature after CO$_2$ quadrupling", xlabel="years after quadrupling", ylabel="K")
    axes[1].plot(yrs, n_obs, color=C_DATA, lw=0.8, label=f"{model}")
    axes[1].plot(yrs, n_fit, color=C_FIT, lw=2, label="fitted energy balance model")
    axes[1].set(title="Top-of-atmosphere imbalance", xlabel="years after quadrupling", ylabel="W m$^{-2}$")
    axes[2].scatter(t_obs, n_obs, s=9, color=C_DATA, alpha=0.6, label=f"{model} (one dot per year)")
    axes[2].plot(t_fit, n_fit, color=C_FIT, lw=2, label="fitted energy balance model")
    axes[2].set(title="Gregory plot", xlabel="temperature (K)", ylabel="TOA imbalance (W m$^{-2}$)")
    if fit_years < len(t_obs):
        for ax in axes[:2]:
            ax.axvline(fit_years, color="#77776f", lw=1, ls=":")
            ax.text(fit_years, ax.get_ylim()[0], "  end of fit window", color="#77776f", fontsize=8, va="bottom")
    for ax in axes:
        style(ax)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{model} abrupt-4xCO2 vs FaIR fitted on years 0-{fit_years - 1}", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp(fig, provenance)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_historical(model, obs, generic, fitted, fit_years, provenance, path):
    fig, ax = plt.subplots(figsize=(10, 5.1))
    for df, color, label in (
        (generic, C_ALT, "FaIR, calibrated climate (841 members)"),
        (fitted, C_FIT, f"FaIR, climate fitted to {model} 4xCO2 yrs 0-{fit_years - 1} " "(841 forcing members)"),
    ):
        df = df.loc[1850:2014]
        q = df.quantile([0.05, 0.5, 0.95], axis=1)
        ax.fill_between(df.index, q.loc[0.05], q.loc[0.95], color=color, alpha=0.18, lw=0)
        ax.plot(df.index, q.loc[0.5], color=color, lw=2, label=f"{label}: median, 5-95% band")
    ax.plot(obs.index, obs.values, color=C_DATA, lw=1.2, label=f"{model} historical r1i1p1f1")
    ax.set(title="Global-mean surface warming relative to 1850-1900", xlabel="year", ylabel="K")
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp(fig, provenance)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def git_state():
    """Short commit hash, plus '+uncommitted' if this script differs from that commit."""

    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()

    try:
        dirty = git("status", "--porcelain", "--", Path(__file__).relative_to(REPO).as_posix())
        return (git("rev-parse", "--short", "HEAD") or "no-git") + ("+uncommitted" if dirty else "")
    except OSError:
        return "no-git"


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="NorESM2-LM")
    ap.add_argument("--n-starts", type=int, default=8, help="random restarts of the least-squares fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fit-years",
        type=int,
        default=500,
        help="fit on the first N years only (150 = standard CMIP length); errors are reported for all",
    )
    ap.add_argument("--check-historical", action="store_true")
    args = ap.parse_args()

    data_dir = REPO / "data" / "fair" / "cmip6" / args.model
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_fit{args.fit_years}y"
    run_dir = data_dir / "fits" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # a new folder per run: nothing is ever overwritten
    git = git_state()
    provenance = f"run {run_id} · {args.model} · scripts/{Path(__file__).name} @ {git}"
    lines = [
        f"run       {run_id}",
        f"command   python {' '.join(sys.argv)}",
        f"git       {git}",
        f"settings  {vars(args)}",
        "",
    ]

    def say(msg=""):
        print(msg)
        lines.append(msg)

    t_obs, n_obs = load_cmip(args.model, "abrupt-4xCO2", data_dir)
    t_pi, n_pi = load_cmip(args.model, "piControl", data_dir)
    sd_t, sd_n = detrended_sd(t_pi), detrended_sd(n_pi)
    nfit = min(args.fit_years, len(t_obs))
    say(
        f"{args.model}: {len(t_obs)} years of abrupt-4xCO2, fitting years 0-{nfit - 1}; "
        f"piControl noise sd T {sd_t:.3f} K, N {sd_n:.3f} W m-2"
    )

    f_greg, lam_greg = gregory(t_obs, n_obs)
    say(
        f"Gregory (years 0-149, from the data): F_4x {f_greg:.2f} W m-2, feedback {lam_greg:.2f} W m-2 K-1, "
        f"effective climate sensitivity {f_greg / lam_greg / 2:.2f} K"
    )

    print("fitting ...")
    p, res = fit(t_obs[:nfit], n_obs[:nfit], sd_t, sd_n, args.n_starts, args.seed)
    t_fit, n_fit = simulate(p, len(t_obs))
    ecs, tcr = emergent(p)
    at_limit = [nm for nm, lo, hi, _ in PARAMS if np.isclose(p[nm], lo, rtol=1e-2) or np.isclose(p[nm], hi, rtol=1e-2)]

    erf_unit, _ = fair_abrupt4x(p, 1.0, 1)
    co2_scale = p["forcing_4co2"] / erf_unit
    erf_check, t_fair = fair_abrupt4x(p, co2_scale, len(t_obs))
    p["forcing_scale[CO2]"] = co2_scale

    say("\nfitted parameters:")
    for k, v in p.items():
        say(f"  {k:24s} {v:9.3f}")
    say(f"  at the edge of their allowed range: {at_limit or 'none'}")
    say(f"\nemergent ECS {ecs:.2f} K, TCR {tcr:.2f} K   (weighted least-squares cost {res.cost:.1f})")
    for lo, hi in ((0, 10), (10, 150), (150, 450), (450, len(t_obs))):
        if lo >= len(t_obs):
            continue
        rt = np.sqrt(np.mean((t_fit[lo:hi] - t_obs[lo:hi]) ** 2))
        rn = np.sqrt(np.mean((n_fit[lo:hi] - n_obs[lo:hi]) ** 2))
        bias = np.mean(t_fit[lo:hi] - t_obs[lo:hi])
        say(
            f"  years {lo:3d}-{hi - 1:3d}: RMS error T {rt:.3f} K (mean bias {bias:+.2f} K), N {rn:.3f} W m-2"
            f"{'' if hi <= nfit else '   <- outside the fit window' if lo >= nfit else ''}"
        )
    say(f"FaIR CO2 forcing at 4x with forcing_scale 1: {erf_unit:.3f} W m-2 -> forcing_scale[CO2] = {co2_scale:.3f}")
    say(
        f"full FaIR check: CO2 forcing {erf_check:.3f} W m-2, max |T_FaIR - T_fit| = "
        f"{np.abs(t_fair - t_fit).max():.4f} K"
    )

    pd.DataFrame([p], index=[args.model]).to_csv(run_dir / "params.csv")
    plot_fit(args.model, t_obs, n_obs, t_fit, n_fit, t_fair, nfit, provenance, run_dir / "fit_4xco2.png")

    if args.check_historical:
        print("\nhistorical check (2 x 841 FaIR runs, 1750-2014) ...")
        obs = load_cmip_historical(Path(str(HIST_DIR).format(m=args.model)))
        obs = obs - obs.loc[1850:1900].mean()
        generic, fitted = fair_historical(), fair_historical(p, co2_scale)
        say("\nhistorical check, warming vs 1850-1900:")
        for name, df in (("calibrated climate", generic), (f"{args.model}-fitted", fitted)):
            med = df.median(axis=1)
            q = df.quantile([0.05, 0.95], axis=1)
            inside = ((obs >= q.loc[0.05, obs.index]) & (obs <= q.loc[0.95, obs.index])).mean()
            say(
                f"  {name:22s}: 1995-2014 mean warming median {med.loc[1995:2014].mean():.2f} K "
                f"(5-95% {q.loc[0.05, 1995:2014].mean():.2f}-{q.loc[0.95, 1995:2014].mean():.2f}); "
                f"RMS vs {args.model} {np.sqrt(((med.loc[obs.index] - obs) ** 2).mean()):.3f} K; "
                f"{100 * inside:.0f}% of {args.model} years inside 5-95%"
            )
        say(f"  {args.model:22s}: 1995-2014 mean warming {obs.loc[1995:2014].mean():.2f} K")
        plot_historical(args.model, obs, generic, fitted, nfit, provenance, run_dir / "historical.png")

    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {run_dir}/ ({', '.join(sorted(x.name for x in run_dir.iterdir()))})")


if __name__ == "__main__":
    main()
