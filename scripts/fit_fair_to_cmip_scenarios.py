"""
Calibrate FaIR's climate to a CMIP6 model's historical + SSP runs (default NorESM2-LM).

This is the FaIR used to make (pre)training data. It must NOT see NorESM2-LM's abrupt-4xCO2 run: that run is
the thesis's held-out test (decided 2026-10-26, FAIR_STATE.md). Nothing here reads 4xCO2 data.

Data
----
Global-mean annual surface temperature from the local ClimateSet copy of the model's runs
(Climateset_DATA/outputs/CMIP6/<model>/<member>/<scenario>/tas/250_km/mon/<year>/*.nc): historical
1850-2014 and every SSP member found. Anomalies relative to the r1i1p1f1 historical 1850-1900 mean.
piControl global means (fair-calibrate repository) give the noise level used to express the fit cost.

Method
------
1. Forcing. FaIR is run once over 1750-2100 for the SSPs, with all greenhouse gases concentration-driven (as in
   the CMIP6 runs, so FaIR's carbon cycle plays no role) and all other forcings from the calibrated setup
   (fair-calibrate v1.4.1, 841 members). For each species the efficacy-weighted forcing is taken as the median
   over members, and the species are summed into three groups: aerosols (radiation + cloud interactions),
   volcanic, and everything else. For greenhouse gases this median equals the forcing with the median forcing
   scale, because concentrations are the same in every member.
2. Fit. The deterministic three-layer energy balance model, stepped exactly as FaIR steps it (state at
   timebound i+1 from state at i and forcing at i+1), is fitted to the model's temperature by least squares.
   Free: heat capacities, heat transfer coefficients, deep-ocean efficacy, gamma, and a scale factor on the
   aerosol forcing and on the volcanic forcing. The non-aerosol, non-volcanic forcing is fixed: with temperature
   alone, scaling all forcing, heat capacities and heat transfers together leaves temperature unchanged, so an
   overall forcing scale cannot be fitted.
3. Checks. (a) The stepping is verified against a FaIR run. (b) The aerosol scale is profiled (fixed on a grid,
   everything else refitted) to show how well the data separate aerosol forcing from climate sensitivity.
   (c) The fitted aerosol forcing is compared with the model's own aerosol ERF series where fair-calibrate
   publishes one (data/smith2023aerosol/).

Usage
-----
    python scripts/fit_fair_to_cmip_scenarios.py --model NorESM2-LM

Outputs: a new folder data/fair/cmip6/<model>/fits_scenarios/<YYYYmmdd-HHMMSS>_hist-ssp/ (never overwritten) with
params.csv, summary.txt and figures stamped with the run id. Reading the netCDF files needs h5netcdf or netCDF4.
"""

import argparse
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
from fair.interface import fill, initialise  # noqa: E402
from fair.io import read_properties  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
# generic helpers only; nothing from that script reads 4xCO2 data when imported
from fit_fair_to_cmip_4xco2 import (  # noqa: E402
    C_ALT,
    C_DATA,
    C_FIT,
    detrended_sd,
    emergent,
    fair_abrupt4x,
    git_state,
    load_cmip,
    make_ebm,
    stamp,
    style,
)

REPO = Path(__file__).resolve().parent.parent
CAL = REPO / "data" / "fair" / "calibrated_constrained_ensemble"
PARAMS_FILE = CAL / "calibrated_constrained_parameters_calibration1.4.1.csv"
SPECIES_FILE = CAL / "species_configs_properties_calibration1.4.1.csv"
CLIMATESET = REPO / "Climateset_DATA" / "outputs" / "CMIP6"
AER_URL = (
    "https://raw.githubusercontent.com/chrisroadmap/fair-calibrate/main/data/smith2023aerosol/"
    "{m}_{r}_aerosol_forcing.csv"
)
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
AEROSOL = ["Aerosol-radiation interactions", "Aerosol-cloud interactions"]
Y0, Y1 = 1750, 2101  # FaIR timebounds

# name, lower, upper, first guess (same units as fit_fair_to_cmip_4xco2.py); the last two scale forcing groups
PARAMS = [
    ("ocean_heat_capacity[0]", 1.0, 30.0, 5.0),
    ("ocean_heat_capacity[1]", 2.0, 200.0, 20.0),
    ("ocean_heat_capacity[2]", 10.0, 2000.0, 100.0),
    ("ocean_heat_transfer[0]", 0.2, 4.0, 1.2),
    ("ocean_heat_transfer[1]", 0.2, 8.0, 2.0),
    ("ocean_heat_transfer[2]", 0.05, 4.0, 0.8),
    ("deep_ocean_efficacy", 0.3, 4.0, 1.3),
    ("gamma_autocorrelation", 0.5, 50.0, 2.0),
    ("aerosol_scale", 0.1, 3.0, 1.0),
    ("volcanic_scale", 0.2, 2.0, 1.0),
]
NAMES = [p[0] for p in PARAMS]


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def global_mean_series(year_dirs):
    days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])  # 365_day calendar
    out = {}
    for year_dir in year_dirs:
        (fp,) = year_dir.glob("*.nc")
        with xr.open_dataset(fp, decode_times=False) as ds:
            monthly = ds["tas"].weighted(np.cos(np.deg2rad(ds["y"]))).mean(("y", "x")).values
        if len(monthly) != 12:
            raise ValueError(f"{fp} has {len(monthly)} months")
        out[int(year_dir.name)] = np.average(monthly, weights=days)
    return pd.Series(out).sort_index()


def load_targets(model):
    """{(scenario, member): annual global-mean tas anomaly vs r1i1p1f1 historical 1850-1900}."""
    root = CLIMATESET / model
    series = {}
    for member_dir in sorted(root.iterdir()):
        for scen in ["historical"] + SSPS:
            d = member_dir / scen / "tas" / "250_km" / "mon"
            if d.is_dir():
                series[(scen, member_dir.name)] = global_mean_series(sorted(d.iterdir(), key=lambda p: int(p.name)))
    base = series[("historical", "r1i1p1f1")].loc[1850:1900].mean()
    return {k: v - base for k, v in series.items()}


def load_model_aerosol(model, out_dir):
    """
    The model's own aerosol ERF series from fair-calibrate (Smith et al.

    2023), if published.
    """
    out = {}
    for r in ("r1i1p2f1", "r2i1p2f1", "r3i1p2f1"):
        path = out_dir / f"{model}_{r}_aerosol_forcing.csv"
        if not path.exists():
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(AER_URL.format(m=model, r=r), path)
            except OSError:
                continue
        out[r] = pd.read_csv(path, index_col=0)["ERF"]
    return out


# ---------------------------------------------------------------------------------------------
# FaIR forcing
# ---------------------------------------------------------------------------------------------
def fair_forcing():
    """Run calibrated FaIR (GHGs concentration-driven) for the SSPs; return grouped median forcing and the run."""
    df_configs = pd.read_csv(PARAMS_FILE, index_col=0)
    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(Y0, Y1, 1)
    f.define_scenarios(SSPS)
    f.define_configs(df_configs.index)
    species, props = read_properties(filename=SPECIES_FILE)
    for s in species:
        if props[s]["greenhouse_gas"]:
            props[s]["input_mode"] = "concentration"
    f.define_species(species, props)
    f.allocate()
    f.fill_from_rcmip()
    f.fill_species_configs(SPECIES_FILE)
    f.override_defaults(PARAMS_FILE)
    # override_defaults switches FaIR's stochastic internal variability ON for every member; this run is for
    # forcing (and the stepping check), so keep it deterministic
    fill(f.climate_configs["stochastic_run"], False)
    for sp in ("Volcanic", "Solar"):
        fill(f.forcing, f.forcing.sel(specie=sp) * df_configs[f"forcing_scale[{sp}]"].values.squeeze(), specie=sp)
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    for var in (f.forcing, f.temperature, f.cumulative_emissions, f.airborne_emissions, f.ocean_heat_content_change):
        initialise(var, 0)
    f.run(progress=False)

    eff = f.forcing * f.species_configs["forcing_efficacy"]  # (timebounds, scenario, config, specie)
    med = eff.median("config", skipna=True).fillna(0.0)  # per species
    groups = {
        "aerosol": med.sel(specie=AEROSOL).sum("specie"),
        "volcanic": med.sel(specie="Volcanic"),
    }
    groups["other"] = med.sum("specie") - groups["aerosol"] - groups["volcanic"]
    forcing = {k: v.transpose("timebounds", "scenario").values for k, v in groups.items()}
    scales = {g: float(df_configs[f"forcing_scale[{g}]"].median()) for g in ("CO2", "CH4", "N2O")}
    # member spread of the aerosol and volcanic forcing relative to the median, for the prior on their scales
    aer_members = (
        eff.sel(specie=AEROSOL).sum("specie").sel(scenario="ssp245", timebounds=slice(1995, 2014)).mean("timebounds")
    )
    spread = {
        "aerosol_scale": float(np.log(aer_members / aer_members.median()).std()),
        "volcanic_scale": float(np.log(df_configs["forcing_scale[Volcanic]"]).std()),
    }
    return forcing, scales, spread, f


CLIMATE = NAMES[:8]


def make_prior(spread):
    """Gaussian prior in log space: climate parameters from the 841 calibrated members (full covariance, so
    their observed correlations are kept); aerosol and volcanic scales centred on 1 with the members' spread."""
    df = pd.read_csv(PARAMS_FILE, index_col=0)
    logs = np.log(df[CLIMATE].to_numpy())
    mu, cov = logs.mean(axis=0), np.cov(logs, rowvar=False)
    chol = np.linalg.cholesky(cov)
    return {"mu": mu, "chol": chol, "sd": spread}


# ---------------------------------------------------------------------------------------------
# energy balance model on time-varying forcing
# ---------------------------------------------------------------------------------------------
def simulate(p, forcing):
    """Surface temperature on timebounds (n_tb, n_scen), stepped as FaIR does."""
    ebm = make_ebm({**p, "forcing_4co2": 8.0})  # forcing_4co2 does not enter the dynamics
    a, b = ebm.eb_matrix_d, ebm.forcing_vector_d
    f_tot = forcing["other"] + p["aerosol_scale"] * forcing["aerosol"] + p["volcanic_scale"] * forcing["volcanic"]
    state = np.zeros((f_tot.shape[0], f_tot.shape[1], a.shape[0]))
    for i in range(f_tot.shape[0] - 1):
        state[i + 1] = state[i] @ a.T + b[None, :] * f_tot[i + 1, :, None]
    return state[..., 1]


def annual_anomaly(t_bounds):
    """Annual means for years Y0..Y1-1 (average of bracketing timebounds), anomaly vs 1850-1900."""
    annual = 0.5 * (t_bounds[:-1] + t_bounds[1:])
    years = np.arange(Y0, Y1)
    ref = annual[(years >= 1850) & (years <= 1900)].mean(axis=0)
    return pd.DataFrame(annual - ref, index=years, columns=SSPS)


def model_series(p, forcing):
    return annual_anomaly(simulate(p, forcing))


def residuals_fn(forcing, targets, sd_t, fixed, prior):
    """Data residuals in units of piControl sd, followed by prior residuals (whitened log-parameter deviations)."""
    free = [n for n in NAMES if n not in fixed]

    def resid(x):
        p = {**fixed, **dict(zip(free, np.exp(x)))}
        m = model_series(p, forcing)
        out = []
        for (scen, _), obs in targets.items():
            col = "ssp245" if scen == "historical" else scen  # RCMIP SSPs share the same 1750-2014 history
            out.append((m.loc[obs.index, col].values - obs.values) / sd_t)
        if prior is not None:
            out.append(np.linalg.solve(prior["chol"], np.log([p[n] for n in CLIMATE]) - prior["mu"]))
            out.append(
                np.array([np.log(p[n]) / prior["sd"][n] for n in ("aerosol_scale", "volcanic_scale") if n not in fixed])
            )
        return np.concatenate(out)

    return free, resid


def fit(forcing, targets, sd_t, n_starts, seed, fixed=None, prior=None):
    fixed = fixed or {}
    free, resid = residuals_fn(forcing, targets, sd_t, fixed, prior)
    bnd = {n: (lo, hi, g) for n, lo, hi, g in PARAMS}
    lo = np.log([bnd[n][0] for n in free])
    hi = np.log([bnd[n][1] for n in free])
    x0 = np.log([bnd[n][2] for n in free])
    rng = np.random.default_rng(seed)
    best = None
    for start in [x0] + [rng.uniform(lo, hi) for _ in range(n_starts - 1)]:
        try:
            res = least_squares(resid, start, bounds=(lo, hi))
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or res.cost < best.cost:
            best = res
    return {**fixed, **dict(zip(free, np.exp(best.x)))}, best


# ---------------------------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------------------------
def plot_fit(model, targets, m_fit, m_default, provenance, path):
    panels = ["historical"] + SSPS
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), sharey=False)
    for ax, scen in zip(axes, panels):
        col = "ssp245" if scen == "historical" else scen
        span = slice(1850, 2014) if scen == "historical" else slice(2015, 2100)
        members = [(mem, s) for (sc, mem), s in targets.items() if sc == scen]
        for i, (mem, s) in enumerate(members):
            ax.plot(
                s.index,
                s.values,
                color=C_DATA,
                lw=0.9,
                alpha=0.9 if i == 0 else 0.45,
                label=f"{model} ({len(members)} member{'s' if len(members) > 1 else ''})" if i == 0 else None,
            )
        ax.plot(
            m_default.loc[span].index,
            m_default.loc[span, col],
            color=C_ALT,
            lw=1.6,
            ls="--",
            label="FaIR before fit (median forcing, aerosol x1)",
        )
        ax.plot(m_fit.loc[span].index, m_fit.loc[span, col], color=C_FIT, lw=2.2, label="FaIR fitted")
        ax.set(title=scen, xlabel="year")
        style(ax)
    axes[0].set_ylabel("warming vs 1850-1900 (K)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"FaIR calibrated to {model} historical + SSPs (4xCO2 not used: held-out test)", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp(fig, provenance)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profile(prof, best, provenance, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, label in zip(
        axes, ("cost", "ecs", "tcr"), ("fit cost (lower = better)", "emergent ECS (K)", "emergent TCR (K)")
    ):
        ax.plot(prof["aerosol_scale"], prof[key], color=C_FIT, lw=2, marker="o", ms=5)
        ax.axvline(best, color=C_DATA, lw=1, ls=":")
        ax.set(xlabel="aerosol forcing scale (fixed)", title=label)
        style(ax)
    fig.suptitle("How well do the data separate aerosol forcing from climate sensitivity?", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp(fig, provenance)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_aerosol(model, forcing, scale, model_aer, provenance, path):
    years = np.arange(Y0, Y1 + 1)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (r, s) in enumerate(model_aer.items()):
        s = s.loc[1850:2014]
        label = f"{model} own aerosol ERF, Smith et al. 2023 (p2 runs, {len(model_aer)} members)"
        ax.plot(s.index, s.values, color=C_DATA, lw=0.8, alpha=0.5, label=label if i == 0 else None)
    sel = (years >= 1850) & (years <= 2014)
    hist = forcing["aerosol"][:, SSPS.index("ssp245")]
    ax.plot(years[sel], hist[sel], color=C_ALT, lw=1.6, ls="--", label="FaIR median aerosol forcing")
    ax.plot(years[sel], scale * hist[sel], color=C_FIT, lw=2.2, label=f"FaIR aerosol forcing x {scale:.2f} (fitted)")
    ax.set(title="Aerosol effective radiative forcing, 1850-2014", xlabel="year", ylabel="W m$^{-2}$")
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp(fig, provenance)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="NorESM2-LM")
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--profile",
        type=str,
        default="0.3,0.5,0.7,0.85,1.0,1.15,1.3,1.5,1.75,2.0,2.5",
        help="aerosol scales at which to refit everything else ('' to skip)",
    )
    ap.add_argument(
        "--prior",
        choices=("calibrated", "none"),
        default="calibrated",
        help="'calibrated': keep parameters the data do not constrain near the 841 calibrated members",
    )
    ap.add_argument(
        "--fix-aerosol-scale",
        type=float,
        default=None,
        help="hold the aerosol scale at this value instead of fitting it (e.g. 1.0 = FaIR median)",
    )
    args = ap.parse_args()

    data_dir = REPO / "data" / "fair" / "cmip6" / args.model
    tag = "" if args.fix_aerosol_scale is None else f"_aer{args.fix_aerosol_scale:.2f}"
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_hist-ssp{tag}"
    run_dir = data_dir / "fits_scenarios" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    git = git_state()
    provenance = f"run {run_id} · {args.model} · scripts/{Path(__file__).name} @ {git}"
    lines = [
        f"run       {run_id}",
        f"command   python {' '.join(sys.argv)}",
        f"git       {git}",
        f"settings  {vars(args)}",
        "data      historical + SSPs only; abrupt-4xCO2 NOT used (held-out test)",
        "",
    ]

    def say(msg=""):
        print(msg)
        lines.append(msg)

    targets = load_targets(args.model)
    say("targets (annual global-mean tas, anomaly vs r1i1p1f1 historical 1850-1900):")
    for (scen, mem), s in targets.items():
        say(f"  {scen:10s} {mem}: {s.index[0]}-{s.index[-1]}  last-decade mean {s.iloc[-10:].mean():.2f} K")
    t_pi, _ = load_cmip(args.model, "piControl", data_dir)
    sd_t = detrended_sd(t_pi)
    say(f"piControl year-to-year sd of T: {sd_t:.3f} K (cost unit)")

    print("running calibrated FaIR for forcing (841 members x 4 SSPs) ...")
    forcing, ghg_scales, spread, f_cal = fair_forcing()
    prior = make_prior(spread) if args.prior == "calibrated" else None
    say(f"median forcing scales used for the fixed GHG forcing: {ghg_scales}")
    say(
        f"prior: {args.prior}"
        + (
            f" (log-sd of aerosol scale {spread['aerosol_scale']:.2f}, "
            f"volcanic scale {spread['volcanic_scale']:.2f})"
            if prior
            else ""
        )
    )
    for g in ("other", "aerosol", "volcanic"):
        say(
            f"  forcing group {g:8s}: 2014 {forcing[g][2014 - Y0, 1]:+.2f} W m-2, "
            f"2100 ssp126/245/370/585 " + "/".join(f"{forcing[g][-1, j]:+.2f}" for j in range(4))
        )

    # check (a): my stepping reproduces FaIR for one calibrated member
    i_cfg = 0
    cc = f_cal.climate_configs.isel(config=i_cfg)
    p_chk = {f"ocean_heat_capacity[{i}]": float(cc["ocean_heat_capacity"][i]) for i in range(3)}
    p_chk.update({f"ocean_heat_transfer[{i}]": float(cc["ocean_heat_transfer"][i]) for i in range(3)})
    p_chk.update({k: float(cc[k]) for k in ("deep_ocean_efficacy", "gamma_autocorrelation")})
    one = (f_cal.forcing * f_cal.species_configs["forcing_efficacy"]).isel(config=i_cfg).sum("specie", skipna=True)
    t_mine = simulate(
        {**p_chk, "aerosol_scale": 0.0, "volcanic_scale": 0.0},
        {
            "other": one.transpose("timebounds", "scenario").values,
            "aerosol": 0 * one.values,
            "volcanic": 0 * one.values,
        },
    )
    t_fair = f_cal.temperature.sel(layer=0).isel(config=i_cfg).transpose("timebounds", "scenario").values
    say(f"check: my stepping vs FaIR (member {i_cfg}, all SSPs): max |diff| {np.abs(t_mine - t_fair).max():.2e} K")

    print("fitting ...")
    fixed = None if args.fix_aerosol_scale is None else {"aerosol_scale": args.fix_aerosol_scale}
    p, res = fit(forcing, targets, sd_t, args.n_starts, args.seed, fixed=fixed, prior=prior)
    if fixed:
        say(f"aerosol scale held fixed at {args.fix_aerosol_scale}")
    n_data = sum(len(s) for s in targets.values())
    data_rms = np.sqrt(np.mean(res.fun[:n_data] ** 2))
    erf_unit, _ = fair_abrupt4x({**p, "forcing_4co2": 8.0}, 1.0, 1)  # FaIR's CO2 forcing at 4x, scale 1
    eff_co2 = float(f_cal.species_configs["forcing_efficacy"].sel(specie="CO2").median())
    p["forcing_4co2"] = erf_unit * ghg_scales["CO2"] * eff_co2
    for g, v in ghg_scales.items():
        p[f"forcing_scale[{g}]"] = v
    ecs, tcr = emergent(p)
    m_fit = model_series(p, forcing)
    m_default = model_series({**p, "aerosol_scale": 1.0, "volcanic_scale": 1.0}, forcing)

    say("\nfitted parameters:")
    for k, v in p.items():
        say(f"  {k:24s} {v:9.3f}")
    at_limit = [n for n, lo, hi, _ in PARAMS if np.isclose(p[n], lo, rtol=1e-2) or np.isclose(p[n], hi, rtol=1e-2)]
    say(f"  at the edge of their allowed range: {at_limit or 'none'}")
    data_cost, prior_cost = 0.5 * np.sum(res.fun[:n_data] ** 2), 0.5 * np.sum(res.fun[n_data:] ** 2)
    say(
        f"\nemergent ECS {ecs:.2f} K, TCR {tcr:.2f} K   (cost {res.cost:.1f} = data {data_cost:.1f} + prior "
        f"{prior_cost:.1f}; {n_data} data points, RMS data residual {data_rms:.2f} x piControl sd)"
    )
    for (scen, mem), obs in targets.items():
        col = "ssp245" if scen == "historical" else scen
        err = m_fit.loc[obs.index, col] - obs
        say(
            f"  {scen:10s} {mem}: RMS {np.sqrt((err ** 2).mean()):.3f} K, mean bias {err.mean():+.3f} K, "
            f"last-decade bias {err.iloc[-10:].mean():+.2f} K"
        )

    if args.profile:
        print("profiling the aerosol scale ...")
        rows = []
        for a in [float(x) for x in args.profile.split(",")]:
            pa, ra = fit(
                forcing, targets, sd_t, max(2, args.n_starts // 2), args.seed, fixed={"aerosol_scale": a}, prior=prior
            )
            pa = {**pa, "forcing_4co2": p["forcing_4co2"]}
            e, t = emergent(pa)
            rows.append(
                {
                    "aerosol_scale": a,
                    "cost": ra.cost,
                    "data_cost": 0.5 * np.sum(ra.fun[:n_data] ** 2),
                    "ecs": e,
                    "tcr": t,
                }
            )
        prof = pd.DataFrame(rows)
        say("\naerosol-scale profile (aerosol scale fixed, everything else refitted):")
        say(prof.round(3).to_string(index=False))
        prof.to_csv(run_dir / "aerosol_profile.csv", index=False)
        plot_profile(prof, p["aerosol_scale"], provenance, run_dir / "aerosol_profile.png")

    model_aer = load_model_aerosol(args.model, data_dir / "aerosol_forcing_smith2023")
    if model_aer:
        fa = pd.Series(forcing["aerosol"][:-1, 1], index=np.arange(Y0, Y1))
        say("\naerosol ERF, mean over 1995-2014 (W m-2):")
        say(
            f"  FaIR median {fa.loc[1995:2014].mean():+.2f}; fitted x{p['aerosol_scale']:.2f} -> "
            f"{p['aerosol_scale'] * fa.loc[1995:2014].mean():+.2f}"
        )
        for r, s in model_aer.items():
            say(f"  {args.model} {r} (Smith et al. 2023): {s.loc[1995:2014].mean():+.2f}")
        plot_aerosol(args.model, forcing, p["aerosol_scale"], model_aer, provenance, run_dir / "aerosol_forcing.png")

    pd.DataFrame([p], index=[args.model]).to_csv(run_dir / "params.csv")
    plot_fit(args.model, targets, m_fit, m_default, provenance, run_dir / "fit_scenarios.png")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {run_dir}/ ({', '.join(sorted(x.name for x in run_dir.iterdir()))})")


if __name__ == "__main__":
    main()
