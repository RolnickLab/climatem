"""
Generate FaIR scenarios in which CO2 and CH4 do NOT move together.

In the historical record CO2 and CH4 rise together, so their effects on temperature cannot be told
apart. This script makes training data where they are decoupled, following the 2026-10-21 meeting:

- hold one variable fixed while the other varies     -> families ``co2_only`` and ``ch4_only``
- CO2 going up while CH4 goes down, and vice versa   -> family ``anticorrelated``; pairs more negatively
  correlated than ``--max-abs-corr-anti`` are redrawn, because a perfect mirror image is just as
  impossible to disentangle as perfect co-movement; pairs less negatively correlated than
  ``--min-abs-corr-anti`` are redrawn too, because a rise-and-fall shape against a falling one can end up
  uncorrelated or even moving together, which is not "opposite directions"
- both varying, but never perfectly correlated       -> family ``mixed``; any pair whose correlation
  exceeds ``--max-abs-corr`` is rejected and redrawn, so CO2 never simply scales with CH4

Each path is drawn from several shapes (ramp up, ramp down, step, overshoot, smooth random walk),
expressed as a multiplier of the pre-industrial baseline concentration and built in log space (CO2
forcing is roughly logarithmic in concentration). Every path starts at the baseline, so the climate
starts in equilibrium.

FaIR is run concentration-driven for CO2, CH4 and N2O (N2O held at baseline). By default CH4's indirect
forcing is included: ozone and stratospheric water vapour are computed from the concentrations (switch off
with ``--no-ch4-indirect``). Optionally, FaIR's stochastic mode adds internal variability.

Climate, two options:
- default: members of the calibrated, observationally constrained ensemble (fair-calibrate v1.4.1), each with
  its own climate and forcing parameters (``--n-configs``);
- ``--climate-params``: one config per params.csv written by scripts/fit_fair_to_cmip_scenarios.py, e.g. the
  NorESM2-LM-like variants. Their climate comes from the file; forcing parameters the fit does not set
  (ozone efficiencies, ozone and water-vapour forcing scales, internal-variability sizes) are the calibrated
  medians, as in the fit. Files under a folder whose README.txt says TEST-ONLY (the NorESM 4xCO2 fits, the
  thesis's held-out test) are refused.

Usage
-----
    python scripts/generate_fair_scenarios.py --n-per-family 20 --n-configs 50 --years 200
    python scripts/generate_fair_scenarios.py --climate-params \
        data/fair/cmip6/NorESM2-LM/fits_scenarios/20261026-231257_hist-ssp/params.csv \
        data/fair/cmip6/NorESM2-LM/fits_scenarios/20261026-232046_hist-ssp_aer1.00/params.csv

Output: one netCDF file with dims (scenario, config, year) holding concentrations, effective radiative
forcing per gas, and surface temperature anomaly relative to pre-industrial. By default it is
data/fair/scenarios/<YYYYmmdd-HHMMSS>_fair_co2_ch4.nc; an existing file is never overwritten.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

REPO = Path(__file__).resolve().parent.parent
CAL = REPO / "data" / "fair" / "calibrated_constrained_ensemble"
PARAMS_FILE = CAL / "calibrated_constrained_parameters_calibration1.4.1.csv"
SPECIES_FILE = CAL / "species_configs_properties_calibration1.4.1.csv"
GHG = ["CO2", "CH4", "N2O"]
CH4_INDIRECT = ["Ozone", "Stratospheric water vapour"]
ERF_NAMES = {
    "CO2": "erf_co2",
    "CH4": "erf_ch4",
    "N2O": "erf_n2o",
    "Ozone": "erf_ozone",
    "Stratospheric water vapour": "erf_strat_h2o",
}

UP_SHAPES = ("ramp_up", "step_up", "overshoot_up")
DOWN_SHAPES = ("ramp_down", "step_down", "overshoot_down")
ALL_SHAPES = UP_SHAPES + DOWN_SHAPES + ("random_walk",)


# ---------------------------------------------------------------------------------------------
# path shapes, as log-multipliers of the baseline concentration (0 = baseline)
# ---------------------------------------------------------------------------------------------
def make_path(shape, n, rng, log_lo, log_hi):
    """Return a length-n log-multiplier path that starts at 0 and stays within [log_lo, log_hi]."""
    t = np.arange(n)
    start = int(rng.uniform(0.05, 0.3) * n)  # quiet period at baseline before anything happens
    length = int(rng.uniform(0.15, 0.5) * n)  # duration of the change
    end = min(start + length, n - 1)
    ramp = np.clip((t - start) / max(end - start, 1), 0, 1)
    up, down = rng.uniform(0.3, 1.0) * log_hi, rng.uniform(0.3, 1.0) * log_lo

    if shape == "ramp_up":
        return up * ramp
    if shape == "ramp_down":
        return down * ramp
    if shape == "step_up":
        return np.where(t >= start, up, 0.0)
    if shape == "step_down":
        return np.where(t >= start, down, 0.0)
    if shape in ("overshoot_up", "overshoot_down"):
        peak = up if shape == "overshoot_up" else down
        settle = rng.uniform(0.0, 0.6) * peak  # partially returns towards baseline
        back = int(rng.uniform(0.1, 0.3) * n)
        rise = peak * ramp
        fall = np.clip((t - end) / max(back, 1), 0, 1)
        return rise - (peak - settle) * fall
    if shape == "random_walk":
        # smoothed random walk: integrated noise, then a moving average, scaled into the bounds
        x = np.cumsum(rng.normal(size=n))
        k = max(n // 20, 3)
        x = np.convolve(x, np.ones(k) / k, mode="same")
        x = x - x[start] if start < n else x
        x[:start] = 0.0
        span = np.max(np.abs(x)) or 1.0
        return np.clip(x / span * rng.uniform(0.3, 1.0) * max(log_hi, -log_lo), log_lo, log_hi)
    raise ValueError(f"unknown shape {shape!r}")


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan  # one variable held fixed: correlation undefined
    return float(np.corrcoef(a, b)[0, 1])


def draw_scenarios(n_per_family, n, rng, bounds, max_abs_corr, max_abs_corr_anti, min_abs_corr_anti):
    """Return a list of dicts: family, shapes, CO2 and CH4 log-multiplier paths, their correlation."""
    out = []
    zero = np.zeros(n)
    co2_lo, co2_hi = bounds["CO2"]
    ch4_lo, ch4_hi = bounds["CH4"]

    for _ in range(n_per_family):  # CO2 varies, CH4 held at baseline
        s = rng.choice(ALL_SHAPES)
        out.append(
            dict(
                family="co2_only", co2_shape=s, ch4_shape="constant", co2=make_path(s, n, rng, co2_lo, co2_hi), ch4=zero
            )
        )
    for _ in range(n_per_family):  # CH4 varies, CO2 held at baseline
        s = rng.choice(ALL_SHAPES)
        out.append(
            dict(
                family="ch4_only", co2_shape="constant", ch4_shape=s, co2=zero, ch4=make_path(s, n, rng, ch4_lo, ch4_hi)
            )
        )
    drawn = 0
    while drawn < n_per_family:  # opposite directions, alternating which one goes up
        if drawn % 2 == 0:
            sc, sm = rng.choice(UP_SHAPES), rng.choice(DOWN_SHAPES)
        else:
            sc, sm = rng.choice(DOWN_SHAPES), rng.choice(UP_SHAPES)
        c, m = make_path(sc, n, rng, co2_lo, co2_hi), make_path(sm, n, rng, ch4_lo, ch4_hi)
        r = corr(c, m)
        if r < -max_abs_corr_anti:  # a mirror image is as collinear as co-movement
            continue
        if r > -min_abs_corr_anti:  # overshoot shapes can end up uncorrelated or co-moving: not opposite
            continue
        out.append(dict(family="anticorrelated", co2_shape=sc, ch4_shape=sm, co2=c, ch4=m))
        drawn += 1
    drawn = 0
    while drawn < n_per_family:  # both vary independently; reject near-collinear pairs
        sc, sm = rng.choice(ALL_SHAPES), rng.choice(ALL_SHAPES)
        c, m = make_path(sc, n, rng, co2_lo, co2_hi), make_path(sm, n, rng, ch4_lo, ch4_hi)
        r = corr(c, m)
        if np.isnan(r) or abs(r) > max_abs_corr:
            continue
        out.append(dict(family="mixed", co2_shape=sc, ch4_shape=sm, co2=c, ch4=m))
        drawn += 1

    for d in out:
        d["corr"] = corr(d["co2"], d["ch4"])
    return out


# ---------------------------------------------------------------------------------------------
# FaIR
# ---------------------------------------------------------------------------------------------
def load_climate_params(paths):
    """One row per params.csv from fit_fair_to_cmip_scenarios.py, named after its run folder."""
    rows = []
    for path in paths:
        path = Path(path).resolve()
        for d in list(path.parents)[:3]:
            readme = d / "README.txt"
            if readme.exists() and "TEST-ONLY" in readme.read_text():
                raise SystemExit(
                    f"refused: {path} lies under {d}, marked TEST-ONLY (held-out test data). "
                    "Never generate training data from it."
                )
        row = pd.read_csv(path, index_col=0).iloc[0]
        row.name = path.parent.name
        rows.append(row)
    return pd.DataFrame(rows)


def apply_climate_params(f, climate, species):
    """Fitted climate per config; forcing parameters the fit does not set come from the calibrated medians."""
    med = pd.read_csv(PARAMS_FILE, index_col=0).median()
    for name, row in climate.iterrows():
        fill(
            f.climate_configs["ocean_heat_capacity"], [row[f"ocean_heat_capacity[{i}]"] for i in range(3)], config=name
        )
        fill(
            f.climate_configs["ocean_heat_transfer"], [row[f"ocean_heat_transfer[{i}]"] for i in range(3)], config=name
        )
        for k in ("deep_ocean_efficacy", "forcing_4co2", "gamma_autocorrelation"):
            fill(f.climate_configs[k], row[k], config=name)
        for k in ("sigma_eta", "sigma_xi"):  # not estimated by the fit
            fill(f.climate_configs[k], med[k], config=name)
        for s in species:
            key = f"forcing_scale[{s}]"
            value = row[key] if key in row.index else med.get(key)
            if value is not None and np.isfinite(value):
                fill(f.species_configs["forcing_scale"], value, config=name, specie=s)
        if "Ozone" in species:
            for s in ("CH4", "N2O"):
                fill(
                    f.species_configs["ozone_radiative_efficiency"],
                    med[f"ozone_radiative_efficiency[{s}]"],
                    config=name,
                    specie=s,
                )


def run_fair(scens, n_years, df_configs, stochastic, seed, climate=None, ch4_indirect=True):
    f = FAIR()
    f.define_time(0, n_years, 1)
    names = [f"s{i:04d}" for i in range(len(scens))]
    f.define_scenarios(names)
    configs = df_configs.index if climate is None else climate.index
    f.define_configs(configs)
    _, props = read_properties(filename=SPECIES_FILE)
    species = GHG + (CH4_INDIRECT if ch4_indirect else [])
    f.define_species(
        species, {s: {**props[s], **({"input_mode": "concentration"} if s in GHG else {})} for s in species}
    )
    f.allocate()
    f.fill_species_configs(SPECIES_FILE)
    if climate is None:
        f.override_defaults(PARAMS_FILE)  # calibrated energy balance + forcing parameters, per member
    else:
        apply_climate_params(f, climate, species)
    fill(f.climate_configs["stochastic_run"], bool(stochastic))  # override_defaults switches it on; set it
    fill(f.climate_configs["use_seed"], True)
    if stochastic:
        fill(f.climate_configs["seed"], seed + np.arange(len(configs)))

    base = {s: float(f.species_configs["baseline_concentration"].sel(specie=s).isel(config=0)) for s in GHG}
    n_tb = n_years + 1  # concentrations live on timebounds
    for name, d in zip(names, scens):
        for s, path in (("CO2", d["co2"]), ("CH4", d["ch4"]), ("N2O", np.zeros(n_tb))):
            conc = base[s] * np.exp(np.asarray(path)[:n_tb])
            fill(f.concentration, conc[:, None], scenario=name, specie=s)

    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.ocean_heat_content_change, 0)
    f.run()
    return f, names


def to_dataset(f, names, scens, args):
    def pick(da, **sel):
        return da.sel(**sel).transpose("scenario", "config", "timebounds").values

    dims = ("scenario", "config", "year")
    species = list(f.species)
    data = {
        "co2": (dims, pick(f.concentration, specie="CO2")),
        "ch4": (dims, pick(f.concentration, specie="CH4")),
        "n2o": (dims, pick(f.concentration, specie="N2O")),
        "tas": (dims, pick(f.temperature, layer=0)),
    }
    data.update({ERF_NAMES[s]: (dims, pick(f.forcing, specie=s)) for s in species})
    configs = np.asarray(f.configs)
    ds = xr.Dataset(
        data,
        coords={
            "scenario": names,
            "config": configs.astype(int) if args.climate_params is None else configs.astype(str),
            "year": np.asarray(f.timebounds).astype(int),
            "family": ("scenario", [d["family"] for d in scens]),
            "co2_shape": ("scenario", [d["co2_shape"] for d in scens]),
            "ch4_shape": ("scenario", [d["ch4_shape"] for d in scens]),
            "co2_ch4_corr": ("scenario", [d["corr"] for d in scens]),
            "ecs": ("config", f.ebms["ecs"].values),
            "tcr": ("config", f.ebms["tcr"].values),
        },
    )
    ds["erf_total"] = sum(ds[ERF_NAMES[s]] for s in species)
    units = dict(
        co2="ppm",
        ch4="ppb",
        n2o="ppb",
        erf_total="W m-2",
        ecs="K",
        tcr="K",
        tas="K, surface layer, relative to pre-industrial equilibrium",
    )
    units.update({ERF_NAMES[s]: "W m-2" for s in species})
    for v, u in units.items():
        ds[v].attrs["units"] = u
    if "Ozone" in species:
        ds["erf_ozone"].attrs["note"] = "from CH4 and N2O changes, plus FaIR's ozone temperature feedback"
    ds.attrs.update(
        description="FaIR concentration-driven runs with decoupled CO2 and CH4 (see generate_fair_scenarios.py)",
        climate=(
            "fair-calibrate v1.4.1 calibrated constrained ensemble"
            if args.climate_params is None
            else f"fitted climate from: {'; '.join(str(p) for p in args.climate_params)}"
        ),
        ch4_indirect_forcing=int("Ozone" in species),
        stochastic=int(args.stochastic),
        seed=args.seed,
        max_abs_corr=args.max_abs_corr,
        max_abs_corr_anti=args.max_abs_corr_anti,
        min_abs_corr_anti=args.min_abs_corr_anti,
        co2_multiplier_range=f"{args.co2_min}-{args.co2_max}",
        ch4_multiplier_range=f"{args.ch4_min}-{args.ch4_max}",
    )
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-family", type=int, default=20, help="scenarios per family (4 families)")
    ap.add_argument("--years", type=int, default=200)
    ap.add_argument("--n-configs", type=int, default=50, help="calibrated members to use (random subset; 0 = all 841)")
    ap.add_argument("--co2-min", type=float, default=0.5, help="lowest CO2 multiplier of baseline")
    ap.add_argument("--co2-max", type=float, default=4.0, help="highest CO2 multiplier of baseline")
    ap.add_argument("--ch4-min", type=float, default=0.5)
    ap.add_argument("--ch4-max", type=float, default=4.0)
    ap.add_argument("--max-abs-corr", type=float, default=0.5, help="reject 'mixed' pairs more correlated than this")
    ap.add_argument(
        "--max-abs-corr-anti",
        type=float,
        default=0.9,
        help="reject 'anticorrelated' pairs more negatively correlated than this",
    )
    ap.add_argument(
        "--min-abs-corr-anti",
        type=float,
        default=0.3,
        help="reject 'anticorrelated' pairs less negatively correlated than this (not clearly opposite)",
    )
    ap.add_argument("--stochastic", action="store_true", help="add FaIR internal variability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--climate-params",
        type=Path,
        nargs="+",
        default=None,
        help="params.csv files from fit_fair_to_cmip_scenarios.py, one config each " "(instead of calibrated members)",
    )
    ap.add_argument(
        "--no-ch4-indirect",
        action="store_true",
        help="leave out CH4's indirect forcing (ozone, stratospheric water vapour)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: data/fair/scenarios/<YYYYmmdd-HHMMSS>_fair_co2_ch4.nc; never overwrites",
    )
    args = ap.parse_args()
    if args.out is None:
        args.out = REPO / "data" / "fair" / "scenarios" / f"{datetime.now():%Y%m%d-%H%M%S}_fair_co2_ch4.nc"
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; pick another --out (outputs are never overwritten)")

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(PARAMS_FILE, index_col=0)
    if 0 < args.n_configs < len(df):
        df = df.iloc[np.sort(rng.choice(len(df), args.n_configs, replace=False))]

    bounds = {"CO2": (np.log(args.co2_min), np.log(args.co2_max)), "CH4": (np.log(args.ch4_min), np.log(args.ch4_max))}
    scens = draw_scenarios(
        args.n_per_family,
        args.years + 1,
        rng,
        bounds,
        args.max_abs_corr,
        args.max_abs_corr_anti,
        args.min_abs_corr_anti,
    )
    climate = None if args.climate_params is None else load_climate_params(args.climate_params)
    f, names = run_fair(
        scens, args.years, df, args.stochastic, args.seed, climate=climate, ch4_indirect=not args.no_ch4_indirect
    )
    ds = to_dataset(f, names, scens, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out)
    n_s, n_c, n_y = ds.sizes["scenario"], ds.sizes["config"], ds.sizes["year"]
    print(f"wrote {args.out}  ({n_s} scenarios x {n_c} configs x {n_y} years)")
    summary = ds.co2_ch4_corr.to_series().groupby(ds.family.to_series()).describe()[["count", "min", "mean", "max"]]
    print("\nCO2-CH4 correlation by family:\n", summary.round(2).to_string())


if __name__ == "__main__":
    main()
