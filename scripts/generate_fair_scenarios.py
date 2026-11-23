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

Emission-driven mode (``--drivers emissions``, for data aligned with NorESM2-LM's ClimateSet inputs)
- paths are yearly emission rates of CO2 (Gt CO2/yr) and CH4 (Mt CH4/yr) that start at zero (pre-industrial)
  and are scaled so their cumulative totals span a random 20-100 % of ``--co2-cum-max`` / ``--ch4-cum-max``
  (defaults: 1.2 x NorESM2-LM's largest 2100 value in the ClimateSet input4mips files, FAIR_STATE.md);
- FaIR computes the concentrations: CO2 carbon cycle from the calibrated medians (or per member), CH4 from
  FaIR's own CH4-only gas cycle (AR6 defaults, ch4_method leach2021). Checked against CMIP6 history: 2014 CO2
  391 vs 398 ppm, CH4 1824 vs 1832 ppb. N2O stays at its baseline concentration;
- the families are the same, but correlation is measured between what drives each gas's warming: cumulative
  CO2 emissions and CH4 emissions smoothed over CH4's ~10-year lifetime (a stand-in for its concentration);
- ``--monthly`` runs FaIR at a monthly step and writes monthly means; the output also holds cumulative
  emissions in kg, the quantity in NorESM's inputs, and CH4 emissions summed with older ones fading out over
  ~10 years (``ch4_faded_emissions``), which follows CH4's warming much more closely. Internal variability at a
  monthly step needs sigmas fitted
  at that step (``--noise-params``, from fit_fair_noise.py);
- ``co2mass`` [kg] is the CO2 concentration as total mass in the air (NorESM's CMIP6 output variable of that
  name, an input of the hierarchical model on origin/sz_forcing_dev); ``ch4`` [ppb] is already in the units of
  its ``ch4global``.

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

import fair
import numpy as np
import pandas as pd
import xarray as xr
from fair import FAIR
from fair.earth_params import mass_atmosphere, molecular_weight_air
from fair.interface import fill, initialise
from fair.io import read_properties
from scipy.signal import lfilter

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

# emission-driven mode
AR6_SPECIES_FILE = Path(fair.__file__).parent / "defaults" / "data" / "ar6" / "species_configs_properties.csv"
KG_PER_UNIT = {"CO2": 1e12, "CH4": 1e9}  # FaIR emission units Gt CO2/yr, Mt CH4/yr -> kg
# CH4's ~decade lifetime: for the correlation stand-in of its concentration, and for ch4_faded_emissions (emissions
# summed with older ones fading out), which tracks CH4-only warming far better than the plain cumulative sum
# (R^2 0.99 vs 0.62, FAIR_STATE.md)
CH4_LIFETIME_YEARS = 10.0
EMIS_UP = ("ramp_up", "step_up")
EMIS_FALL = ("peak",)  # rises, then falls back towards (CO2: possibly below) zero
EMIS_ALL = EMIS_UP + EMIS_FALL + ("random_walk",)
# NorESM2-LM, ClimateSet input4mips, largest cumulative total since 1850 by 2100 (ssp585 / ssp370), in kg
NORESM_CUM_MAX_KG = {"CO2": 9.24e15, "CH4": 6.86e13}


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
# emission-driven mode: yearly emission rates on FaIR's timepoints
# ---------------------------------------------------------------------------------------------
def make_emission_path(shape, n, rng, neg_frac=0.0):
    """Emission-rate shape with peak 1 that starts at 0 (pre-industrial); dips no lower than -neg_frac."""
    t = np.arange(n)
    start = int(rng.uniform(0.05, 0.3) * n)  # quiet period before anything happens
    end = min(start + int(rng.uniform(0.15, 0.5) * n), n - 1)
    ramp = np.clip((t - start) / max(end - start, 1), 0, 1)
    if shape == "ramp_up":
        return ramp
    if shape == "step_up":
        return (t >= start).astype(float)
    if shape == "peak":
        settle = rng.uniform(-neg_frac, 0.5)
        fall = np.clip((t - end) / max(int(rng.uniform(0.1, 0.3) * n), 1), 0, 1)
        return ramp - (1 - settle) * fall
    if shape == "random_walk":
        x = np.cumsum(rng.normal(size=n))
        k = max(n // 20, 3)
        x = np.convolve(x, np.ones(k) / k, mode="same")
        x = np.abs(x - x[start])  # folded at zero: clipping instead left mostly-zero paths with a spike at the end
        x[:start] = 0.0
        return x / (np.max(x) or 1.0)
    raise ValueError(f"unknown shape {shape!r}")


def scale_emissions(unit, step, rng, cum_max, rate_max):
    """Scale a shape so its largest cumulative total is 20-100 % of cum_max, rates within rate_max; None if the
    cumulative total ever goes below zero (more removed than was ever emitted)."""
    peak = np.max(np.cumsum(unit)) * step
    if peak <= 0:
        return None
    path = unit * min(rng.uniform(0.2, 1.0) * cum_max / peak, rate_max / np.max(np.abs(unit)))
    return None if np.min(np.cumsum(path)) < 0 else path


def warming_driver(rate, gas, step):
    """What a gas's warming follows: cumulative emissions for CO2, emissions smoothed over CH4's lifetime for CH4."""
    if gas == "CO2":
        return np.cumsum(rate) * step
    a = np.exp(-step / CH4_LIFETIME_YEARS)
    return lfilter([1 - a], [1, -a], rate)


def draw_emission_scenarios(n_per_family, n, step, rng, limits, max_abs_corr, max_abs_corr_anti, min_abs_corr_anti):
    """Like draw_scenarios, for emission rates; correlations are between the two warming drivers."""
    zero = np.zeros(n)

    def path(gas, shape):
        while True:
            unit = make_emission_path(shape, n, rng, limits[gas]["neg_frac"])
            p = scale_emissions(unit, step, rng, limits[gas]["cum_max"], limits[gas]["rate_max"])
            if p is not None:
                return p

    def r_of(c, m):
        return corr(warming_driver(c, "CO2", step), warming_driver(m, "CH4", step))

    tries = {"anticorrelated": 0, "mixed": 0}

    def tried(family):
        tries[family] += 1
        if tries[family] > 1000 * n_per_family:
            raise SystemExit(f"could not draw enough '{family}' pairs within the correlation limits; loosen them")

    out = []
    for _ in range(n_per_family):
        s = rng.choice(EMIS_ALL)
        out.append(dict(family="co2_only", co2_shape=s, ch4_shape="zero", co2=path("CO2", s), ch4=zero))
    for _ in range(n_per_family):
        s = rng.choice(EMIS_ALL)
        out.append(dict(family="ch4_only", co2_shape="zero", ch4_shape=s, co2=zero, ch4=path("CH4", s)))
    drawn = 0
    # CO2's warming rises while CH4's falls. Not the reverse: CO2's warming only falls with net removal, and with
    # every path starting at pre-industrial and cumulative CO2 kept >= 0 not one of 400 draws per shape came out
    # anticorrelated, even with removals as large as the peak emissions (FAIR_STATE.md).
    while drawn < n_per_family:
        tried("anticorrelated")
        sc, sm = rng.choice(EMIS_UP), "peak"
        c, m = path("CO2", sc), path("CH4", sm)
        if -max_abs_corr_anti <= r_of(c, m) <= -min_abs_corr_anti:
            out.append(dict(family="anticorrelated", co2_shape=sc, ch4_shape=sm, co2=c, ch4=m))
            drawn += 1
    drawn = 0
    while drawn < n_per_family:
        tried("mixed")
        sc, sm = rng.choice(EMIS_ALL), rng.choice(EMIS_ALL)
        c, m = path("CO2", sc), path("CH4", sm)
        r = r_of(c, m)
        if not np.isnan(r) and abs(r) <= max_abs_corr:
            out.append(dict(family="mixed", co2_shape=sc, ch4_shape=sm, co2=c, ch4=m))
            drawn += 1
    for d in out:
        d["corr"] = r_of(d["co2"], d["ch4"])
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


def apply_gas_cycles(f, fitted_climate):
    """
    Gas cycles for emission-driven runs.

    CO2: calibrated median carbon cycle for a fitted climate (the species file leaves iirf_* blank; with the
    calibrated ensemble override_defaults has already set it per member). CH4: FaIR's CH4-only gas cycle (AR6
    defaults, ch4_method leach2021) — calibration 1.4.1's chemistry-based lifetime needs NOx, VOCs etc. and
    overshoots without them (2014: 2617 vs 1832 ppb from historical emissions; FAIR_STATE.md).
    """
    if fitted_climate:
        med = pd.read_csv(PARAMS_FILE, index_col=0).median()
        for k in ("iirf_0", "iirf_airborne", "iirf_uptake", "iirf_temperature", "baseline_concentration"):
            fill(f.species_configs[k], med[f"{k}[CO2]"], specie="CO2")
    ar6 = pd.read_csv(AR6_SPECIES_FILE, index_col=0).loc["CH4"]
    for k in ("iirf_0", "iirf_airborne", "iirf_uptake", "iirf_temperature", "baseline_emissions"):
        fill(f.species_configs[k], ar6[k], specie="CH4")
    fill(f.species_configs["unperturbed_lifetime"], ar6["unperturbed_lifetime0"], specie="CH4")
    f.calculate_concentration_per_emission()


def run_fair(
    scens,
    n_years,
    df_configs,
    stochastic,
    seed,
    climate=None,
    ch4_indirect=True,
    step=1,
    noise=None,
    drivers="concentration",
):
    """
    Run FaIR for the given paths: log multipliers on the timebounds (drivers="concentration") or CO2/CH4
    emission rates on the timepoints (drivers="emissions", N2O then held at its baseline concentration).

    step is the time step in years. FaIR's noise depends on it: sigma_eta/sigma_xi set for one step give a
    different variability at another, so ``noise`` (sigma_eta, sigma_xi per config, from fit_fair_noise.py)
    must have been fitted at this step. to_dataset() still assumes step=1.
    """
    emitted = {"CO2", "CH4"} if drivers == "emissions" else set()
    f = FAIR(ch4_method="leach2021")
    f.define_time(0, n_years, step)
    names = [f"s{i:04d}" for i in range(len(scens))]
    f.define_scenarios(names)
    configs = df_configs.index if climate is None else climate.index
    f.define_configs(configs)
    _, props = read_properties(filename=SPECIES_FILE)
    species = GHG + (CH4_INDIRECT if ch4_indirect else [])
    modes = {s: ("emissions" if s in emitted else "concentration") for s in GHG}
    f.define_species(species, {s: {**props[s], **({"input_mode": modes[s]} if s in GHG else {})} for s in species})
    f.allocate()
    f.fill_species_configs(SPECIES_FILE)
    if climate is None:
        f.override_defaults(PARAMS_FILE)  # calibrated energy balance + forcing parameters, per member
    else:
        apply_climate_params(f, climate, species)
    if emitted:
        apply_gas_cycles(f, fitted_climate=climate is not None)
    if noise is not None:
        for name in configs:
            for k in ("sigma_eta", "sigma_xi"):
                fill(f.climate_configs[k], noise.loc[name, k], config=name)
    fill(f.climate_configs["stochastic_run"], bool(stochastic))  # override_defaults switches it on; set it
    fill(f.climate_configs["use_seed"], True)
    if stochastic:
        fill(f.climate_configs["seed"], seed + np.arange(len(configs)))

    base = {s: float(f.species_configs["baseline_concentration"].sel(specie=s).isel(config=0)) for s in GHG}
    n_tb = int(round(n_years / step)) + 1  # concentrations live on timebounds, emissions on the n_tb - 1 timepoints
    for name, d in zip(names, scens):
        for s, path in (("CO2", d["co2"]), ("CH4", d["ch4"]), ("N2O", np.zeros(n_tb))):
            if s in emitted:
                fill(f.emissions, np.asarray(path)[: n_tb - 1, None], scenario=name, specie=s)
            else:
                conc = base[s] * np.exp(np.asarray(path)[:n_tb])
                fill(f.concentration, conc[:, None], scenario=name, specie=s)
    if emitted:  # emission-driven gases start at their baseline concentration (per config)
        for s in emitted:
            initialise(f.concentration, f.species_configs["baseline_concentration"].sel(specie=s), specie=s)

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


def to_dataset_emissions(f, names, scens, args, step):
    """Emission-driven output: one value per FaIR step (mean of its two timebounds), monthly if step = 1/12."""

    def mean_step(da, **sel):
        v = da.sel(**sel).transpose("scenario", "config", "timebounds").values
        return 0.5 * (v[..., 1:] + v[..., :-1])

    def per_step(da, **sel):
        return da.sel(**sel).transpose("scenario", "config", "timepoints").values

    dims = ("scenario", "config", "time")
    species = list(f.species)
    data = {
        "co2": (dims, mean_step(f.concentration, specie="CO2")),
        "ch4": (dims, mean_step(f.concentration, specie="CH4")),
        "gmst": (dims, mean_step(f.temperature, layer=0)),
        "co2_emissions": (dims, per_step(f.emissions, specie="CO2")),
        "ch4_emissions": (dims, per_step(f.emissions, specie="CH4")),
    }
    for s in ("CO2", "CH4"):  # through the end of each step, as in NorESM's inputs summed since 1850
        data[f"{s.lower()}_cumulative_emissions"] = (
            dims,
            np.cumsum(data[f"{s.lower()}_emissions"][1], -1) * step * KG_PER_UNIT[s],
        )
    fade = np.exp(-step / CH4_LIFETIME_YEARS)  # each step, older CH4 emissions count for this much less
    data["ch4_faded_emissions"] = (
        dims,
        lfilter([step], [1, -fade], data["ch4_emissions"][1], axis=-1) * KG_PER_UNIT["CH4"],
    )
    # total CO2 mass in the air, as NorESM's CMIP6 output variable co2mass (Shan's model input): ppm -> kg with
    # FaIR's own atmosphere mass and molecular weights, the same mapping FaIR uses from emissions to ppm
    mw_co2 = np.unique(f.species_configs["molecular_weight"].sel(specie="CO2").values)
    assert mw_co2.size == 1, f"CO2 molecular weight differs between configs: {mw_co2}"
    kg_per_ppm_co2 = mass_atmosphere * 1e-6 * mw_co2[0] / molecular_weight_air
    data["co2mass"] = (dims, data["co2"][1] * kg_per_ppm_co2)
    data.update({ERF_NAMES[s]: (dims, mean_step(f.forcing, specie=s)) for s in species})
    configs = np.asarray(f.configs)
    tp = np.asarray(f.timepoints)
    ds = xr.Dataset(
        data,
        coords={
            "scenario": names,
            "config": configs.astype(int) if args.climate_params is None else configs.astype(str),
            # plain "years" on purpose: a "years since ..." unit makes xarray try to decode calendar dates and fail
            "time": (
                "time",
                tp,
                {"units": "years", "description": "since the pre-industrial start; middle of each step"},
            ),
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
        gmst="K, global-mean surface temperature anomaly vs pre-industrial equilibrium, no seasonal cycle",
        co2_emissions="Gt CO2 / yr",
        ch4_emissions="Mt CH4 / yr",
        co2_cumulative_emissions="kg CO2",  # never "... since ...": xarray would decode it as a calendar date
        ch4_cumulative_emissions="kg CH4",
        ch4_faded_emissions="kg CH4",
        co2mass="kg CO2",
        erf_total="W m-2",
        ecs="K",
        tcr="K",
    )
    units.update({ERF_NAMES[s]: "W m-2" for s in species})
    for v, u in units.items():
        ds[v].attrs["units"] = u
    ds["co2_ch4_corr"].attrs["note"] = "corr(cumulative CO2 emissions, CH4 emissions smoothed over ~10 y)"
    for s in ("co2", "ch4"):
        ds[f"{s}_cumulative_emissions"].attrs[
            "description"
        ] = "summed from the pre-industrial start to the end of each step"
    ds["ch4_faded_emissions"].attrs["description"] = (
        f"CH4 emissions summed with older ones fading out over {CH4_LIFETIME_YEARS:g} years (e-folding), "
        "roughly the CH4 still in the air; tracks CH4-driven warming better than the plain cumulative sum"
    )
    ds["co2mass"].attrs["description"] = (
        f"total atmospheric mass of CO2 (as CMIP6 co2mass) = co2 [ppm] x {kg_per_ppm_co2:.4e} kg/ppm "
        "(FaIR's atmosphere mass and molecular weights)"
    )
    ds["co2mass"].attrs["kg_per_ppm"] = kg_per_ppm_co2
    ds.attrs.update(
        description="FaIR emission-driven runs with decoupled CO2 and CH4 (see generate_fair_scenarios.py)",
        climate=(
            "fair-calibrate v1.4.1 calibrated constrained ensemble"
            if args.climate_params is None
            else f"fitted climate from: {'; '.join(str(p) for p in args.climate_params)}"
        ),
        timestep_years=step,
        ch4_fade_years=CH4_LIFETIME_YEARS,
        ch4_indirect_forcing=int("Ozone" in species),
        stochastic=int(args.stochastic),
        noise_params=str(args.noise_params or ""),
        seed=args.seed,
        max_abs_corr=args.max_abs_corr,
        max_abs_corr_anti=args.max_abs_corr_anti,
        min_abs_corr_anti=args.min_abs_corr_anti,
        co2_cum_max_kg=args.co2_cum_max,
        ch4_cum_max_kg=args.ch4_cum_max,
        co2_rate_max=args.co2_rate_max,
        ch4_rate_max=args.ch4_rate_max,
        co2_neg_frac=args.co2_neg_frac,
    )
    return ds


def load_noise_params(path, configs, step):
    """sigma_eta/sigma_xi per climate from fit_fair_noise.py; refuses a file fitted at another time step."""
    noise = pd.read_csv(path, index_col=0)
    if not np.allclose(noise["timestep_years"], step):
        raise SystemExit(f"{path} was fitted at a {noise['timestep_years'].iloc[0]:.4f}-year step, not {step:.4f}")
    missing = [c for c in configs if c not in noise.index]
    if missing:
        raise SystemExit(f"{path} has no noise fit for {missing}")
    return noise.loc[list(configs), ["sigma_eta", "sigma_xi"]]


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
    ap.add_argument("--drivers", choices=("concentration", "emissions"), default="concentration")
    ap.add_argument("--monthly", action="store_true", help="emission mode: monthly FaIR step and monthly output")
    ap.add_argument(
        "--co2-cum-max",
        type=float,
        default=1.2 * NORESM_CUM_MAX_KG["CO2"],
        help="emission mode: largest cumulative CO2 [kg] (default 1.2 x NorESM2-LM's 2100 maximum)",
    )
    ap.add_argument(
        "--ch4-cum-max",
        type=float,
        default=1.2 * NORESM_CUM_MAX_KG["CH4"],
        help="emission mode: largest cumulative CH4 [kg] (default 1.2 x NorESM2-LM's 2100 maximum)",
    )
    ap.add_argument("--co2-rate-max", type=float, default=150.0, help="emission mode: Gt CO2/yr (NorESM max 128)")
    ap.add_argument(
        "--ch4-rate-max",
        type=float,
        default=700.0,
        help="emission mode: Mt CH4/yr (NorESM max 696; 1000 gave CH4 up to 5.8x pre-industrial)",
    )
    ap.add_argument(
        "--co2-neg-frac",
        type=float,
        default=0.2,
        help="emission mode: CO2 removal down to this fraction of the path's peak rate",
    )
    ap.add_argument(
        "--noise-params",
        type=Path,
        default=None,
        help="noise_params.csv from fit_fair_noise.py (needed for --stochastic with --monthly)",
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
    emission_mode = args.drivers == "emissions"
    if args.monthly and not emission_mode:
        raise SystemExit("--monthly is only implemented for --drivers emissions")
    step = 1 / 12 if args.monthly else 1
    if args.stochastic and args.monthly and args.noise_params is None:
        raise SystemExit(
            "--stochastic --monthly needs --noise-params: FaIR's sigmas only hold at the step they were set for"
        )
    if args.out is None:
        tag = "_emissions" + ("_monthly" if args.monthly else "") if emission_mode else ""
        args.out = REPO / "data" / "fair" / "scenarios" / f"{datetime.now():%Y%m%d-%H%M%S}_fair_co2_ch4{tag}.nc"
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; pick another --out (outputs are never overwritten)")

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(PARAMS_FILE, index_col=0)
    if 0 < args.n_configs < len(df):
        df = df.iloc[np.sort(rng.choice(len(df), args.n_configs, replace=False))]

    if emission_mode:
        limits = {
            "CO2": dict(
                cum_max=args.co2_cum_max / KG_PER_UNIT["CO2"], rate_max=args.co2_rate_max, neg_frac=args.co2_neg_frac
            ),
            "CH4": dict(cum_max=args.ch4_cum_max / KG_PER_UNIT["CH4"], rate_max=args.ch4_rate_max, neg_frac=0.0),
        }
        scens = draw_emission_scenarios(
            args.n_per_family,
            int(round(args.years / step)),
            step,
            rng,
            limits,
            args.max_abs_corr,
            args.max_abs_corr_anti,
            args.min_abs_corr_anti,
        )
        climate = None if args.climate_params is None else load_climate_params(args.climate_params)
        configs = df.index if climate is None else climate.index
        noise = None if args.noise_params is None else load_noise_params(args.noise_params, configs, step)
        f, names = run_fair(
            scens,
            args.years,
            df,
            args.stochastic,
            args.seed,
            climate=climate,
            ch4_indirect=not args.no_ch4_indirect,
            step=step,
            noise=noise,
            drivers="emissions",
        )
        ds = to_dataset_emissions(f, names, scens, args, step)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(args.out)
        xr.open_dataset(args.out).close()  # fails here, not at the user's end, if a unit trips xarray's decoding
        n_s, n_c, n_t = ds.sizes["scenario"], ds.sizes["config"], ds.sizes["time"]
        print(f"wrote {args.out}  ({n_s} scenarios x {n_c} configs x {n_t} steps of {step:.4f} y)")
        summary = ds.co2_ch4_corr.to_series().groupby(ds.family.to_series()).describe()[["count", "min", "mean", "max"]]
        print("\nCO2-CH4 warming-driver correlation by family:\n", summary.round(2).to_string())
        return

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
