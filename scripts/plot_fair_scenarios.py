"""
Plot example scenarios from an emission-driven FaIR file: temperature on the left, the CO2/CH4 inputs on the right.

Laid out like Shan's toy_distange_FaIR notebook: one row per scenario, everything normalised (mean 0, sd 1 over
the whole file, per variable), x = step. By default one scenario per family. ``--inputs`` picks which form of
CO2/CH4 is shown: cumulative emissions (what NorESM's ClimateSet inputs hold), yearly emissions, concentrations
or effective radiative forcing.

``--noresm`` plots the model's own data the same way instead (NorESM2-LM by default): one row per SSP, each the
historical run 1850-2014 joined to the SSP 2015-2100, member r1i1p1f1. Temperature = monthly global mean tas
minus that calendar month's 1850-1900 mean (removes the seasonal cycle); CO2/CH4 = ClimateSet input4mips emissions
(CH4 no-fires), as monthly global totals
and their sums since 1850 (fluxes taken as kg m-2 s-1; the files carry no units). Only emissions exist there,
so ``--inputs`` must be cumulative or emissions.

Usage
-----
    python scripts/plot_fair_scenarios.py data/fair/scenarios/<file>.nc [--inputs forcing] [--config 0]
    python scripts/plot_fair_scenarios.py --noresm [--inputs emissions]

Output: <file>_examples_<inputs>.png next to the data file, or
data/fair/cmip6/<model>/observed/<YYYYmmdd-HHMMSS>_examples_<inputs>.png for --noresm.
"""

import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_fair_noise import monthly_global_means  # noqa: E402
from fit_fair_to_cmip_4xco2 import C_ALT, C_FIT, git_state, stamp, style  # noqa: E402
from generate_fair_scenarios import CH4_LIFETIME_YEARS  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CLIMATESET = REPO / "Climateset_DATA"
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
SECONDS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 86400  # 365_day calendar

INPUTS = {
    "cumulative": ("co2_cumulative_emissions", "ch4_cumulative_emissions", "cumulative emissions"),
    "emissions": ("co2_emissions", "ch4_emissions", "emissions"),
    "concentration": ("co2", "ch4", "concentration"),
    "forcing": ("erf_co2", "erf_ch4", "forcing"),
}


def normalise(da):
    return (da - da.mean()) / da.std()


def monthly_emissions(scen, sp, years):
    """Global monthly totals in kg per month from ClimateSet input4mips fluxes (assumed kg m-2 s-1)."""
    tag, area, out = ("no-fires" if sp == "CH4" else ""), None, []
    for yr in years:
        (fp,) = glob.glob(str(CLIMATESET / f"inputs/input4mips/{scen}/{sp}_sum/250_km/mon/{yr}/*{tag}*.nc"))
        with xr.open_dataset(fp, decode_times=False) as ds:
            if area is None:
                lat, lon = ds["lat"].values, ds["lon"].values
                cell = np.deg2rad(abs(lat[1] - lat[0])) * np.deg2rad(abs(lon[1] - lon[0]))
                area = (6.371e6**2 * cell * np.cos(np.deg2rad(lat)))[:, None] * np.ones(len(lon))
            out.append(np.sum(ds[sp].values * area[None], axis=(1, 2)) * SECONDS)
    return np.concatenate(out)


def noresm_dataset(model):
    """The model's historical+SSP data in the variable names of an emission-driven FaIR file (config dim dropped)."""
    hist_years, ssp_years = range(1850, 2015), range(2015, 2101)
    tas_dir = CLIMATESET / "outputs" / "CMIP6" / model / "r1i1p1f1" / "{s}" / "tas" / "250_km" / "mon"
    hist_t = monthly_global_means(Path(str(tas_dir).format(s="historical")))
    # per calendar month: removes the seasonal cycle (several K in the global mean; FaIR has none)
    base = np.tile(np.mean([hist_t[y] for y in range(1850, 1901)], axis=0), len(hist_years) + len(ssp_years))
    hist_e = {sp: monthly_emissions("historical", sp, hist_years) for sp in ("CO2", "CH4")}
    rows = {}
    for scen in SSPS:
        ssp_t = monthly_global_means(Path(str(tas_dir).format(s=scen)))
        t = np.concatenate([hist_t[y] for y in hist_years] + [ssp_t[y] for y in ssp_years]) - base
        e = {sp: np.concatenate([hist_e[sp], monthly_emissions(scen, sp, ssp_years)]) for sp in ("CO2", "CH4")}
        fade = np.exp(-(1 / 12) / CH4_LIFETIME_YEARS)  # as ch4_faded_emissions in the FaIR files
        rows[scen] = dict(
            gmst=t,
            co2_emissions=e["CO2"],
            ch4_emissions=e["CH4"],
            co2_cumulative_emissions=np.cumsum(e["CO2"]),
            ch4_cumulative_emissions=np.cumsum(e["CH4"]),
            ch4_faded_emissions=lfilter([1], [1, -fade], e["CH4"]),
        )
    dims = ("scenario", "time")
    data = {v: (dims, np.stack([rows[s][v] for s in SSPS])) for v in rows[SSPS[0]]}
    return xr.Dataset(data, coords={"scenario": SSPS, "family": ("scenario", SSPS), "config": model})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path, nargs="?", default=None)
    ap.add_argument(
        "--noresm",
        nargs="?",
        const="NorESM2-LM",
        default=None,
        metavar="MODEL",
        help="plot the model's own historical+SSP data instead of a FaIR file",
    )
    ap.add_argument("--inputs", choices=INPUTS, default="cumulative")
    ap.add_argument("--config", type=int, default=0, help="which climate (index)")
    ap.add_argument("--scenarios", nargs="+", default=None, help="scenario names; default: first of each family")
    ap.add_argument(
        "--per-family", type=int, default=1, help="first N scenarios of each family; N > 1 writes one figure per family"
    )
    ap.add_argument(
        "--overlay",
        action="store_true",
        help="temperature, CO2 and CH4 in one panel per scenario (all normalised) instead of two panels",
    )
    args = ap.parse_args()
    if (args.file is None) == (args.noresm is None):
        raise SystemExit("give either a FaIR file or --noresm")
    if args.noresm and args.inputs not in ("cumulative", "emissions"):
        raise SystemExit("--noresm only has emissions (ClimateSet inputs): use --inputs cumulative or emissions")

    if args.noresm:
        ds = noresm_dataset(args.noresm)
        source = f"{args.noresm} r1i1p1f1 historical+SSP, ClimateSet"
        out_dir = REPO / "data" / "fair" / "cmip6" / args.noresm / "observed"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}_examples_{args.inputs}.png"
    else:
        ds = xr.open_dataset(args.file).isel(config=args.config).load()
        source = args.file.name
        out = args.file.with_name(f"{args.file.stem}_examples_{args.inputs}.png")
    if args.overlay:
        out = out.with_name(f"{out.stem}_overlay{out.suffix}")
    co2_var, ch4_var, what = INPUTS[args.inputs]
    # with cumulative inputs, also show CH4 with older emissions fading out (follows CH4's warming far better)
    extra = "ch4_faded_emissions" if args.inputs == "cumulative" and "ch4_faded_emissions" in ds else None
    norm = {v: normalise(ds[v]) for v in ("gmst", co2_var, ch4_var) + ((extra,) if extra else ())}
    fam_of = [str(f) for f in ds.family.values]
    if args.scenarios is not None:
        groups = {"": args.scenarios}
    else:
        groups = {
            fam: [str(s) for s, f in zip(ds.scenario.values, fam_of) if f == fam][: args.per_family]
            for fam in dict.fromkeys(fam_of)
        }
        if args.per_family == 1:
            groups = {"": [names[0] for names in groups.values()]}
    for fam, names in groups.items():
        path = out if not fam else out.with_name(f"{out.stem}_{fam}{out.suffix}")
        plot_rows(ds, names, norm, co2_var, ch4_var, what, args.noresm, source, path, args.overlay, extra)


def plot_rows(ds, names, norm, co2_var, ch4_var, what, noresm, source, out, overlay=False, extra=None):
    ncols = 1 if overlay else 2
    fig, axes = plt.subplots(len(names), ncols, figsize=(13 if ncols == 2 else 10, 3.2 * len(names)), squeeze=False)
    step = range(ds.sizes["time"])
    who = noresm or "FaIR"
    for row, name in zip(axes, names):
        ax_t, ax_f = (row[0], row[0]) if overlay else row
        fam = str(ds.family.sel(scenario=name).values)
        label = f"{who} {name}" + ("" if noresm else f" ({fam})")
        if noresm:
            temp_label = f"{who} temperature" if overlay else who
        else:  # FaIR's own temperature for an invented scenario: say which climate it stands in for
            tuned = "NorESM2-LM" in str(ds.attrs.get("climate", ""))
            temp_label = "FaIR temperature (FaIR tuned to NorESM2-LM)" if tuned else "FaIR temperature"
        ax_t.plot(step, norm["gmst"].sel(scenario=name), color="black", lw=1.0 if noresm else 1.5, label=temp_label)
        ax_f.plot(
            step, norm[co2_var].sel(scenario=name), color=C_FIT, lw=1.6, label=f"CO2 {what}" if overlay else "CO2"
        )
        ax_f.plot(
            step, norm[ch4_var].sel(scenario=name), color=C_ALT, lw=1.6, label=f"CH4 {what}" if overlay else "CH4"
        )
        if extra:
            ax_f.plot(
                step,
                norm[extra].sel(scenario=name),
                color=C_ALT,
                lw=1.6,
                ls="--",
                label=f"CH4 emissions fading over {CH4_LIFETIME_YEARS:g} y",
            )
        if overlay:
            ax_t.set(
                title=f"{label}: temperature and {what}", xlabel="step (month)", ylabel="normalised (mean 0, sd 1)"
            )
        else:
            ax_t.set(title=f"{label}: temperature", xlabel="step (month)", ylabel="normalised temperature")
            ax_f.set(title=f"{label}: {what}", xlabel="step (month)", ylabel=f"normalised {what}")
            ax_f.legend(frameon=False, fontsize=8)
            style(ax_f)
        ax_t.legend(frameon=False, fontsize=8)
        style(ax_t)
    stamp(fig, f"{source} · config {str(ds.config.values)} · scripts/{Path(__file__).name} @ {git_state()}")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
