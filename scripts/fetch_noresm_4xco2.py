"""Fetch NorESM2-LM abrupt-4xCO2 monthly fields from the public CMIP6 cloud mirrors and write them in the CMIP6
directory layout that climatem's CMIP6 loader expects
(…/CMIP6/CMIP/NCC/NorESM2-LM/abrupt-4xCO2/r1i1p1f1/Amon/<var>/<grid>/<version>/).

Why not ESGF: the ESGF index knows the datasets, but from Mila the DKRZ file links are 404 and the CEDA hosts time out
(checked 2026-11-23). The Pangeo (Google) and AWS mirrors serve the same CMOR output as zarr over plain HTTPS.

What is written
- ``ts`` and ``tas`` (Amon, gn, v20210118): 500 years monthly, 96 x 144, split into 10-year files named like CMOR
  output.
- ``co2mass`` and ``ch4global`` (Amon, gm): NOT published for this run on ESGF or the mirrors. In abrupt-4xCO2 both are
  constant (CO2 at 4 x pre-industrial, CH4 at pre-industrial), so they are constructed as constants from the NorESM2-LM
  historical Jan-1850 values and written with the same time axis. The attrs say so.
- ``NOTE_provenance.txt`` next to the data.

Sanity check: the area-weighted annual global-mean ``tas`` is compared with the fair-calibrate 500-year series
(``data/fair/cmip6/NorESM2-LM/NorESM2-LM_abrupt-4xCO2_r1i1p1f1_anomalies.txt``) via year-to-year changes, which are
independent of the piControl baseline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

MIRRORS = {
    "gcs": "https://storage.googleapis.com/cmip6/CMIP6",
    "aws": "https://cmip6-pds.s3.amazonaws.com/CMIP6",
}
MODEL, EXPERIMENT, MEMBER, TABLE = "NorESM2-LM", "abrupt-4xCO2", "r1i1p1f1", "Amon"
VERSION = "v20210118"
REL = f"CMIP/NCC/{MODEL}/{EXPERIMENT}/{MEMBER}/{TABLE}"
TIME_ENCODING = {"units": "days since 0001-01-01 00:00:00", "calendar": "noleap", "dtype": "float64"}
ANOMALIES = Path("data/fair/cmip6/NorESM2-LM/NorESM2-LM_abrupt-4xCO2_r1i1p1f1_anomalies.txt")


def git_state() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "scripts"], text=True).strip()
        return sha + ("+uncommitted" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def open_cloud(mirror: str, var: str) -> xr.Dataset:
    url = f"{MIRRORS[mirror]}/{REL}/{var}/gn/{VERSION}/"
    ds = xr.open_zarr(fsspec.get_mapper(url), consolidated=True, use_cftime=True)
    ds.attrs["cloud_source"] = url
    return ds


def year_of(t) -> int:
    return int(t.year)


def split_10y(ds: xr.Dataset):
    """Yield (label, subset) in 10-year blocks named like CMOR files: 000101-001012."""
    years = np.array([year_of(t) for t in ds.time.values])
    for y0 in range(years.min(), years.max() + 1, 10):
        y1 = min(y0 + 9, years.max())
        sel = (years >= y0) & (years <= y1)
        yield f"{y0:04d}01-{y1:04d}12", ds.isel(time=sel)


def write_cmor_like(ds: xr.Dataset, var: str, grid: str, out_root: Path) -> list[Path]:
    out_dir = out_root / REL / var / grid / VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for label, sub in split_10y(ds):
        f = out_dir / f"{var}_{TABLE}_{MODEL}_{EXPERIMENT}_{MEMBER}_{grid}_{label}.nc"
        enc = {"time": TIME_ENCODING, var: {"dtype": "float32", "zlib": True, "complevel": 4}}
        if "time_bnds" in sub:
            enc["time_bnds"] = {
                "units": TIME_ENCODING["units"],
                "calendar": TIME_ENCODING["calendar"],
                "dtype": "float64",
            }
        # engine given explicitly: the Mila env has no netCDF4, and xarray would fall back to the NetCDF3 scipy writer
        sub.to_netcdf(f, engine="h5netcdf", encoding=enc, unlimited_dims=["time"])
        written.append(f)
    return written


def constant_global(template: xr.Dataset, var: str, value: float, attrs: dict) -> xr.Dataset:
    ds = xr.Dataset(
        {var: ("time", np.full(template.sizes["time"], value, dtype="float32"), attrs)},
        coords={"time": template.time},
    )
    if "time_bnds" in template:
        ds["time_bnds"] = template["time_bnds"]
    ds.attrs.update({k: template.attrs[k] for k in template.attrs if k not in ("cloud_source",)})
    ds.attrs["grid_label"] = "gm"
    ds.attrs["variable_id"] = var
    return ds


def global_mean(da: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(da.lat))
    return da.weighted(w).mean(("lat", "lon"))


def check_against_anomalies(tas: xr.DataArray) -> str:
    if not ANOMALIES.exists():
        return f"no cross-check: {ANOMALIES} missing"
    ref = pd.read_csv(ANOMALIES, index_col=0)["tas"].values  # annual, 500 y, anomaly vs piControl
    gm = global_mean(tas)
    years = np.array([year_of(t) for t in gm.time.values])
    ann = np.array([gm.values[years == y].mean() for y in np.unique(years)])
    n = min(len(ann), len(ref))
    d_ours, d_ref = np.diff(ann[:n]), np.diff(ref[:n])
    r = np.corrcoef(d_ours, d_ref)[0, 1]
    off = ann[:n] - ref[:n]  # = piControl baseline if the series agree; should be nearly constant
    return (
        f"annual global-mean tas vs fair-calibrate series: {n} years; corr of year-to-year changes {r:.4f}; "
        f"max |diff of changes| {np.abs(d_ours - d_ref).max():.4f} K; implied piControl baseline "
        f"{off.mean():.3f} K (sd {off.std():.4f} K); year-1 mean {ann[0]:.3f} K, year-{n} mean {ann[n - 1]:.3f} K"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out-root", type=Path, required=True, help="…/CMIP6 root to write into (e.g. $SCRATCH/cmip6_extra/CMIP6)"
    )
    ap.add_argument("--mirror", choices=MIRRORS, default="gcs")
    ap.add_argument("--variables", nargs="+", default=["ts", "tas"])
    ap.add_argument(
        "--co2mass-pi", type=float, default=1.45335e15, help="kg, NorESM2-LM historical Jan 1850 (read 2026-11-23)"
    )
    ap.add_argument("--ch4-pi", type=float, default=808.24, help="ppb, NorESM2-LM historical Jan 1850")
    ap.add_argument("--no-constants", action="store_true", help="do not write the constant co2mass/ch4global files")
    args = ap.parse_args()

    out_root: Path = args.out_root
    lines = [
        f"written {dt.datetime.now():%Y-%m-%d %H:%M} by scripts/fetch_noresm_4xco2.py @ {git_state()}",
        f"source: {args.mirror} mirror of CMIP6 {MODEL} {EXPERIMENT} {MEMBER} {TABLE} {VERSION}",
        "Mila: lives in $SCRATCH (purged after 90 days unused); re-run this script to restore.",
    ]
    template = None
    for var in args.variables:
        ds = open_cloud(args.mirror, var)
        print(f"{var}: {dict(ds.sizes)} from {ds.attrs['cloud_source']}", flush=True)
        ds = ds.load()
        files = write_cmor_like(ds, var, "gn", out_root)
        size = sum(f.stat().st_size for f in files) / 1e6
        lines.append(
            f"{var}: {len(files)} files, {size:.1f} MB, {ds.sizes['time']} months, "
            f"grid {ds.sizes['lat']}x{ds.sizes['lon']}"
        )
        print(lines[-1], flush=True)
        if var == "tas":
            lines.append(check_against_anomalies(ds["tas"]))
            print(lines[-1], flush=True)
        if template is None:
            template = ds.drop_vars(
                [v for v in ds.data_vars if v != "time_bnds"] + ["lat", "lon", "lat_bnds", "lon_bnds"], errors="ignore"
            )

    if not args.no_constants and template is not None:
        co2 = constant_global(
            template,
            "co2mass",
            4.0 * args.co2mass_pi,
            {
                "standard_name": "atmosphere_mass_of_carbon_dioxide",
                "long_name": "Total Atmospheric Mass of CO2",
                "units": "kg",
                "note": f"CONSTRUCTED constant = 4 x {args.co2mass_pi:.5e} kg "
                "(NorESM2-LM historical Jan 1850 co2mass); "
                "not published for this run. NorESM2-MM abrupt-4xCO2 has 5.8222e15.",
            },
        )
        ch4 = constant_global(
            template,
            "ch4global",
            args.ch4_pi,
            {
                "standard_name": "mole_fraction_of_methane_in_air",
                "long_name": "Global Mean Mole Fraction of CH4",
                "units": "1e-09",
                "note": f"CONSTRUCTED constant = {args.ch4_pi} ppb (NorESM2-LM historical Jan 1850 ch4global); "
                "not published for this run. NorESM2-MM abrupt-4xCO2 has 808.25.",
            },
        )
        for var, ds in (("co2mass", co2), ("ch4global", ch4)):
            files = write_cmor_like(ds, var, "gm", out_root)
            lines.append(f"{var}: {len(files)} files, constant {float(ds[var][0]):.6g} {ds[var].attrs['units']}")
            print(lines[-1], flush=True)

    note = out_root / REL / "NOTE_provenance.txt"
    note.write_text("\n".join(lines) + "\n")
    print(f"wrote {note}")


if __name__ == "__main__":
    main()
