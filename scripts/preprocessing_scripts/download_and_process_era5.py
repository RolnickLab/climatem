import os
import argparse
import xarray as xr
import cdsapi
import numpy as np
from tqdm import tqdm

# Directories
base_dir = "/network/scratch/l/lastc/data/ERA5_DATA_PROCESSED"
os.makedirs(base_dir, exist_ok=True)

# CDS API client
client = cdsapi.Client()

def download_grib(variable, year, pressure_level=None, area=None, target_file=None):
    request = {
        "product_type": "reanalysis",
        "variable": variable,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00"],
        "format": "grib"
    }

    if pressure_level:
        request["pressure_level"] = pressure_level
        dataset = "reanalysis-era5-pressure-levels"
    else:
        dataset = "reanalysis-era5-single-levels"

    if area:
        request["area"] = area  # [North, West, South, East]

    print(f"Downloading {variable} for {year}...")
    client.retrieve(dataset, request, target_file)
    print(f"Saved to {target_file}")

def preprocess_t2m(grib_path, output_path):
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    t2m = ds["t2m"] - 273.15  # Convert to °C
    t2m.name = "t2m"

    # Resample to 5-day average
    t2m_5day = t2m.resample(time="5D").mean()

    # Save
    t2m_5day.to_netcdf(output_path)
    print(f"Processed t2m saved: {output_path}")

def preprocess_z500(grib_path, output_path):
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    z = ds["z"] / 9.80665  # Convert geopotential (m^2/s^2) to height (meters)
    z.name = "z"

    # Crop to 20N–80N, -180W to -30W
    z_cropped = z.sel(latitude=slice(80, 20), longitude=slice(-180, -30))

    # Save
    z_cropped.to_netcdf(output_path)
    print(f"Processed z500 saved: {output_path}")

def stack_yearly_files(varname, years, out_path):
    files = [os.path.join(base_dir, f"{varname}_{y}.nc") for y in years]
    existing = [f for f in files if os.path.exists(f)]

    if not existing:
        print(f"No {varname} files found to stack.")
        return

    print(f"Stacking {len(existing)} files for {varname}...")
    combined = xr.open_mfdataset(existing, combine="by_coords")
    combined.to_netcdf(out_path)
    print(f"Final combined {varname} saved: {out_path}")

def main(args):
    years = list(range(1950, 2025))

    if args.t2m:
        for y in tqdm(years, desc="t2m"):
            grib_path = os.path.join(base_dir, f"t2m_{y}.grib")
            nc_path = os.path.join(base_dir, f"t2m_{y}.nc")

            if not os.path.exists(nc_path):
                download_grib("2m_temperature", y, target_file=grib_path)
                preprocess_t2m(grib_path, nc_path)

        stack_yearly_files("t2m", years, os.path.join(base_dir, "t2m_1950_2024_combined.nc"))

    if args.z500:
        for y in tqdm(years, desc="z500"):
            grib_path = os.path.join(base_dir, f"z500_{y}.grib")
            nc_path = os.path.join(base_dir, f"z500_{y}.nc")

            if not os.path.exists(nc_path):
                download_grib("geopotential", y, pressure_level="500", area=[80, -180, 20, -30], target_file=grib_path)
                preprocess_z500(grib_path, nc_path)

        stack_yearly_files("z500", years, os.path.join(base_dir, "z500_1950_2024_combined.nc"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2m", action="store_true", help="Download and process 2m temperature")
    parser.add_argument("--z500", action="store_true", help="Download and process 500 hPa geopotential")
    args = parser.parse_args()

    main(args)
