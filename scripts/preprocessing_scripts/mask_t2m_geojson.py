import os
import glob
import xarray as xr
import rioxarray
import geopandas as gpd
from tqdm import tqdm

# === CONFIG ===
INPUT_DIR = "/home/mila/l/lastc/scratch/data/ERA5_DATA_TEST/daily/t2m"
MASK_PATH = "/home/mila/l/lastc/scratch/data/ERA5_DATA_TEST/cascadia_bioregion.geojson"
EXT = "_masked.nc"

# Load GeoJSON mask and ensure it's in WGS84
mask = gpd.read_file(MASK_PATH).to_crs("EPSG:4326")

# Find all t2m daily files (excluding already masked ones)
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*/*.nc")))
files = [f for f in files if not f.endswith(EXT)]

print(f"Found {len(files)} files to process...")

for fpath in tqdm(files):
    try:
        ds = xr.open_dataset(fpath)

        # Setup for rioxarray clipping
        varname = list(ds.data_vars)[0]
        da = ds[varname]
        da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        da.rio.write_crs("EPSG:4326", inplace=True)

        clipped = da.rio.clip(mask.geometry, mask.crs, drop=True)
        out_path = fpath.replace(".nc", EXT)
        clipped.to_netcdf(out_path)

    except Exception as e:
        print(f"Failed to process {fpath}: {e}")
