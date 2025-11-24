import healpy as hp
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def remap_reg_to_healpix(
    arr,
    coords_lon,
    coords_lat,
    res_down=2,
):

    n_lon = len(coords_lon)
    n_lat = len(coords_lat)
    NSIDE = int(np.sqrt(n_lon * n_lat / (12 * res_down)))
    npix = hp.nside2npix(NSIDE)

    theta, phi = hp.pix2ang(NSIDE, np.arange(npix))
    lat_hp = np.pi / 2 - theta  # convert colatitude to latitude
    lon_hp = phi.copy()

    points = np.array([lat_hp, lon_hp]).T

    coords_lon_ext = np.hstack([coords_lon, coords_lon[-1] + (coords_lon[1] - coords_lon[0])])
    arr_ext = np.hstack([arr[0, 0], arr[0, 0][:, 0][:, None]])

    interp_func = RegularGridInterpolator(
        (np.radians(coords_lat), np.radians(coords_lon_ext)), arr_ext, bounds_error=False, fill_value=np.nan
    )

    healpix_map = interp_func(points)

    latitudes_new = -(np.degrees(theta) - 90)  # Between -90 and 90
    longitudes_new = np.degrees(lon_hp)  # Between 0 and 360

    return healpix_map, latitudes_new, longitudes_new
