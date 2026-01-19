import healpy as hp
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange


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

    latitudes_new = -(np.degrees(theta) - 90)  # Between -90 and 90
    longitudes_new = np.degrees(lon_hp)  # Between 0 and 360

    points = np.array([lat_hp, lon_hp]).T
    coords_lon_ext = np.hstack([coords_lon, coords_lon[-1] + (coords_lon[1] - coords_lon[0])])

    final_shape = (arr.shape[0], arr.shape[1], arr.shape[2], npix)

    arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
    remapped_data = np.zeros((arr.shape[0], npix))

    for k in trange(arr.shape[0]):
        arr_ext = arr[k]
        arr_ext = np.hstack([arr_ext, arr_ext[:, 0][:, None]])
        interp_func = RegularGridInterpolator(
            (np.radians(coords_lat), np.radians(coords_lon_ext)), arr_ext, bounds_error=False, fill_value=np.nan
        )
        # healpix_map = interp_func(points)
        remapped_data[k] = interp_func(points)

    remapped_data = remapped_data.reshape(final_shape)

    return remapped_data, latitudes_new, longitudes_new
