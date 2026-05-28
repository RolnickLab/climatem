import healpy as hp
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, NearestNDInterpolator
from tqdm import trange

def remap_reg_to_healpix(
    arr,
    coords_lon,
    coords_lat,
    res_down=4,
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

def remap_healpix_to_reg(
    arr,
    coords_lon,
    coords_lat,
):
    """
    Remap data from a HEALPix grid to a regular lat/lon grid.

    Parameters
    ----------
    arr : np.ndarray
        Input array with HEALPix data, shape (..., npix).
        Any number of leading dimensions is supported.
    coords_lon : np.ndarray
        1D array of target longitudes in degrees [0, 360).
    coords_lat : np.ndarray
        1D array of target latitudes in degrees [-90, 90].

    Returns
    -------
    remapped_data : np.ndarray
        Remapped array of shape (..., n_lat, n_lon).
    """
    npix = arr.shape[-1]
    NSIDE = hp.npix2nside(npix)

    # Get the lat/lon of every HEALPix pixel
    theta, phi = hp.pix2ang(NSIDE, np.arange(npix))
    lat_hp = np.degrees(np.pi / 2 - theta)   # [-90, 90]
    lon_hp = np.degrees(phi)                  # [0, 360)

    # Build the regular grid query points
    lon_grid, lat_grid = np.meshgrid(coords_lon, coords_lat)  # (n_lat, n_lon)
    grid_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    # Source points for the interpolator
    src_points = np.column_stack([lat_hp, lon_hp])

    original_shape = arr.shape
    arr_flat = arr.reshape(-1, npix)          # (N, npix)

    n_lat = len(coords_lat)
    n_lon = len(coords_lon)
    remapped_flat = np.zeros((arr_flat.shape[0], n_lat * n_lon))

    for k in trange(arr_flat.shape[0]):
        values = arr_flat[k]                  # (npix,)

        # Primary: linear interpolation from scattered HEALPix points
        interp_linear = LinearNDInterpolator(src_points, values)
        result = interp_linear(grid_points)

        # Fallback: nearest-neighbour for any NaNs near the poles / hull edges
        nan_mask = np.isnan(result)
        if nan_mask.any():
            interp_nearest = NearestNDInterpolator(src_points, values)
            result[nan_mask] = interp_nearest(grid_points[nan_mask])

        remapped_flat[k] = result

    # Restore leading dimensions and append (n_lat, n_lon)
    new_shape = original_shape[:-1] + (n_lat, n_lon)
    remapped_data = remapped_flat.reshape(new_shape)

    return remapped_data
