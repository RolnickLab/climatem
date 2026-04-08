"""
Orchestration layer for generating synthetic SAVAR datasets.

This module creates SAVAR datasets with randomized
causal structure, saves them to disk (as .npy, .csv, and .json), and produces
diagnostic plots of the spatial mode and noise weight patterns.

The main entry point is :func:`generate_save_savar_data`, which:
1. Constructs randomized spatial modes and noise weights on a 2-D grid.
2. Builds a causal link structure (optionally including CO2/aerosol forcing).
3. Instantiates a :class:`~climatem.synthetic_data.savar.SAVAR` model and
   generates both deterministic and noisy data fields.
4. Persists every artefact (parameters, weights, forcing fields, latent
   trajectories, background state) into a per-dataset directory.
"""

import copy
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import beta

from climatem.synthetic_data.savar import SAVAR
from climatem.synthetic_data.utils import check_stability, create_random_mode
from climatem.utils import get_logger

logger = get_logger(__name__)


def convert_ndarray_to_list(d):
    """
    Recursively convert all :class:`numpy.ndarray` values in *d* to Python lists.

    This is used to make a parameter dictionary JSON-serialisable before saving.

    Parameters
    ----------
    d : dict
        Dictionary whose values may contain ndarrays or nested dicts.
    """
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, dict):
            convert_ndarray_to_list(value)


def np_encoder(object):
    """
    JSON encoder fallback for numpy scalar types.

    Passed as the ``default`` argument to :func:`json.dump` so that
    ``np.float64``, ``np.int64``, etc. are converted to native Python types.

    Parameters
    ----------
    object : Any
        The object that the default JSON encoder could not serialise.

    Returns
    -------
    int or float or None
        The Python-native equivalent, or ``None`` if *object* is not a numpy
        generic type (which will cause :func:`json.dump` to raise).
    """
    if isinstance(object, np.generic):
        return object.item()


def save_parameters_to_csv(filename, parameters):
    """
    Write experiment parameters to a human-readable CSV file.

    Large array-valued parameters listed in *excluded_keys* are omitted to
    keep the CSV concise.  Dictionary values (e.g. ``links_coeffs``) are
    serialised as JSON strings.

    Parameters
    ----------
    filename : str or pathlib.Path
        Destination CSV path.
    parameters : dict
        Flat or shallowly nested parameter dictionary.
    """
    excluded_keys = ["noise_weights"]  # keep noise weights to get permutations
    filtered_params = {key: value for key, value in parameters.items() if key not in excluded_keys}

    # Open the file in write mode
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])
        for key, value in filtered_params.items():
            if isinstance(value, dict) and key == "links_coeffs":
                # Convert dictionaries to a JSON string for better readability and to preserve structure
                value = json.dumps(value, default=np_encoder)
            elif isinstance(value, dict):
                value = json.dumps(value)
            writer.writerow([key, value])


def save_links_coeffs_to_csv(filename, links_coeffs):
    """
    Write the causal link structure to a CSV with one row per edge.

    Each row contains the target component index, the source component index,
    the time lag, and the edge coefficient.

    Parameters
    ----------
    filename : str or pathlib.Path
        Destination CSV path.
    links_coeffs : dict[int, list[tuple]]
        Mapping from target component index to a list of
        ``((source_index, lag), coefficient)`` tuples.
    """
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Component", "Link", "Lag", "Coefficient"])
        for key, values in links_coeffs.items():
            for value in values:
                writer.writerow([key, value[0][0], value[0][1], value[1]])


def create_circular_mode(shape, radius=10):
    """
    Create a spatial mode with non-zero random values inside a circular mask.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(height, width)`` of the output array.
    radius : int, optional
        Radius (in pixels) of the circular region centred in the grid.

    Returns
    -------
    numpy.ndarray
        2-D array of the given *shape* with standard-normal values inside the
        circle and zeros outside.
    """
    mode = np.zeros(shape)
    center = (shape[0] // 2, shape[1] // 2)
    Y, X = np.ogrid[: shape[0], : shape[1]]
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    mask = dist_from_center <= radius
    mode[mask] = np.random.randn(np.sum(mask))
    return mode


def create_links_coeffs(n_modes, prob_edge=0.2, tau=5, a=4, b=8, difficulty="easy"):
    """
    Build a randomised causal link structure for *n_modes* latent components.

    Every component receives exactly one autoregressive (self-)link at a random
    lag in ``[1, tau]``.  Cross-component links are added with probability
    *prob_edge*, subject to a stability guard that keeps the total absolute
    coefficient for each target below 1.

    Parameters
    ----------
    n_modes : int
        Number of latent components.
    prob_edge : float
        Probability that a directed edge j -> k exists (for j != k).
    tau : int
        Maximum time lag for any causal link.
    a, b : float
        Shape parameters of the Beta(a, b) distribution used to draw link
        coefficient magnitudes.  The default ``a=4, b=8`` yields a right-skewed
        distribution concentrated around 0.3, producing moderate-strength links
        that are unlikely to be very large (keeps the VAR process stable).
    difficulty : str
        One of ``"easy"``, ``"med_easy"``, ``"med_hard"``, ``"hard"``.
        Higher difficulty halves or quarters coefficient magnitudes, making
        causal discovery harder.

    Returns
    -------
    dict[int, list[tuple]]
        Mapping from target component index to a list of
        ``((source_index, -lag), coefficient)`` tuples.
    """
    links_coeffs = {}
    for k in range(n_modes):
        val = 0
        links_coeffs[k] = []
        auto_reg_tau = np.random.choice(np.arange(1, tau + 1))
        # Draw coefficient magnitude from Beta(a=4, b=8); mean ~0.33, right-skewed
        r = beta.rvs(a, b)
        if difficulty == "med_hard":
            r /= 2
        if difficulty == "hard":
            r /= 4
        links_coeffs[k].append(((k, -auto_reg_tau), int(r * 100) / 100))
        val += int(r * 100) / 100
        arr = np.arange(n_modes)
        np.random.shuffle(arr)
        for j in arr:
            if j != k:
                auto_reg_tau = np.random.choice(np.arange(1, tau + 1))
                if np.random.choice([0, 1], p=[1 - prob_edge, prob_edge]):
                    # Draw cross-link coefficient from the same Beta(a=4, b=8)
                    r = beta.rvs(a, b)
                    if difficulty == "med_hard":
                        r /= 2
                    if difficulty == "hard":
                        r /= 4
                    val += int(r * 100) / 100
                    # Guard: total absolute coefficient for target k must stay < 1
                    # to help ensure VAR stability
                    if val < 1:
                        links_coeffs[k].append(((j, -auto_reg_tau), int(r * 100) / 100))
    return links_coeffs


def create_forcing_links_coeffs(
    n_climate_modes,
    n_co2_latents=1,
    n_aerosol_latents=4,
    tau=5,
    co2_effect_strength=0.15,
    aerosol_effect_strength=0.10,
    co2_affected_modes=None,
    aerosol_affected_modes=None,
    forcing_autoreg_strength=0.8,
):
    """
    Create causal links from forcing latents to climate modes.

    This extends the links_coeffs structure to include:
    - CO2 forcing latent(s) → climate modes
    - Aerosol forcing latent(s) → climate modes
    - Autoregressive terms for forcing latents

    Forcing indices are assigned as:
    - Climate modes: 0 to n_climate_modes-1
    - CO2 latents: n_climate_modes to n_climate_modes + n_co2_latents - 1
    - Aerosol latents: n_climate_modes + n_co2_latents to end

    Args:
        n_climate_modes: Number of climate modes (e.g., 4)
        n_co2_latents: Number of latents for CO2 (default 1)
        n_aerosol_latents: Number of latents for aerosols (default 4)
        tau: Maximum time lag
        co2_effect_strength: Coefficient strength for CO2 → mode connections
        aerosol_effect_strength: Coefficient strength for aerosol → mode connections
        co2_affected_modes: List of climate mode indices affected by CO2 (default: all)
        aerosol_affected_modes: List of climate mode indices affected by aerosols (default: all)
        forcing_autoreg_strength: Autoregressive coefficient for forcing latents

    Returns:
        forcing_links: Dictionary with forcing → mode causal links
        forcing_indices: Dict with 'co2' and 'aerosol' index ranges
    """
    # Default: all climate modes are affected by forcings
    if co2_affected_modes is None:
        co2_affected_modes = list(range(n_climate_modes))
    if aerosol_affected_modes is None:
        aerosol_affected_modes = list(range(n_climate_modes))

    # Compute forcing indices
    co2_start_idx = n_climate_modes
    co2_end_idx = co2_start_idx + n_co2_latents
    aerosol_start_idx = co2_end_idx
    aerosol_end_idx = aerosol_start_idx + n_aerosol_latents

    forcing_indices = {
        "co2": list(range(co2_start_idx, co2_end_idx)),
        "aerosol": list(range(aerosol_start_idx, aerosol_end_idx)),
        "n_total": aerosol_end_idx,
    }

    forcing_links = {}

    # Initialize forcing latent entries (no self-loops: forcings are exogenous,
    # their trajectories are prescribed by _generate_forcing_trajectories())
    for co2_idx in forcing_indices["co2"]:
        forcing_links[co2_idx] = []
    for aerosol_idx in forcing_indices["aerosol"]:
        forcing_links[aerosol_idx] = []

    # Add CO2 → climate mode connections
    # CO2 affects all modes with lag 1 (immediate effect on next timestep)
    for mode_idx in co2_affected_modes:
        for co2_idx in forcing_indices["co2"]:
            # Add with some random variation in strength
            strength = co2_effect_strength * (0.8 + 0.4 * np.random.rand())
            lag = -np.random.choice([1, 2])  # Lag 1 or 2
            # This link means: climate_mode[mode_idx] is caused by co2[co2_idx] at lag
            # We store this in the CLIMATE MODE's entry (target's perspective)
            if mode_idx not in forcing_links:
                forcing_links[mode_idx] = []
            forcing_links[mode_idx].append(((co2_idx, lag), round(strength, 3)))

    # Add Aerosol → climate mode connections
    # Each aerosol latent affects a subset of modes (more localized effect)
    for i, aerosol_idx in enumerate(forcing_indices["aerosol"]):
        # Each aerosol latent primarily affects ~1-2 climate modes
        # Distribute aerosol effects across modes
        primary_mode = i % n_climate_modes
        affected = [primary_mode]
        # 50% chance of also affecting the neighbouring mode
        if np.random.rand() > 0.5 and n_climate_modes > 1:
            neighbor = (primary_mode + 1) % n_climate_modes
            affected.append(neighbor)

        for mode_idx in affected:
            strength = aerosol_effect_strength * (0.7 + 0.6 * np.random.rand())
            # 70% chance the aerosol effect is negative (cooling), reflecting
            # the dominant real-world radiative effect of sulphate aerosols
            if np.random.rand() > 0.3:
                strength = -strength
            lag = -np.random.choice([1, 2, 3])
            if mode_idx not in forcing_links:
                forcing_links[mode_idx] = []
            forcing_links[mode_idx].append(((aerosol_idx, lag), round(strength, 3)))

    return forcing_links, forcing_indices


def merge_links_coeffs(climate_links, forcing_links):
    """
    Merge climate mode links with forcing links into a single links_coeffs dict.

    Args:
        climate_links: links_coeffs for climate modes (indices 0 to N-1)
        forcing_links: links from forcing latents (includes forcing → mode links)

    Returns:
        merged: Combined links_coeffs dictionary
    """
    merged = {}

    # Copy climate links
    for k, v in climate_links.items():
        merged[k] = list(v)

    # Add/extend with forcing links
    for k, v in forcing_links.items():
        if k in merged:
            merged[k].extend(v)
        else:
            merged[k] = list(v)

    return merged


def _plot_mode_weights(modes_weights, noise_weights, savar_dataset_dir):
    """Plot and save spatial mode and noise weight visualizations."""
    sum_modes = modes_weights.sum(axis=0)
    fig, ax = plt.subplots()
    im = ax.imshow(sum_modes)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title("Sum of Circular Modes")
    plt.savefig(savar_dataset_dir / "modes.png")
    np.save(savar_dataset_dir / "modes.npy", modes_weights)
    plt.close()

    sum_noise = noise_weights.sum(axis=0)
    fig, ax = plt.subplots()
    im = ax.imshow(sum_noise)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title("Sum of Circular Noise")
    plt.savefig(savar_dataset_dir / "noise_modes.png")
    np.save(savar_dataset_dir / "noise_modes.npy", sum_noise)
    plt.close()


def _log_snr_metrics(noise_field, deterministic_data, noisy_data, noise_val):
    """Compute and log signal-to-noise ratio metrics."""
    if noise_field is None or deterministic_data is None:
        return
    actual_noise = noisy_data - deterministic_data
    signal_std = np.std(deterministic_data)
    noise_std = np.std(actual_noise)
    snr_std = signal_std / noise_std if noise_std > 0 else np.inf
    snr_db = 10 * np.log10(snr_std) if snr_std > 0 else -np.inf

    logger.info("SIGNAL-TO-NOISE RATIO METRICS")
    logger.info("Signal std:        %.6f", signal_std)
    logger.info("Noise std:         %.6f", noise_std)
    logger.info("SNR (std):         %.3f (%.2f dB)", snr_std, snr_db)
    logger.info("Noise strength:    %.6f", noise_val)


def _save_savar_artifacts(savar_model, savar_dataset_dir):
    """Save forcing fields, latent trajectories, templates, and background to disk."""
    artifacts = [
        ("forcing_data_field", "forcing_data_field.npy"),
        ("co2_forcing_data_field", "co2_forcing.npy"),
        ("aerosol_forcing_data_field", "aerosol_forcing.npy"),
        ("co2_latent_trajectory", "co2_latent_trajectory.npy"),
        ("aerosol_latent_trajectory", "aerosol_latent_trajectory.npy"),
        ("aerosol_spatial_templates", "aerosol_spatial_templates.npy"),
    ]
    for attr_name, filename in artifacts:
        data = getattr(savar_model, attr_name, None)
        if data is not None:
            path = savar_dataset_dir / filename
            np.save(path, data)
            logger.debug("Saved %s to %s (shape: %s)", attr_name, path, data.shape)

    background_field = getattr(savar_model, "background_data_field", None)
    if background_field is not None:
        background_path = savar_dataset_dir / "background_data_field.npy"
        np.save(background_path, background_field)
        logger.debug("Saved background data field to %s (shape: %s)", background_path, background_field.shape)

        bg_std = background_field.std()
        final_data = savar_model.data_field
        data_std = final_data.std() if final_data is not None else 0.0
        ratio = bg_std / data_std if data_std > 0 else 0.0
        logger.info("Background contribution: std=%.6f, ratio to total data=%.3f", bg_std, ratio)


def generate_save_savar_data(
    save_dir_path,
    name,
    time_len=10_000,
    comp_size=10,
    noise_val=0.2,
    n_per_col=2,  # Number of components N = n_per_col**2
    difficulty="easy",
    seasonality=True,
    periods=[12, 6, 3],
    amplitudes=[0.1, 0.05, 0.02],
    phases=[0.0, 0.7853981634, 1.5707963268],  # [0, pi/4, pi/2] radians
    yearly_jitter_amp: float = 0.05,
    yearly_jitter_phase: float = 0.10,
    overlap=0,
    is_forced=False,
    f_1=1,
    f_2=2,
    f_time_1=2000,
    f_time_2=8000,
    ramp_type="linear",
    linearity="polynomial",
    poly_degrees=[2, 3],
    plotting=True,
    aerosol_scale=0.02,
    aerosol_spatial_contrast=1.05,
    aerosol_ramp_up_time=2000,
    aerosol_peak_time=5000,
    aerosol_decline_time=8000,
    # Forcing causal structure parameters
    n_co2_latents=1,
    n_aerosol_latents=2,  # Updated from 4 to 2 (2026-01-22 aerosol refactor)
    co2_effect_strength=0.25,  # Updated from 0.15 to 0.25
    aerosol_effect_strength=0.20,  # Updated from 0.10 to 0.20
    forcing_amplification=1.0,  # Updated from 1.5 to 1.0
    noise_ar1=True,  # Use AR(1) noise for realistic temporal correlations
    noise_ar1_rho=0.95,  # AR(1) persistence parameter rho (or "decay" for mode-dependent rho_k)
    tau=5,
    # Background state parameters
    enable_background=False,
    background_strength=0.3,
    background_strength_mode="relative",
    background_smoothness=0.15,
    background_timescale_rho=0.995,
    background_n_modes=3,
):
    """
    Generate a complete SAVAR dataset and persist all artefacts to disk.

    This is the main orchestration function.  It builds spatial modes, creates
    a randomised causal graph (optionally extended with CO2/aerosol forcing
    latents), generates deterministic and noisy time series via
    :class:`~climatem.synthetic_data.savar.SAVAR`, and saves everything
    (parameters, weights, forcing fields, latent trajectories, diagnostic
    plots) into ``save_dir_path / name``.

    Parameters
    ----------
    **Grid parameters**

    save_dir_path : pathlib.Path
        Parent directory under which a sub-folder *name* will be created.
    name : str
        Identifier for this dataset; used as the sub-folder name and in logs.
    comp_size : int, optional
        Side length (in pixels) of each spatial component tile.  Default 10.
    n_per_col : int, optional
        Number of components per row/column; total components N = n_per_col**2.
    overlap : float, optional
        Spatial overlap between component tiles, in [0, 1].  0 = no overlap,
        1 = all tiles centred at the grid midpoint.

    **Temporal parameters**

    time_len : int, optional
        Number of time steps to generate.  Default 10 000, chosen to provide
        enough samples for stable VAR estimation and spectral analysis while
        keeping generation time manageable.
    tau : int, optional
        Maximum causal time lag (in time steps).  Default 5.
    seasonality : bool, optional
        Whether to add seasonal (periodic) components.
    periods, amplitudes, phases : list, optional
        Lists defining the harmonic decomposition of the seasonal cycle.
    yearly_jitter_amp, yearly_jitter_phase : float, optional
        Year-to-year random perturbation of seasonal amplitude and phase.

    **Causal structure parameters**

    difficulty : str, optional
        Controls edge density and coefficient magnitude.  One of
        ``"easy"`` (no cross-links), ``"med_easy"`` (sparse),
        ``"med_hard"`` (moderate, halved coefficients),
        ``"hard"`` (dense, quartered coefficients).

    **Forcing parameters**

    is_forced : bool, optional
        Whether to include exogenous CO2 and aerosol forcing.
    f_1, f_2 : float, optional
        Forcing magnitude at the first and second plateau, respectively.
    f_time_1 : int, optional
        End of the first (low) forcing plateau.  Default 2000, representing
        the pre-industrial steady-state period (~20 % of the time series).
    f_time_2 : int, optional
        Start of the second (high) forcing plateau.  Default 8000,
        representing the point at which forcing stabilises (~80 %).
    ramp_type : str, optional
        Interpolation between the two plateaus (``"linear"`` or other).
    aerosol_scale, aerosol_spatial_contrast : float, optional
        Magnitude and spatial heterogeneity of aerosol forcing.
    aerosol_ramp_up_time : int, optional
        Time step at which aerosol emissions begin increasing.  Default 2000,
        aligned with f_time_1 to co-locate the start of anthropogenic forcing.
    aerosol_peak_time : int, optional
        Time step at which aerosol emissions peak.  Default 5000, placing the
        peak at the midpoint of the ramp-up window.
    aerosol_decline_time : int, optional
        Time step at which aerosol emissions finish declining.  Default 8000,
        aligned with f_time_2 to model clean-air regulations.
    n_co2_latents, n_aerosol_latents : int, optional
        Number of latent variables representing CO2 and aerosol forcing.
    co2_effect_strength, aerosol_effect_strength : float, optional
        Coefficient magnitude for forcing -> climate mode causal links.
    forcing_amplification : float, optional
        Global multiplier applied to all forcing signals.

    **Noise parameters**

    noise_val : float, optional
        Noise standard deviation (before AR(1) colouring).
    noise_ar1 : bool, optional
        Whether to use temporally correlated AR(1) noise.
    noise_ar1_rho : float, optional
        AR(1) persistence parameter.

    **Background state parameters**

    enable_background : bool, optional
        Whether to add a slowly varying background state.
    background_strength : float, optional
        Magnitude of the background state.
    background_strength_mode : str, optional
        ``"relative"`` scales background relative to signal std.
    background_smoothness : float, optional
        Controls spatial smoothness of background modes.
    background_timescale_rho : float, optional
        AR(1) coefficient for the background time series.
    background_n_modes : int, optional
        Number of spatial modes used to construct the background.

    **Miscellaneous**

    linearity : str, optional
        ``"linear"`` or ``"nonlinear"``; selects the SAVAR transition model.
    poly_degrees : list[int], optional
        Polynomial degrees used when ``linearity="nonlinear"``.
    plotting : bool, optional
        Whether to save diagnostic spatial-mode plots.

    Returns
    -------
    numpy.ndarray
        The generated (noisy) data field with shape ``(time_len, nx * ny)``.
    """

    # Setup spatial weights of underlying processes
    ny = nx = n_per_col * comp_size
    N = n_per_col**2  # Number of components

    if not (0 <= overlap <= 1):
        raise ValueError("overlap must be between 0 and 1")

    noise_weights = np.zeros((N, nx, ny))
    modes_weights = np.zeros((N, nx, ny))

    # Create a subfolder for this specific SAVAR dataset
    savar_dataset_dir = save_dir_path / name
    savar_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Specify the path where you want to save the data
    npy_name = "savar.npy"
    save_path = savar_dataset_dir / npy_name

    # Center starting position (for fully overlapping modes)
    center_x_start = (nx - comp_size) // 2
    center_y_start = (ny - comp_size) // 2

    # Create modes weights
    for k in range(n_per_col):
        for j in range(n_per_col):
            idx = k * n_per_col + j
            # Original starting position (no overlap)
            orig_x_start = k * comp_size
            orig_y_start = j * comp_size
            # New starting positions (interpolated between original and central)
            new_x_start = int((1 - overlap) * orig_x_start + overlap * center_x_start)
            new_y_start = int((1 - overlap) * orig_y_start + overlap * center_y_start)
            new_x_end = new_x_start + comp_size
            new_y_end = new_y_start + comp_size
            modes_weights[idx, new_x_start:new_x_end, new_y_start:new_y_end] = create_random_mode(
                (comp_size, comp_size), random=True
            )
            # for k in range(n_per_col):
            #    for j in range(n_per_col):
            noise_weights[idx, new_x_start:new_x_end, new_y_start:new_y_end] = create_random_mode(
                (comp_size, comp_size), random=True
            )

    # Probability of having a link between latent k and j (k != j).
    # Latents always have one autoregressive link with themselves at a previous time.
    if difficulty == "easy":
        prob = 0  # expected N out of N^2 total links
    if difficulty == "med_easy":
        prob = 1 / (N - 1)  # expected 2N out of N^2 total links
    if difficulty == "med_hard":
        prob = 2 / (N - 1)  # expected 3N out of N^2 total links
    if difficulty == "hard":
        prob = 1 / 2  # (N + 2*N / 2)/N^2

    # Create climate mode links (N x N)
    climate_links_coeffs = create_links_coeffs(N, prob_edge=prob, tau=tau, difficulty=difficulty)

    # One good thing of SAVAR is that if the underlying process is stable and stationary, then SAVAR is also both.
    # Independently of W. This is, we only need to check for stationarity of \PHI and not of W^+\PHI W
    check_stability(climate_links_coeffs)

    # Initialize forcing_indices (will be populated if is_forced)
    forcing_indices = None

    if is_forced:
        # Create forcing -> mode causal links
        forcing_links, forcing_indices = create_forcing_links_coeffs(
            n_climate_modes=N,
            n_co2_latents=n_co2_latents,
            n_aerosol_latents=n_aerosol_latents,
            tau=tau,
            co2_effect_strength=co2_effect_strength,
            aerosol_effect_strength=aerosol_effect_strength,
        )

        # Merge climate and forcing links into complete links_coeffs
        links_coeffs = merge_links_coeffs(climate_links_coeffs, forcing_links)

        logger.info(
            "Created extended causal graph with %d total latents: "
            "climate modes 0-%d, CO2 latents %s, aerosol latents %s",
            forcing_indices["n_total"],
            N - 1,
            forcing_indices["co2"],
            forcing_indices["aerosol"],
        )
    else:
        links_coeffs = climate_links_coeffs

    if is_forced:
        # turn off forcing by setting the time to the last time step
        w_f = modes_weights
        # A very simple method for adding a forcing term (bias on the mean of the noise term)
        forcing_dict = {
            "w_f": w_f,  # Shape of the mode of the forcing
            "f_1": f_1,  # Value of the forcing at period_1
            "f_2": f_2,  # Value of the forcing at period_2
            "f_time_1": f_time_1,  # The period one goes from t=0  to t=f_time_1
            "f_time_2": f_time_2,  # The period two goes from t= f_time_2 to the end. Between the two periods, the forcing is risen linearly
            "time_len": time_len,
            "ramp_type": ramp_type,
            "aerosol_scale": aerosol_scale,  # Scale parameter for aerosol forcing
            "aerosol_spatial_contrast": aerosol_spatial_contrast,  # Spatial contrast parameter
            "aerosol_ramp_up_time": aerosol_ramp_up_time,  # When aerosols start increasing
            "aerosol_peak_time": aerosol_peak_time,  # When aerosols peak
            "aerosol_decline_time": aerosol_decline_time,  # When aerosols finish declining
        }

    season_dict = None
    if seasonality:
        lat = np.linspace(-90, 90, nx)  # vary along rows
        lat2d = np.repeat(lat[:, None], ny, axis=1)  # shape (nx, ny)
        season_weight = np.abs(np.sin(2 * np.deg2rad(lat2d))).ravel()

        if phases is None:
            phases = [0.0] * len(amplitudes)

        if not (len(amplitudes) == len(periods) == len(phases)):
            raise ValueError("season_amplitudes, season_periods, season_phases must have identical lengths.")

        season_dict = {
            "amplitudes": amplitudes,  # e.g. [0.06, 0.02, 0.01]
            "periods": periods,  # e.g. [365, 182.5, 60]
            "phases": phases,  # radian offsets
            "season_weight": season_weight,
            "yearly_jitter": {
                "amplitude": yearly_jitter_amp,  # e.g. 0.05
                "phase": yearly_jitter_phase,  # e.g. 0.10
            },
        }

    if plotting:
        _plot_mode_weights(modes_weights, noise_weights, savar_dataset_dir)

    # Creating a dictionary of parameters
    parameters = {
        "name": name,
        "nx": nx,
        "ny": ny,
        "T": time_len,
        "N": N,
        "links_coeffs": links_coeffs,
        "f_1": f_1,
        "f_2": f_2,
        "f_time_1": f_time_1,
        "f_time_2": f_time_2,
        "ramp_type": ramp_type,
        "linearity": linearity,
        "poly_degrees": poly_degrees,
        "season_dict": season_dict,
        "seasonality": True,
        "noise_val": noise_val,
        # Forcing causal structure info
        "is_forced": is_forced,
        "forcing_indices": forcing_indices,
        "n_co2_latents": n_co2_latents if is_forced else 0,
        "n_aerosol_latents": n_aerosol_latents if is_forced else 0,
        "n_climate_modes": N,
        "n_total_latents": forcing_indices["n_total"] if forcing_indices else N,
        # Forcing signal strength parameters
        "forcing_amplification": forcing_amplification if is_forced else None,
        "co2_effect_strength": co2_effect_strength if is_forced else None,
        "aerosol_effect_strength": aerosol_effect_strength if is_forced else None,
        # Background state parameters
        "enable_background": enable_background,
        "background_strength": background_strength,
        "background_strength_mode": background_strength_mode,
        "background_smoothness": background_smoothness,
        "background_timescale_rho": background_timescale_rho,
        "background_n_modes": background_n_modes,
    }

    parameters_copy = copy.deepcopy(parameters)
    convert_ndarray_to_list(parameters_copy)  # safe to mutate

    # Specify the path to save the parameters
    param_names = "parameters.npy"
    params_path = savar_dataset_dir / param_names
    # Save the dictionary of parameters to a .npy file
    np.save(params_path, parameters)

    param_names = "parameters.csv"
    params_path = savar_dataset_dir / param_names
    save_parameters_to_csv(params_path, parameters_copy)
    param_names = "links_coeffs.csv"
    params_path = savar_dataset_dir / param_names
    save_links_coeffs_to_csv(params_path, parameters["links_coeffs"])
    param_names = "mode_weights.npy"
    params_path = savar_dataset_dir / param_names
    np.save(params_path, modes_weights)

    # Save noise_weights for diagnostics
    param_names = "noise_weights.npy"
    params_path = savar_dataset_dir / param_names
    np.save(params_path, noise_weights)

    # Create a copy of the parameters to modify
    convert_ndarray_to_list(parameters_copy)

    # Specify the path to save the parameters
    param_names = "parameters.json"
    params_path = savar_dataset_dir / param_names

    # Save the dictionary of parameters to a JSON file
    with open(params_path, "w") as json_file:
        json.dump(parameters_copy, json_file, indent=4, default=np_encoder)

    # Add the parameters
    if not is_forced:
        savar_model = SAVAR(
            links_coeffs=links_coeffs,
            time_length=time_len,
            mode_weights=modes_weights,
            # noise_weights defaults to mode_weights inside SAVAR (baseline behavior)
            noise_strength=noise_val,
            season_dict=season_dict,
            linearity=linearity,
            poly_degrees=poly_degrees,
            output_save_dir=str(savar_dataset_dir),
            # Background state parameters
            enable_background=enable_background,
            background_strength=background_strength,
            background_strength_mode=background_strength_mode,
            background_smoothness=background_smoothness,
            background_timescale_rho=background_timescale_rho,
            background_n_modes=background_n_modes,
        )
    else:
        savar_model = SAVAR(
            links_coeffs=links_coeffs,
            time_length=time_len,
            mode_weights=modes_weights,
            # noise_weights defaults to mode_weights inside SAVAR (baseline behavior)
            noise_strength=noise_val,
            noise_ar1=noise_ar1,
            noise_ar1_rho=noise_ar1_rho,
            season_dict=season_dict,
            forcing_dict=forcing_dict,
            forcing_indices=forcing_indices,  # Pass forcing indices for causal structure
            forcing_amplification=forcing_amplification,
            linearity=linearity,
            poly_degrees=poly_degrees,
            output_save_dir=str(savar_dataset_dir),
            # Background state parameters
            enable_background=enable_background,
            background_strength=background_strength,
            background_strength_mode=background_strength_mode,
            background_smoothness=background_smoothness,
            background_timescale_rho=background_timescale_rho,
            background_n_modes=background_n_modes,
        )

    # Generate data with noise (baseline behavior: noise added to data_field before dynamics)
    logger.info("Generating data with noise (baseline behavior)...")
    savar_model.generate_data(include_noise=True)
    np.save(save_path, savar_model.data_field)
    logger.info("Saved noisy data to %s (shape: %s)", save_path, savar_model.data_field.shape)

    # Save noise data field for diagnostics
    noise_field = getattr(savar_model, "noise_data_field", None)
    if noise_field is not None:
        noise_data_path = savar_dataset_dir / "noise_data_field.npy"
        np.save(noise_data_path, noise_field)
        logger.debug("Saved noise data field to %s (shape: %s)", noise_data_path, noise_field.shape)

    # Also generate deterministic data (separate pass, for diagnostics/SNR)
    # Save all deterministic components so they can be preserved
    saved_seasonal = savar_model.seasonal_data_field
    saved_forcing = savar_model.forcing_data_field
    saved_co2_forcing = savar_model.co2_forcing_data_field
    saved_aerosol_forcing = savar_model.aerosol_forcing_data_field
    saved_background = savar_model.background_data_field
    saved_co2_latent = savar_model.co2_latent_trajectory
    saved_aerosol_latent = savar_model.aerosol_latent_trajectory
    saved_aerosol_templates = savar_model.aerosol_spatial_templates

    savar_model.data_field = None
    savar_model.seasonal_data_field = saved_seasonal
    savar_model.forcing_data_field = saved_forcing
    savar_model.co2_forcing_data_field = saved_co2_forcing
    savar_model.aerosol_forcing_data_field = saved_aerosol_forcing
    savar_model.background_data_field = saved_background
    savar_model.co2_latent_trajectory = saved_co2_latent
    savar_model.aerosol_latent_trajectory = saved_aerosol_latent
    savar_model.aerosol_spatial_templates = saved_aerosol_templates

    logger.info("Generating deterministic data (no noise, for diagnostics)...")
    savar_model.generate_data(include_noise=False)
    deterministic_data = savar_model.data_field.copy()
    deterministic_path = savar_dataset_dir / "savar_deterministic.npy"
    np.save(deterministic_path, deterministic_data)
    logger.info("Saved deterministic data to %s (shape: %s)", deterministic_path, deterministic_data.shape)

    # Restore the noisy data as the primary output
    noisy_data = np.load(save_path)
    savar_model.data_field = noisy_data

    _log_snr_metrics(noise_field, deterministic_data, noisy_data, noise_val)

    _save_savar_artifacts(savar_model, savar_dataset_dir)

    logger.info("%s DONE!", name)

    return savar_model.data_field
