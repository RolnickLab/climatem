import copy
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import beta

from climatem.synthetic_data.savar import SAVAR
from climatem.synthetic_data.utils import check_stability, create_random_mode


# Before saving the parameters to JSON, convert ndarray to list
def convert_ndarray_to_list(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, dict):
            convert_ndarray_to_list(value)


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def save_parameters_to_csv(filename, parameters):
    # Exclude array data
    # excluded_keys = ['modes_weights', 'noise_weights']
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
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Component", "Link", "Lag", "Coefficient"])
        for key, values in links_coeffs.items():
            for value in values:
                writer.writerow([key, value[0][0], value[0][1], value[1]])


# Function to create a circular mode
def create_circular_mode(shape, radius=10):
    mode = np.zeros(shape)
    center = (shape[0] // 2, shape[1] // 2)
    Y, X = np.ogrid[: shape[0], : shape[1]]
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    mask = dist_from_center <= radius
    mode[mask] = np.random.randn(np.sum(mask))
    return mode


def create_links_coeffs(n_modes, prob_edge=0.2, tau=5, a=4, b=8, difficulty="easy"):
    links_coeffs = {}
    for k in range(n_modes):
        val = 0
        links_coeffs[k] = []
        auto_reg_tau = np.random.choice(np.arange(1, tau + 1))
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
                    r = beta.rvs(a, b)
                    if difficulty == "med_hard":
                        r /= 2
                    if difficulty == "hard":
                        r /= 4
                    val += int(r * 100) / 100
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

    # CO2 forcing latents: autoregressive + effects on climate modes
    for i, co2_idx in enumerate(forcing_indices["co2"]):
        forcing_links[co2_idx] = []
        # Autoregressive term (CO2 is persistent)
        forcing_links[co2_idx].append(((co2_idx, -1), forcing_autoreg_strength))

    # Aerosol forcing latents: autoregressive + effects on climate modes
    for i, aerosol_idx in enumerate(forcing_indices["aerosol"]):
        forcing_links[aerosol_idx] = []
        # Autoregressive term (aerosols have shorter persistence)
        forcing_links[aerosol_idx].append(((aerosol_idx, -1), forcing_autoreg_strength * 0.7))

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
        # Maybe also affect neighboring mode
        if np.random.rand() > 0.5 and n_climate_modes > 1:
            neighbor = (primary_mode + 1) % n_climate_modes
            affected.append(neighbor)

        for mode_idx in affected:
            strength = aerosol_effect_strength * (0.7 + 0.6 * np.random.rand())
            # Aerosols can be negative (cooling effect)
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
    phases=[0.0, 0.7853981634, 1.5707963268],  # [0, π/4, π/2] radians
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
    n_aerosol_latents=4,
    co2_effect_strength=0.15,
    aerosol_effect_strength=0.10,
    tau=5,
):

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

    # This is the probabiliity of having a link between latent k and j, with k different from j. latents always have one link with themselves at a previous time.
    if difficulty == "easy":
        prob = 0
    if difficulty == "med_easy":
        prob = 1 / (N - 1)
    if difficulty == "med_hard":
        prob = 2 / (N - 1)
    if difficulty == "hard":
        prob = 1 / 2

    # Create climate mode links (N x N)
    climate_links_coeffs = create_links_coeffs(N, prob_edge=prob, tau=tau, difficulty=difficulty)

    # One good thing of SAVAR is that if the underlying process is stable and stationary, then SAVAR is also both.
    # Independently of W. This is, we only need to check for stationarity of \PHI and not of W^+\PHI W
    check_stability(climate_links_coeffs)

    # Initialize forcing_indices (will be populated if is_forced)
    forcing_indices = None

    if is_forced:
        # Create forcing → mode causal links
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

        print(f"Created extended causal graph with {forcing_indices['n_total']} total latents:")
        print(f"  - Climate modes: 0-{N-1}")
        print(f"  - CO2 latents: {forcing_indices['co2']}")
        print(f"  - Aerosol latents: {forcing_indices['aerosol']}")
    else:
        links_coeffs = climate_links_coeffs

    if is_forced:
        # turn off forcing by setting the time to the last time step
        w_f = modes_weights
        # A very simple method for adding a focring term (bias on the mean of the noise term)
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
        # Plot the sum of mode weights
        sum_modes = modes_weights.sum(axis=0)
        fig, ax = plt.subplots()
        im = ax.imshow(sum_modes)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title("Sum of Circular Modes")
        fig_name = "modes.png"
        modenpy_name = "modes.npy"
        fig_path = savar_dataset_dir / fig_name
        modenpy_path = savar_dataset_dir / modenpy_name
        plt.savefig(fig_path)
        np.save(modenpy_path, sum_modes)
        plt.close()

        # Plot the sum of noise weights
        sum_noise = noise_weights.sum(axis=0)
        fig, ax = plt.subplots()
        im = ax.imshow(sum_noise)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title("Sum of Circular Noise")

        fig_name = "noise_modes.png"
        noisenpy_name = "noise_modes.npy"
        fig_path = savar_dataset_dir / fig_name
        sum_noise_npypath = savar_dataset_dir / noisenpy_name

        plt.savefig(fig_path)
        np.save(sum_noise_npypath, sum_noise)
        plt.close()

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
        # Forcing causal structure info
        "is_forced": is_forced,
        "forcing_indices": forcing_indices,
        "n_co2_latents": n_co2_latents if is_forced else 0,
        "n_aerosol_latents": n_aerosol_latents if is_forced else 0,
        "n_climate_modes": N,
        "n_total_latents": forcing_indices["n_total"] if forcing_indices else N,
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
            noise_strength=noise_val,  # How to play with this parameter?
            season_dict=season_dict,
            linearity=linearity,
            poly_degrees=poly_degrees,
            output_save_dir=str(savar_dataset_dir),
        )
    else:
        savar_model = SAVAR(
            links_coeffs=links_coeffs,
            time_length=time_len,
            mode_weights=modes_weights,
            noise_strength=noise_val,
            season_dict=season_dict,
            forcing_dict=forcing_dict,
            forcing_indices=forcing_indices,  # Pass forcing indices for causal structure
            linearity=linearity,
            poly_degrees=poly_degrees,
            output_save_dir=str(savar_dataset_dir),
        )

    savar_model.generate_data()  # Remember to generate data, otherwise the data field will be empty
    np.save(save_path, savar_model.data_field)

    # Save combined forcing field (backward compatibility)
    forcing_field = getattr(savar_model, "forcing_data_field", None)
    if forcing_field is not None:
        forcing_path = savar_dataset_dir / "forcing_data_field.npy"
        np.save(forcing_path, forcing_field)

    # Save separate CO2 and aerosol forcings (for dual exogenous conditioning)
    co2_forcing_field = getattr(savar_model, "co2_forcing_data_field", None)
    if co2_forcing_field is not None:
        co2_forcing_path = savar_dataset_dir / "co2_forcing.npy"
        np.save(co2_forcing_path, co2_forcing_field)
        print(f"Saved CO2 forcing to {co2_forcing_path}")

    aerosol_forcing_field = getattr(savar_model, "aerosol_forcing_data_field", None)
    if aerosol_forcing_field is not None:
        aerosol_forcing_path = savar_dataset_dir / "aerosol_forcing.npy"
        np.save(aerosol_forcing_path, aerosol_forcing_field)
        print(f"Saved aerosol forcing to {aerosol_forcing_path}")

    # Save forcing latent trajectories (ground truth for supervision)
    co2_latent_traj = getattr(savar_model, "co2_latent_trajectory", None)
    if co2_latent_traj is not None:
        co2_latent_path = savar_dataset_dir / "co2_latent_trajectory.npy"
        np.save(co2_latent_path, co2_latent_traj)
        print(f"Saved CO2 latent trajectory to {co2_latent_path} (shape: {co2_latent_traj.shape})")

    aerosol_latent_traj = getattr(savar_model, "aerosol_latent_trajectory", None)
    if aerosol_latent_traj is not None:
        aerosol_latent_path = savar_dataset_dir / "aerosol_latent_trajectory.npy"
        np.save(aerosol_latent_path, aerosol_latent_traj)
        print(f"Saved aerosol latent trajectory to {aerosol_latent_path} (shape: {aerosol_latent_traj.shape})")

    print(f"{name} DONE!")

    return savar_model.data_field
