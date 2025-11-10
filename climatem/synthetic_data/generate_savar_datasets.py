import copy
import csv
import json

import matplotlib.animation as animation
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
    f_time_1=4000,
    f_time_2=8000,
    ramp_type="linear",
    linearity="polynomial",
    poly_degrees=[2, 3],
    plotting=True,
):

    # Setup spatial weights of underlying processes
    ny = nx = n_per_col * comp_size
    N = n_per_col**2  # Number of components

    if not (0 <= overlap <= 1):
        raise ValueError("overlap must be between 0 and 1")

    noise_weights = np.zeros((N, nx, ny))
    modes_weights = np.zeros((N, nx, ny))

    # Specify the path where you want to save the data
    npy_name = f"{name}.npy"
    save_path = save_dir_path / npy_name

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

    links_coeffs = create_links_coeffs(N, prob_edge=prob, difficulty=difficulty)

    # One good thing of SAVAR is that if the underlying process is stable and stationary, then SAVAR is also both.
    # Independently of W. This is, we only need to check for stationarity of \PHI and not of W^+\PHI W
    check_stability(links_coeffs)

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
        fig_name = f"{name}_modes.png"
        modenpy_name = f"{name}_modes.npy"
        fig_path = save_dir_path / fig_name
        modenpy_path = save_dir_path / modenpy_name
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

        fig_name = f"{name}_noise_modes.png"
        noisenpy_name = f"{name}_noise_modes.npy"
        fig_path = save_dir_path / fig_name
        sum_noise_npypath = save_dir_path / noisenpy_name

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
    }

    parameters_copy = copy.deepcopy(parameters)
    convert_ndarray_to_list(parameters_copy)  # safe to mutate

    # Specify the path to save the parameters
    param_names = f"{name}_parameters.npy"
    params_path = save_dir_path / param_names
    # Save the dictionary of parameters to a .npy file
    np.save(params_path, parameters)

    param_names = f"{name}_parameters.csv"
    params_path = save_dir_path / param_names
    save_parameters_to_csv(params_path, parameters_copy)
    param_names = f"{name}_links_coeffs.csv"
    params_path = save_dir_path / param_names
    save_links_coeffs_to_csv(params_path, parameters["links_coeffs"])
    param_names = f"{name}_mode_weights.npy"
    params_path = save_dir_path / param_names
    np.save(params_path, modes_weights)

    # Create a copy of the parameters to modify
    convert_ndarray_to_list(parameters_copy)

    # Specify the path to save the parameters
    param_names = f"{name}_parameters.json"
    params_path = save_dir_path / param_names

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
            season_dict=season_dict,  # turn off by commenting out
            # forcing_dict=forcing_dict #turn off by commenting out
            linearity=linearity,
            poly_degrees=poly_degrees,
        )
    else:
        savar_model = SAVAR(
            links_coeffs=links_coeffs,
            time_length=time_len,
            mode_weights=modes_weights,
            noise_strength=noise_val,
            season_dict=season_dict,  # turn off by commenting out
            forcing_dict=forcing_dict,  # turn off by commenting out
            linearity=linearity,
            poly_degrees=poly_degrees,
        )

    savar_model.generate_data()  # Remember to generate data, otherwise the data field will be empty
    np.save(save_path, savar_model.data_field)

    print(f"{name} DONE!")

    plot_original_savar(savar_model.data_field, nx, ny, save_dir_path / f"{name}_original_savar_data2.gif")

    if seasonality:
        plot_seasonality_only(savar_model.seasonal_data_field, nx, ny, save_dir_path / f"{name}_seasonality_only.gif")

        # Check the seasonality
        season_std = savar_model.seasonal_data_field.std()
        total_std = savar_model.data_field.std()
        ratio = season_std / total_std
        print(f"[diag] σ_season / σ_total = {ratio:.3f}")

        if ratio < 0.05:
            print("[diag] Seasonality is tiny – raise amplitudes or lower noise_val.")
        elif ratio < 0.15:
            print("[diag] Seasonality is subtle – visible but easy to miss.")
        else:
            print("[diag] Seasonality should be visually obvious.")

        hovmoller_seasonality(
            savar_model.seasonal_data_field,
            nx,
            ny,
            n_steps=100,
            path=save_dir_path / f"{name}_seasonality_hovmoller.png",
        )

    return savar_model.data_field


def plot_original_savar(data, lat, lon, path):
    """Plotting the original savar data."""
    print(f"data shape {data.shape}")
    # Get the dimensions
    time_steps = data.shape[1]
    data_reshaped = data.T.reshape((time_steps, lat, lon))

    avg_data = np.mean(data_reshaped, axis=0)
    vmin_avg, vmax_avg = avg_data.min(), avg_data.max()
    vmin_all, vmax_all = data_reshaped.min(), data_reshaped.max()
    print(f"avg_data range: {vmin_avg:.3e} to {vmax_avg:.3e}")
    print(f"full data range: {vmin_all:.3e} to {vmax_all:.3e}")

    fig, ax = plt.subplots(figsize=(lon / 10, lat / 10))
    cax = ax.imshow(data_reshaped[0], aspect="auto", cmap="viridis", vmin=vmin_avg, vmax=vmax_avg)
    # cbar = fig.colorbar(cax, ax=ax)

    def animate(i):
        cax.set_data(data_reshaped[i])
        ax.set_title(f"Time step: {i+1}")
        return (cax,)

    # Create an animation
    ani = animation.FuncAnimation(fig, animate, frames=100, blit=False)

    # Save the animation as a video file
    ani.save(path, writer="pillow", fps=10)

    plt.close()


# Save an animation of the pure seasonality component
def plot_seasonality_only(seasonal_field, nx, ny, path, n_frames=100):
    """
    Save an animation (GIF) of the pure seasonal component.

    Parameters
    ----------
    seasonal_field : (L, T) ndarray
    nx, ny         : grid dims  (L must equal nx*ny)
    path           : output filename (.gif)
    n_frames       : how many timesteps to animate (default 100)
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np

    T = seasonal_field.shape[1]
    n_frames = min(n_frames, T)

    vmax = np.abs(seasonal_field).max()
    vmin = -vmax

    field_2d = seasonal_field[:, :n_frames].T.reshape(n_frames, nx, ny)

    fig, ax = plt.subplots(figsize=(ny / 6, nx / 10))
    im = ax.imshow(field_2d[0], cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.02)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("seasonal anomaly")

    def animate(i):
        im.set_data(field_2d[i])
        ax.set_title(f"Seasonality – t={i+1}")
        return (im,)

    ani = animation.FuncAnimation(fig, animate, frames=n_frames, blit=False)

    ani.save(path, writer="pillow", fps=10)
    plt.close()


def hovmoller_seasonality(seasonal_field, nx, ny, n_steps=100, path="seasonality_hovmoller.png"):
    """
    Hovmöller-style plot that compresses the whole seasonal signal into one figure.

    Parameters
    ----------
    seasonal_field : ndarray  shape (L, T)
    nx, ny         : grid dimensions  (L must equal nx × ny)
    n_steps        : how many time-steps to include on the x-axis
    path           : PNG filename to save
    """
    import matplotlib.pyplot as plt
    import numpy as np

    L, T = seasonal_field.shape
    n_steps = min(n_steps, T)

    # reshape to [time, lat, lon]
    field_3d = seasonal_field.T.reshape(T, nx, ny)

    # zonal mean (average over lon axis)
    hov = field_3d[:n_steps].mean(axis=2)  # shape (n_steps, nx)

    # transpose so y-axis is latitude, x-axis is time
    hov = hov.T  # shape (nx, n_steps)

    vmax = np.abs(hov).max()  # symmetric colour scale
    vmin = -vmax

    plt.figure(figsize=(8, 4))
    im = plt.imshow(hov, aspect="auto", origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)  # south pole at bottom
    plt.colorbar(im, label="seasonal anomaly")
    plt.xlabel("time-step")
    plt.ylabel("latitude index")
    plt.title("Seasonality • zonal mean (first {:d} steps)".format(n_steps))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("saved:", path)
