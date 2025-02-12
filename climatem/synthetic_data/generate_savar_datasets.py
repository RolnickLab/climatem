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


def generate_save_savar_data(
    save_dir_path,
    name,
    time_len=10_000,
    comp_size=10,
    noise_val=0.2,
    n_per_col=2,  # Number of components N = n_per_col**2
    difficulty="easy",
    seasonality=False,
    overlap=False,
    is_forced=False,
    plotting=True,
):

    # Setup spatial weights of underlying processes
    ny = nx = n_per_col * comp_size
    N = n_per_col**2  # Number of components
    noise_weights = np.zeros((N, nx, ny))
    modes_weights = np.zeros((N, nx, ny))

    if overlap:
        raise ValueError("SAVAR data with overlapping modes not implemented yet")

    # Specify the path where you want to save the data
    npy_name = f"{name}.npy"
    save_path = save_dir_path / npy_name

    # Create modes weights
    for k in range(n_per_col):
        for j in range(n_per_col):
            modes_weights[
                k * n_per_col + j, k * comp_size : (k + 1) * comp_size, j * comp_size : (j + 1) * comp_size
            ] = create_random_mode((comp_size, comp_size), random=True)
    for k in range(n_per_col):
        for j in range(n_per_col):
            noise_weights[
                k * n_per_col + j, k * comp_size : (k + 1) * comp_size, j * comp_size : (j + 1) * comp_size
            ] = create_random_mode((comp_size, comp_size), random=True)

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
        raise ValueError("SAVAR data with forcings not implemented yet")
        f_1, f_2, f_time_1, f_time_2 = 1, 2, 4000, 8000  # turn off forcing by setting the time to the last time step
        w_f = modes_weights
        # A very simple method for adding a focring term (bias on the mean of the noise term)
        forcing_dict = {
            "w_f": w_f,  # Shape of the mode of the forcing
            "f_1": f_1,  # Value of the forcing at period_1
            "f_2": f_2,  # Value of the forcing at period_2
            "f_time_1": f_time_1,  # The period one goes from t=0  to t=f_time_1
            "f_time_2": f_time_2,  # The period two goes from t= f_time_2 to the end. Between the two periods, the forcing is risen linearly
            "time_len": time_len,
        }
    if seasonality:
        raise ValueError("SAVAR data with seasonality not implemented yet")
        # We could introduce seasonality if we would wish
        # season_dict = {"amplitude": 0.08, "period": 12}

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
        # "f_1": f_1,
        # "f_2": f_2,
        # "f_time_1": f_time_1,
        # "f_time_2": f_time_2,
        # "time_len": time_len,
        # "season_dict": season_dict,
        # "seasonality" : True,
    }

    # Specify the path to save the parameters
    param_names = f"{name}_parameters.npy"
    params_path = save_dir_path / param_names
    # Save the dictionary of parameters to a .npy file
    np.save(params_path, parameters)

    param_names = f"{name}_parameters.csv"
    params_path = save_dir_path / param_names
    save_parameters_to_csv(params_path, parameters)
    param_names = f"{name}_links_coeffs.csv"
    params_path = save_dir_path / param_names
    save_links_coeffs_to_csv(params_path, parameters["links_coeffs"])
    param_names = f"{name}_mode_weights.npy"
    params_path = save_dir_path / param_names
    np.save(params_path, modes_weights)

    # Create a copy of the parameters to modify
    parameters_copy = parameters.copy()
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
            # season_dict=season_dict, #turn off by commenting out
            # forcing_dict=forcing_dict #turn off by commenting out
        )
    else:
        savar_model = SAVAR(
            links_coeffs=links_coeffs,
            time_length=time_len,
            mode_weights=modes_weights,
            noise_strength=noise_val,
            # season_dict=season_dict, #turn off by commenting out
            forcing_dict=forcing_dict,  # turn off by commenting out
        )

    savar_model.generate_data()  # Remember to generate data, otherwise the data field will be empty
    np.save(save_path, savar_model.data_field)
    print(f"{name} DONE!")
    return savar_model.data_field
