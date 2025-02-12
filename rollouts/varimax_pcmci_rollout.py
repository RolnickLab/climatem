import json
from pathlib import Path

import numpy as np
import torch
from numpy import asarray, diag, dot, eye, sum
from numpy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tigramite.independence_tests.parcorr import ParCorr
from tqdm.auto import tqdm

from climatem.data_loader.causal_datamodule import CausalClimateDataModule

# Why not gpu here?
# from climatem.data_loader.to_delete_climate_data_loader_explore_ensembles.py import CausalClimateDataModule
# from climatem.climate_data_loader_explore_ensembles import CausalClimateDataModule
from climatem.model.tsdcd_latent import LatentTSDCD


def compute_next_step(initial_5, varimax_rotation, n_modes, adj_matrix_inferred, regressions, tau=5):
    latent_test_data = pca_model.transform(initial_5)
    varimaxpcs_test, _ = varimax(latent_test_data, R=varimax_rotation)

    estimated_Ys = np.zeros((n_modes))
    for mode in range(n_modes):
        varimaxpcs_X_reg = (varimaxpcs_test * adj_matrix_inferred[:, mode, :]).T.reshape((tau * n_modes, 1)).T
        reg = regressions[mode]
        mean_regression = reg.predict(varimaxpcs_X_reg)[0]
        # Sample N particles from Gaussian(mean_regression, std_regression)
        estimated_Ys[mode] = mean_regression

    inverse_varimax_Y0 = dot(estimated_Ys, np.linalg.pinv(varimax_rotation))
    observations_Y0 = pca_model.inverse_transform(inverse_varimax_Y0)

    return estimated_Ys, observations_Y0


def compute_next_step_samples(
    initial_5, varimax_rotation, n_modes, adj_matrix_inferred, regressions, stds_ar, tau=5, N_samples=10
):
    latent_test_data = pca_model.transform(initial_5)
    varimaxpcs_test, _ = varimax(latent_test_data, R=varimax_rotation)

    estimated_Ys = np.zeros((n_modes))
    for mode in range(n_modes):
        varimaxpcs_X_reg = (varimaxpcs_test * adj_matrix_inferred[:, mode, :]).T.reshape((tau * n_modes, 1)).T
        reg = regressions[mode]
        mean_regression = reg.predict(varimaxpcs_X_reg)[0]
        # Sample N particles from Gaussian(mean_regression, std_regression)
        estimated_Ys[mode] = mean_regression

    # particles N_samples*n_modes
    particles = np.random.multivariate_normal(estimated_Ys, np.diag(stds_ar), size=N_samples)
    #     particles = estimated_Ys[None]
    log_score = -((particles - estimated_Ys[None]) ** 2) / ((np.array(stds_ar)[None]) ** 2 * 2 * np.pi)
    inverse_varimax_particles = dot(particles, np.linalg.pinv(varimax_rotation))
    # N samples * lon*lat
    observations_particles = pca_model.inverse_transform(inverse_varimax_particles)

    return particles, log_score, observations_particles


def logscore_the_samples_for_spatial_spectra_bayesian(
    y_true_fft_mean,
    y_true_fft_std,
    y_pred_samples,
    coords: np.ndarray,
    sigma: float = 1.0,
    num_particles: int = 100,
    batch_size: int = 64,
    distribution_spatial_spectra: str = "laplace",
):
    """
    Calculate the spatial spectra of the true values and the predicted values, and then calculate a score between them.
    This is a measure of how well the model is predicting the spatial spectra of the true values.

    Args:
        true_values: torch.Tensor, observed values in a batch
        y_pred: torch.Tensor, a selection of predicted values
        num_particles: int, the number of samples that have been taken from the model
    """

    fft_pred = np.fft.rfft2(y_pred_samples, axes=(-1,))

    spatial_spectra_score = np.abs((fft_pred - y_true_fft_mean[None]) / (y_true_fft_std[None]))

    #     print("Spatial spectra score shape before summing:", spatial_spectra_score.shape)

    # take the mean of the spatial spectra score across the variables and the wavenumbers, the final 2 axes
    spatial_spectra_score = -np.sum(spatial_spectra_score, axis=1)

    #     print("The spatial spectra score shape should be (num_particles):", spatial_spectra_score.shape)
    # score = ...
    return spatial_spectra_score


def particle_filter_weighting_bayesian(
    x,
    #     y,
    y_true_fft_mean,
    y_true_fft_std,
    varimax_rotation,
    n_modes,
    adj_matrix_inferred,
    regressions,
    stds_ar,
    tau=5,
    num_particles: int = 500,
    num_particles_per_particle: int = 20,
    timesteps: int = 600,
    score: str = "variance",
    save_dir: str = None,
    save_name: str = None,
    batch_size: int = 16,
):
    """
    Implement a particle filter to make a set of autoregressive predictions, where each created sample is evaluated by
    some score, and we do a particle filter to select only best samples to continue the autoregressive rollout.

    We need to pass the directory to save stuff to, and the stem of the filenames...
    TODO: REMOVE FOR LOOP OVER BATCH - torch/model can deal with the additional row?

    TODO: Code is quite confusing because here x is latent and z is reconstruction + y is fixed obs corresponding to FFT
    """

    predicted_trajectory = np.zeros((timesteps, 6250))

    #     print('Initial number of particles:', num_particles)

    for _ in tqdm(range(timesteps)):
        if _ % 10 == 0:
            print(f"Filtering timestep {_}")

        # Prediction
        # make all the new predictions, taking samples from the latents

        if _ == 0:
            #             print("This is the first timestep, so we are going to generate samples from the initial latents.")
            particles, log_scores, observations_particles = compute_next_step_samples(
                x,
                varimax_rotation,
                n_modes,
                adj_matrix_inferred,
                regressions,
                stds_ar,
                tau=tau,
                N_samples=num_particles * num_particles_per_particle,
            )
            log_scores = log_scores.sum(1)
        #             print(f"particles.shape {particles.shape}")

        else:
            #             print("Not the first timestep, so generating samples using initial particles.")

            for i in range(num_particles):
                #                 print(f"Generating mean sample for particle {i}")
                next_particles, next_log_scores, next_observations_particles = compute_next_step_samples(
                    x[i],
                    varimax_rotation,
                    n_modes,
                    adj_matrix_inferred,
                    regressions,
                    stds_ar,
                    tau=tau,
                    N_samples=num_particles_per_particle,
                )
                next_log_scores = next_log_scores.sum(1)
                if i == 0:
                    particles = next_particles
                    log_scores = next_log_scores
                    observations_particles = next_observations_particles
                else:
                    particles = np.concatenate([particles, next_particles], axis=0)
                    log_scores = np.concatenate([log_scores, next_log_scores], axis=0)
                    observations_particles = np.concatenate(
                        [observations_particles, next_observations_particles], axis=0
                    )

        #         print(f"particles.shape {particles.shape}")
        # Then fft_true shape after repeating: torch.Size([K*L, batch size, 1, 3126]
        scores_spatial_spectra = logscore_the_samples_for_spatial_spectra_bayesian(
            y_true_fft_mean,
            y_true_fft_std,
            observations_particles,
            coords=coordinates,
            num_particles=num_particles * num_particles_per_particle,
            batch_size=batch_size,
        )
        new_weights = log_scores + scores_spatial_spectra
        # normalise the weights along the first dimension

        max_weight = np.max(new_weights, axis=0)
        new_weights -= max_weight
        new_weights = np.exp(new_weights)

        new_weights = new_weights / np.sum(new_weights, axis=0)
        new_weights[new_weights < 1e-16] = 0
        new_weights = new_weights / np.sum(new_weights, axis=0)

        resampled_indices_array = np.random.choice(
            np.arange(num_particles * num_particles_per_particle), p=new_weights, size=num_particles, replace=True
        )

        particles = particles[resampled_indices_array]

        if _ == 0:
            x = x[None].repeat(num_particles, axis=0)
        #             print("Shape of x after repeating, in the first timestep:", x.shape)

        x = x[:, 1:, :]

        # now we just need to unsqueeze the selected samples, so that we can concatenate them to x
        inverse_varimax_particles = dot(particles, np.linalg.pinv(varimax_rotation))
        # N samples * lon*lat
        observations_particles = pca_model.inverse_transform(inverse_varimax_particles)

        # then we need to append the selected samples to x, along the right axis
        # Here shouldn't it be the unused samples fromxs???
        x = np.concatenate([x, observations_particles[:, None]], axis=1)
        # then we are going back to the top of the loop
        predicted_trajectory[_] = observations_particles

    np.save(save_dir / save_name, predicted_trajectory)
    return x


def varimax(Phi, R=None, gamma=1, q=20, tol=1e-6):
    if R is None:
        p, k = Phi.shape
        R = eye(k)
        d = 0
        for i in range(q):
            d_old = d
            Lambda = dot(Phi, R)
            u, s, vh = svd(
                dot(Phi.T, asarray(Lambda) ** 3 - (gamma / p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda)))))
            )
            R = dot(u, vh)
            d = sum(s)
            if d / d_old < tol:
                break
    return dot(Phi, R), R


def calculate_fft_mean_std_across_all_noresm(datamodule, number_of_batches: int = 18):

    # Start again at the beginning of the dataloader.
    train_dataloader = iter(datamodule.train_dataloader())

    # iterate through the data and append all the y values together
    y_all = []
    for i in range(number_of_batches):
        _, y_whole_dataloader = next(train_dataloader)
        y_all.append(y_whole_dataloader[:, 0])
    y_all = torch.cat(y_all, dim=0)
    y_all = torch.nan_to_num(y_all)

    # make sure we reset the dataloader
    train_dataloader = iter(datamodule.train_dataloader())

    y_true_fft_data = torch.abs(torch.fft.rfft(y_all[:, :, :], dim=2))

    # calculate the mean and std of the fft of the true data across all the data
    y_true_fft_mean = y_true_fft_data.mean(dim=0)
    y_true_fft_std = y_true_fft_data.std(dim=0)

    return y_true_fft_mean, y_true_fft_std


if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")

if __name__ == "__main__":

    print("Loading PiControl data")
    with open("$HOME/scripts/configs/data_loading.json", "r") as f:
        hp = json.load(f)
    coordinates = np.loadtxt("$HOME/vertex_lonlat_mapping.txt")
    hp["config_exp_path"] = "$HOME/scripts/configs/params.json"

    config_fname = hp["config_exp_path"]
    with open(config_fname) as f:
        data_params = json.load(f)

    datamodule = CausalClimateDataModule(**data_params)  # ...
    datamodule.setup()

    train_dataloader = iter(datamodule.train_dataloader())

    n_batches = 18  # number of data loader batches

    for i in range(n_batches):  # , batch in enumerate(train_dataloader):
        if i == 0:
            x, y = next(train_dataloader)
            x = torch.nan_to_num(x)
            y = torch.nan_to_num(y)
            y = y[:, 0]
        if i > 0:
            x_bis, y_bis = next(train_dataloader)
            x_bis = torch.nan_to_num(x_bis)
            y_bis = torch.nan_to_num(y_bis)
            y_bis = y_bis[:, 0]
            x = torch.cat((x, x_bis))
            y = torch.cat((y, y_bis))

    y_true_fft_mean, y_true_fft_std = calculate_fft_mean_std_across_all_noresm(datamodule, number_of_batches=18)

    y_true_fft_mean = y_true_fft_mean.cpu().numpy()[0]
    y_true_fft_std = y_true_fft_std.cpu().numpy()[0]

    y = y.detach().cpu().numpy()

    # Fit PCA + apply varimax to training data
    train_data = y[:, 0].T

    parcorr = ParCorr(significance="analytic")

    tau = 5
    n_modes = 90
    var_names = []
    for k in range(n_modes):
        var_names.append(rf"$X^{k}$")

    sparsity_value = 0.05
    n_desired_links = sparsity_value * n_modes * n_modes * tau

    pca_model = PCA(n_modes)
    pca_model.fit(train_data.T)
    latent_train_data = pca_model.transform(train_data.T)
    varimaxpcs_train, varimax_rotation = varimax(latent_train_data)
    adj_matrix_inferred = np.load("pcmci_inferred_graph.npy")

    length_training = varimaxpcs_train.shape[1] - tau

    indices = np.zeros((tau, length_training))
    for k in range(tau):
        indices[k] = np.arange(k, length_training + k)
    varimaxpcs_X = varimaxpcs_train[indices.astype("int")]  # X ihere is for the regression X
    varimaxpcs_Y = varimaxpcs_train[np.arange(tau, length_training + tau)]

    varimaxpcs_X = varimaxpcs_X.transpose(1, 2, 0)
    varimaxpcs_X.shape

    regressions = []
    stds_ar = []
    for mode in range(n_modes):
        varimaxpcs_X_reg = (varimaxpcs_X * adj_matrix_inferred[:, mode, :].T[None]).reshape((length_training, -1))
        reg = LinearRegression().fit(varimaxpcs_X_reg, varimaxpcs_Y[:, mode])
        regressions.append(reg)
        # Get stds for bayesian filter
        estimated_Y = reg.predict(varimaxpcs_X_reg)
        stds_ar.append((estimated_Y - varimaxpcs_Y[:, mode]).std())

    print("Loading SSP data")
    # This json file needs to correspond to the test scenario
    with open("$HOME/scripts/configs/params_data_loading_ssp370.json", "r") as f:
        hp = json.load(f)

    hp["config_exp_path"] = "$HOME/scripts/configs/params.json"

    config_fname = hp["config_exp_path"]
    with open(config_fname) as f:
        data_params = json.load(f)

    datamodule = CausalClimateDataModule(**data_params)  # ...
    datamodule.setup()

    train_dataloader = iter(datamodule.train_dataloader())

    n_batches = 2
    for i in range(n_batches):
        print(i)
        if i == 0:
            x, y = next(train_dataloader)
            x = torch.nan_to_num(x)
            y = torch.nan_to_num(y)
            y = y[:, 0]
        if i > 0:
            x_bis, y_bis = next(train_dataloader)
            x_bis = torch.nan_to_num(x_bis)
            y_bis = torch.nan_to_num(y_bis)
            y_bis = y_bis[:, 0]
            x = torch.cat((x, x_bis))
            y = torch.cat((y, y_bis))

    y_true_fft_mean, y_true_fft_std = calculate_fft_mean_std_across_all_noresm(datamodule, number_of_batches=2)

    y_true_fft_mean = y_true_fft_mean.cpu().numpy()[0]
    y_true_fft_std = y_true_fft_std.cpu().numpy()[0]

    x_test, y_test = next(train_dataloader)
    x_test = torch.nan_to_num(x_test)
    y_test = torch.nan_to_num(y_test)
    y_test = y_test[:, 0]
    x_test = x_test[:, :, 0]
    x_test = x_test.cpu().numpy()
    y_test = y_test.cpu().numpy()

    fname_dir_results = "predicted_trajectories_pcmci_SSP"
    save_dir = Path(f"$DATA/{fname_dir_results}")

    n_initial_conditions = 50
    timestep_total = 600
    arr_cond_indices = np.arange(256)
    all_initial_cond = np.random.choice(arr_cond_indices, n_initial_conditions, replace=False)

    for j, initcond in enumerate(all_initial_cond):
        print(f"Initial condition {j}")
        inferred_observations = particle_filter_weighting_bayesian(
            x_test[initcond],
            y_true_fft_mean,
            y_true_fft_std,
            varimax_rotation,
            n_modes,
            adj_matrix_inferred,
            regressions,
            stds_ar,
            tau=tau,
            num_particles=1,
            num_particles_per_particle=1000,
            timesteps=timestep_total,
            save_dir=save_dir,
            save_name=f"trajectory_init_cond_{initcond}.npy",
        )
