# This is a script to run a particle filtering rollout for a model.
# We can choose the number of timesteps, and what we want to filter for.
# Be careful with the number of batches we use for calculating the true data spectra.

# hack to go a couple of directories up if we need to import from python files in some parent directory.

import os
import sys
from pathlib import Path

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

import json

import numpy as np
import torch

from climatem.climate_data_loader_explore_ensembles import CausalClimateDataModule
from climatem.model.tsdcd_latent_explore import LatentTSDCD

# import climatem.climate_dataset_explore_ensembles as climate_dataset

# Now I want to apply my particle filter to this rollout.
# Here we have scoring functions and a function to do the particle filtering.

# if we are doing ssps, and we want to only look at the last 30 years:
final_30_years_of_ssps = False

if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")


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

    fft_pred = torch.abs(torch.fft.rfft(y_pred_samples[:, :, :], dim=3))

    # extend fft_true so it is the same value but extended to the same shape as fft_pred
    fft_true = y_true_fft_mean.repeat(num_particles, batch_size, 1, 1)
    fft_true_std = y_true_fft_std.repeat(num_particles, batch_size, 1, 1)

    if fft_pred.dim() == fft_true.dim() + 1:
        print("I am flattening the preds here.")
        fft_pred = torch.flatten(fft_pred, start_dim=0, end_dim=1)

    assert fft_true.shape == fft_pred.shape
    assert fft_true_std.shape == fft_pred.shape

    # calculate the difference between the true and predicted spatial spectra
    # TODO: sigma should be a vector for every wavenumber since the spectrum has different values,
    # and I should calculate this directly for NorESM for each wavenumber - DONE.

    if distribution_spatial_spectra == "laplace":
        spatial_spectra_score = torch.abs((fft_pred - fft_true) / (fft_true_std))
    elif distribution_spatial_spectra == "gaussian":
        spatial_spectra_score = ((fft_pred - fft_true) ** 2) / (2 * fft_true_std**2)

    print("Spatial spectra score shape before summing:", spatial_spectra_score.shape)

    # take the mean of the spatial spectra score across the variables and the wavenumbers, the final 2 axes
    spatial_spectra_score = -torch.sum(spatial_spectra_score, dim=(2, 3))
    # then normalise all the values of spatial_spectra_score by the maximum value
    # print("Spatial spectra score before normalising:", spatial_spectra_score)

    # Do normalisation and 1 - if we want the score to be increasing
    # spatial_spectra_score = spatial_spectra_score / torch.max(spatial_spectra_score)
    # print("Spatial spectra score normalised:", spatial_spectra_score)

    # the do 1 - score to give the score to be increasing...
    # spatial_spectra_score = 1 - spatial_spectra_score
    # print("Spatial spectra score doing 1 - score:", spatial_spectra_score)

    print("The spatial spectra score shape should be (num_particles, num_batch_size):", spatial_spectra_score.shape)
    # score = ...
    return spatial_spectra_score


def score_the_samples_for_spatial_spectra(
    y_true, y_pred_samples, coords: np.ndarray, num_particles: int = 100, mid_latitudes: bool = False
):
    """
    Calculate the spatial spectra of the true values and the predicted values, and then calculate a score between them.
    This is a measure of how well the model is predicting the spatial spectra of the true values.

    Args:
        true_values: torch.Tensor, observed values in a batch
        y_pred: torch.Tensor, a selection of predicted values
        num_particles: int, the number of samples that have been taken from the model
    """

    if mid_latitudes:
        print("Doing spectral regularisation for only the mid-latitudes")
        # isolate just the latitude values
        lat_values = coords[:, 1]
        # check these are the right values
        # get the indices of the points that are in the extratropics
        extratropics_indices = np.where(
            (lat_values > -65) & (lat_values < -25) | (lat_values > 25) & (lat_values < 65)
        )[0]
        # select just the coordinates of the extratropics for y_true, y_recons, and y_pred
        # print('Shapes of y_true and y_pred_samples before selecting the extratropics:', y_true.shape, y_pred_samples.shape)
        y_true = y_true[:, :, extratropics_indices]
        y_pred_samples = y_pred_samples[:, :, :, extratropics_indices]

    # calculate the average spatial spectra of the true values, averaging across the batch
    # print("y_true shape:", y_true.shape)
    # fft_true = torch.mean(torch.abs(torch.fft.rfft(y_true[:, :, :], dim=2)), dim=0)
    fft_true = torch.abs(torch.fft.rfft(y_true[:, :, :], dim=2))
    # calculate the average spatial spectra of the individual predicted fields - I think this below is wrong
    # print("y_pred shape:", y_pred_samples.shape)
    # fft_pred = torch.mean(torch.abs(torch.fft.rfft(y_pred_samples[:, :, :], dim=3)), dim=1)
    fft_pred = torch.abs(torch.fft.rfft(y_pred_samples[:, :, :], dim=3))

    # extend fft_true so it is the same value but extended to the same shape as fft_pred
    fft_true = fft_true.repeat(num_particles, 1, 1, 1)

    # assert that the first two elements of fft_true are the same
    # assert torch.allclose(fft_true[0, :, :], fft_true[1, :, :])

    # print("fft_true shape after repeating:", fft_true.shape)
    # print("fft_pred shape:", fft_pred.shape)

    assert fft_true.shape == fft_pred.shape

    # calculate the difference between the true and predicted spatial spectra
    spatial_spectra_score = torch.abs(fft_pred - fft_true)

    # take the mean of the spatial spectra score across the variables and the wavenumbers, the final 2 axes
    spatial_spectra_score = torch.mean(spatial_spectra_score, dim=(2, 3))

    return spatial_spectra_score


def particle_filter_weighting_bayesian(
    x,
    y,
    y_true_fft_mean,
    y_true_fft_std,
    num_particles: int = 100,
    num_particles_per_particle: int = 10,
    timesteps: int = 120,
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

    print("Initial number of particles:", num_particles)

    for _ in range(timesteps):
        print(f"Filtering timestep {_}")

        # Prediction
        # make all the new predictions, taking samples from the latents

        if _ == 0:
            print("This is the first timestep, so I am going to generate samples from the initial latents.")
            if score == "log_bayesian":
                print(f"x shape {x.shape}")
                print(f"y shape {y.shape}")
                unused_samples_from_xs, samples_from_zs, y, logscore_samples_fromzs = (
                    model.predict_sample_bayesianfiltering(
                        x, y, num_particles * num_particles_per_particle, with_zs_logprob=True
                    )
                )
                logscore_samples_fromzs = torch.sum(logscore_samples_fromzs, -1).squeeze(2)
                print(f"unused_samples_from_xs shape {unused_samples_from_xs.shape}")
                print(f"samples_from_zs shape {samples_from_zs.shape}")
                print(f"logscore_samples_fromzs shape {logscore_samples_fromzs.shape}")
            else:
                unused_samples_from_xs, samples_from_zs, y = model.predict_sample_bayesianfiltering(
                    x, y, num_particles * num_particles_per_particle, with_zs_logprob=False
                )

        else:
            print("Not the first timestep, so generating samples using initial particles.")
            # px_mu, y, z, pz_mu, pz_std = model.predict(x, y, num_particles)
            # note, here I think x is no. of samples - dimensional
            # REMOVE THIS FOR LOOP IF POSSIBLE
            for i in range(num_particles):
                # print(f"Generating mean sample for particle {i}")
                # px_mu, y, z, pz_mu, pz_std = model.predict(x[:, i, :, :], y[i, :, :])

                # New code
                # Here for each particle at time t predict num_particles_per_particle at time t+1
                if score == "log_bayesian":
                    unused_samples_from_xs, next_sample_from_zs, y, next_logscore_samples_fromzs = (
                        model.predict_sample_bayesianfiltering(
                            x[i, :, :, :, :], y, num_particles_per_particle, with_zs_logprob=True
                        )
                    )
                    # clip the next_sample_from_zs to avoid numerical instability
                    print("Clipping the next_samples_from_zs to avoid numerical instability.")
                    next_sample_from_zs = torch.clip(next_sample_from_zs, -1e10, 1e10)

                    next_logscore_samples_fromzs = torch.sum(next_logscore_samples_fromzs, -1).squeeze()
                #                     print("What should be the correct shape??")
                #                     print(f"shape of new samples {next_sample_from_zs.shape}")
                #                     print(f"{sfug}")
                else:
                    next_sample_from_zs, y, unused_z, unused_pz_mu, unused_pz_std = model.predict(x[i, :, :, :, :], y)
                #                     print("Here is the correct shape??")
                #                     print(f"shape of new samples {next_sample_from_zs.shape}")
                #                     print(f"{sfug}")
                if i == 0:
                    samples_from_zs = next_sample_from_zs.unsqueeze(0)
                    logscore_samples_fromzs = next_logscore_samples_fromzs.unsqueeze(0)
                else:
                    samples_from_zs = torch.cat([samples_from_zs, next_sample_from_zs.unsqueeze(0)], dim=0)
                    logscore_samples_fromzs = torch.cat(
                        [logscore_samples_fromzs, next_logscore_samples_fromzs.unsqueeze(0)], dim=0
                    )
            # samples_from_zs, y, unused_z, unused_pz_mu, unused_pz_std = model.predict(x, y)

        # then calculate the score of each of the samples
        # Update the weights, where we want the weights to increase as the score improves

        if score == "spatial_spectra":
            new_weights = score_the_samples_for_spatial_spectra(
                y,
                samples_from_zs,
                coords=coordinates,
                num_particles=num_particles * num_particles_per_particle,
                mid_latitudes=True,
            )
        elif score == "log_bayesian":
            print(f"logscore_samples_fromzs shape {logscore_samples_fromzs.shape}")
            # This is [K, L, batch size]
            print(f"y shape {y.shape}")
            if _ > 0:
                # In correct dimension?? should be
                logscore_samples_fromzs = torch.flatten(logscore_samples_fromzs, start_dim=0, end_dim=1)
                samples_from_zs = torch.flatten(samples_from_zs, start_dim=0, end_dim=1)
            print(f"samples_from_zs shape {samples_from_zs.shape}")
            # This is [K*L, batch size, 1, 6250]--> Is this expected?
            # Then fft_true shape after repeating: torch.Size([K*L, batch size, 1, 3126]
            scores_spatial_spectra = logscore_the_samples_for_spatial_spectra_bayesian(
                y_true_fft_mean,
                y_true_fft_std,
                samples_from_zs,
                coords=coordinates,
                num_particles=num_particles * num_particles_per_particle,
                batch_size=batch_size,
            )
            print(f"spatial_spectra shape {scores_spatial_spectra.shape}")
            # This is [K*L, batch size]

            print("********************************************************************************")
            print("What are the top 10 scores for the spatial spectra?", torch.topk(scores_spatial_spectra, 10))
            print(
                "What are the bottom 10 scores for the spatial spectra?",
                torch.topk(scores_spatial_spectra, 10, largest=False),
            )
            print("What is the median of the scores for the spatial spectra?", torch.median(scores_spatial_spectra))
            print("What is the mean of the scores for the spatial spectra?", torch.mean(scores_spatial_spectra))
            print("What is the std of the scores for the spatial spectra?", torch.std(scores_spatial_spectra))

            new_weights = logscore_samples_fromzs + scores_spatial_spectra
        #             new_weights = torch.exp(new_weights) # Here we might be able to sample directly from the log probabilities in torch to avoid taking the exp
        else:
            raise ValueError("Score must be either variance or spatial_spectra")

        print("New log weights are higher is better if log_bayesian otherwise lower...")
        print("Shape of new weights:", new_weights.shape)

        #         print('Minimum of the new weights, along the first dimension:', torch.min(new_weights, dim=0))
        #         print('Maximum of the new weights, along the 0th dimension:', torch.max(new_weights, dim=0))
        print("What is the shape of the min calculated above:", torch.min(new_weights, dim=0).values.shape)

        # normalise the weights along the first dimension

        # TODO below this!!
        max_weight = torch.max(new_weights, dim=0)
        if score != "log_bayesian":
            min_weight = torch.min(new_weights, dim=0)
            # normalise along the first dimension
            # normalised_weights = (new_weights - min_weight.values) / (max_weight.values - min_weight.values)
        else:
            # might get overflows here - might need to clip...for torch.exp
            new_weights = torch.exp(new_weights - max_weight.values)
            min_weight = torch.min(new_weights, dim=0)
            max_weight = torch.max(new_weights, dim=0)
            normalised_weights = (new_weights - min_weight.values) / (max_weight.values - min_weight.values)
        # Do we need to normalize if log scores

        print("shape of normalised weights:", normalised_weights.shape)
        # assert that the sum of the normalised weights is 1 for each row
        #         print("Sum of the normalised weights:", torch.sum(normalised_weights, dim=0))

        # new_weights = 1 - normalised_weights  # Here no need to invert! we're already in prob space

        # also clip here!
        new_weights = torch.clamp(new_weights, min=1e-6, max=1.0)
        new_weights = new_weights / torch.sum(new_weights, dim=0)

        print("Shape of the new_weights after normalising:", new_weights.shape)
        #         print("Sum of the new normalised weights:", torch.sum(new_weights, dim=0))

        # clip the new_weights to avoid numerical instability
        new_weights = torch.clamp(new_weights, min=1e-6, max=1.0)

        print("What is the shape of the new weights after clipping?", new_weights.shape)
        print("Are all the values 0?", torch.all(new_weights == 0))
        print("What is the minimum value of the new weights?", torch.min(new_weights))
        print("What is the maximum value of the new weights?", torch.max(new_weights))
        print("Are there any nans in the new weights?", torch.any(torch.isnan(new_weights)))
        print("Are there any infs in the new weights?", torch.any(torch.isinf(new_weights)))

        # replace any nans with 1e-8
        new_weights[torch.isnan(new_weights)] = 1e-6

        print("Are there any nans in the new weights after replacing?", torch.any(torch.isnan(new_weights)))
        print("What is the new minimum value of the new weights?", torch.min(new_weights))
        print("What is the new maximum value of the new weights?", torch.max(new_weights))

        # REMOVE THIS FOR LOOP IF POSSIBLE
        for i in range(batch_size):
            resampled_indices = torch.multinomial(new_weights[:, i], num_particles, replacement=True)
            # append these resampled indices to n array so we get an output of shape (5, batch_size)
            if i == 0:
                resampled_indices_array = resampled_indices.unsqueeze(1)
            else:
                resampled_indices_array = torch.cat([resampled_indices_array, resampled_indices.unsqueeze(1)], dim=1)

        # Use list comprehension to collect resampled indices for each column
        # resampled_indices_array2 = torch.stack([torch.multinomial(new_weights[:, i], num_particles, replacement=True) for i in range(256)], dim=1)

        # assert that the two resampled indices are the same
        # assert torch.all(resampled_indices_array == resampled_indices_array2)

        selected_samples = samples_from_zs[resampled_indices_array, torch.arange(batch_size), :, :]
        np.save(os.path.join(save_dir, f"{save_name}_{_}.npy"), selected_samples.detach().cpu().numpy())
        print("Saved the selected samples with name:", f"{save_name}_{_}.npy")

        if _ == 0:
            x = x.repeat(num_particles, 1, 1, 1, 1)
            print("Shape of x after repeating, in the first timestep:", x.shape)

        x = x[:, :, 1:, :, :]

        # now we just need to unsqueeze the selected samples, so that we can concatenate them to x
        selected_samples = selected_samples.unsqueeze(2)

        print("What is the shape of x, just before we concatenate?", x.shape)
        print("What is the shape of the selected samples, just before we concatenate?", selected_samples.shape)

        # then we need to append the selected samples to x, along the right axis
        # Here shouldn't it be the unused samples fromxs???
        x = torch.cat([x, selected_samples], dim=2)

        # then we are going back to the top of the loop

    return selected_samples


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Read the coordinates too...

home_dir_path = Path("/home/mila/s/sebastian.hickman")

local_folder = home_dir_path / "work"
scratch_dir = home_dir_path / "scratch"  # Where large data is stored
results_dir = scratch_dir / "results"
os.makedirs(results_dir, exist_ok=True)
climatem_repo = local_folder / "new_climatem/climatem/climatem"

coordinates_path = climatem_repo / "vertex_lonlat_mapping.txt"
print("coordinates path:", coordinates_path)
coordinates = np.loadtxt(coordinates_path)
coordinates = coordinates[:, 1:]

# NOTE: here saving SSP runs...
results_save_folder = results_dir / "new_climatem_spectral_filtered_100_year"
# Make below updated with variables automatically + simpler
results_save_folder_var = results_save_folder / "logspectraltrain_300particles_ablations"
results_save_folder_var_spectral = (
    results_save_folder_var / "full_model_1crps_1000spec_2000tspec_16batch_filtered_val_high_wavs_fix_penalty"
)

# path to the results directory that I care about
# Now doing for two models, one where we learned a causal graph (taking the final model) and one where we didn't

# local_results_dir = results_dir / "climatem_spectral"
# os.makedirs(local_results_dir, exist_ok=True)

# TODO: These names are bad... the [] and '' make it super annoying + the params should update the name automatically
# name_res_ts_vae = "var_['ts']_scenarios_piControl_tau_5_z_90_lr_0.001_spreg_0.743706_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_20_tempspcoef_2000"
# name_res_ts_novae = "var_[ts]_scenarios_piControl_tau_5_z_90_lr_0.001_spreg_0.743706_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_20_tempspcoef_2000"

# load a full model
# local_results_dir = results_dir / "new_climatem_spectral"
# name_res_ts_vae = "var_['ts']_scenarios_piControl_tau_5_z_90_lr_0.001_spreg_0.1_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_20_tempspcoef_2000"

# local_results_dir = results_dir / "new_climatem_spectral"
# local_results_dir = results_dir / "new_climatem_spectral_high_wavs"
local_results_dir = results_dir / "new_climatem_spectral_high_wavs_fix_penalty"

# name_res_ts_vae = "var_['ts']_scenarios_piControl_nonlinear_True_tau_5_z_90_lr_0.001_spreg_0.1_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_50_tempspcoef_2000"
# name_res_ts_vae = "var_['ts']_scenarios_piControl_nonlinear_True_tau_5_z_110_lr_0.001_spreg_0.2_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_5000_tempspcoef_20000"
# name_res_ts_vae = "var_['ts']_scenarios_piControl_nonlinear_True_tau_5_z_150_lr_0.001_spreg_0.1_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_5000_tempspcoef_20000"
# name_res_ts_vae = "var_['ts']_scenarios_piControl_nonlinear_True_tau_5_z_90_lr_0.001_spreg_0.1_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_1000_tempspcoef_2000"
# name_res_ts_vae = "var_['ts']_scenarios_piControl_nonlinear_True_tau_5_z_90_lr_0.001_spreg_0.1_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_0_spcoef_1000_tempspcoef_0"

# name_res_ts_vae = "var_['ts']_scenarios_piControl_nonlinear_True_tau_5_z_90_lr_0.001_spreg_0.1_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_1000_tempspcoef_2000"
# name_res_ts_vae = "var_['ts']_scenarios_piControl_nonlinear_True_tau_5_z_90_lr_0.001_spreg_0.1_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_0_tempspcoef_2000"

name_res_ts_vae = "var_['ts']_scenarios_piControl_nonlinear_True_tau_5_z_90_lr_0.001_spreg_0.1_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_1000_tempspcoef_2000"

results_dir_ts_vae = local_results_dir / name_res_ts_vae
os.makedirs(results_dir_ts_vae, exist_ok=True)

with open(results_dir_ts_vae / "params.json", "r") as f:
    hp = json.load(f)

# Let's overwrite some of the hyperparameters to see if we can load in some different ssp data...
# overwrite the config_exp_path here:
# TODO -- update this
hp["config_exp_path"] = (
    "/home/mila/s/sebastian.hickman/work/new_climatem/climatem/scripts/configs/climate_predictions_picontrol_icosa_nonlinear_ensembles_hilatent_all_icosa_1ens.json"
)

# once I have loaded in the state_dict, I can load it into a model
# first I need to define the model architecture

config_fname = hp["config_exp_path"]
with open(config_fname) as f:
    data_params = json.load(f)

datamodule = CausalClimateDataModule(**data_params)  # ...
datamodule.setup()


# y_true_fft_mean, y_true_fft_std = calculate_fft_mean_std_across_all_noresm(datamodule)
y_true_fft_mean, y_true_fft_std = calculate_fft_mean_std_across_all_noresm(datamodule, number_of_batches=18)

print("y_true_fft_mean shape:", y_true_fft_mean.shape)
print("y_true_fft_std shape:", y_true_fft_std.shape)

# getting the training data in place so that I can forecast using this data.
# NOTE: check these dataloaders
train_dataloader = iter(datamodule.val_dataloader())
# val_dataloader = iter(datamodule.val_dataloader())
x, y = next(train_dataloader)

if final_30_years_of_ssps:
    print("Taking the final 30 years of the SSP data, ~ 2070-2100")
    x, y = next(train_dataloader)
    x, y = next(train_dataloader)


x = torch.nan_to_num(x)
y = torch.nan_to_num(y)
y = y[:, 0]
z = None

# move all this to GPU
x = x.to(device)
y = y.to(device)
# z = z.to(device)

print("Where is the data?", x.device, y.device)

# some little numbers that I am going to need later:
d = x.shape[2]
num_input = d * hp["tau"] * (hp["tau_neigh"] * 2 + 1)

# Instantiate a model here with the hyperparameters that we have loaded in.
model = LatentTSDCD(
    num_layers=hp["num_layers"],
    num_hidden=hp["num_hidden"],
    num_input=num_input,
    num_output=2,
    num_layers_mixing=hp["num_layers_mixing"],
    num_hidden_mixing=hp["num_hidden_mixing"],
    coeff_kl=hp["coeff_kl"],
    d=d,
    distr_z0="gaussian",
    distr_encoder="gaussian",
    distr_transition="gaussian",
    distr_decoder="gaussian",
    d_x=hp["d_x"],
    d_z=hp["d_z"],
    tau=hp["tau"],
    instantaneous=hp["instantaneous"],
    nonlinear_mixing=hp["nonlinear_mixing"],
    hard_gumbel=hp["hard_gumbel"],
    no_gt=hp["no_gt"],
    debug_gt_graph=hp["debug_gt_graph"],
    debug_gt_z=hp["debug_gt_z"],
    debug_gt_w=hp["debug_gt_w"],
    # gt_w=data_loader['gt_w'],
    # gt_graph=data_loader['gt_graph'],
    tied_w=hp["tied_w"],
    # also
    fixed=hp["fixed"],
    fixed_output_fraction=hp["fixed_output_fraction"],
)

# Here we load a final model, when we do learn the causal graph. Make sure  it is on GPU:
# state_dict_vae_final = torch.load(results_dir_ts_vae / "model.pth", map_location=None)

# if we need to load the model that has been trained on 2 GPUs, I think we need to do this:
# state_dict_vae_final = torch.load(results_dir_ts_vae / "model.pth", map_location=torch.device("cuda:0"))

state_dict_vae_final = torch.load(
    results_dir_ts_vae / "best_model_for_average_spectra.pth", map_location=torch.device("cuda:0")
)

model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict_vae_final.items()})

# Move the model to the GPU
model = model.to(device)
print("Where is the model?", next(model.parameters()).device)

# make sure the model is on GPU, and this all runs on GPU
# model = model.cuda()

batch_size = 32


# select 16 random samples from the batch
def sample_from_tensor_reproducibly(tensor1, tensor2, num_samples, seed=5):
    if num_samples > tensor1.shape[0]:
        raise ValueError("Number of samples cannot exceed the tensor's first dimension.")

    torch.manual_seed(seed)  # Set the random seed
    indices = torch.randperm(tensor1.shape[0])[:num_samples]
    return tensor1[indices], tensor2[indices]


# First call with the seed
x_samples, y_samples = sample_from_tensor_reproducibly(x, y, batch_size)

os.makedirs(results_save_folder_var_spectral, exist_ok=True)
np.save(
    results_save_folder_var_spectral / "forpowerspectra_random1_batch_xs_we_start_with.npy",
    x_samples.detach().cpu().numpy(),
)

with torch.no_grad():
    final_picontrol_particles = particle_filter_weighting_bayesian(
        x_samples,
        y_samples,
        y_true_fft_mean,
        y_true_fft_std,
        num_particles=300,
        num_particles_per_particle=10,
        timesteps=1200,
        score="log_bayesian",
        save_dir=results_save_folder_var_spectral,
        # Make below simpler and automatic
        save_name="forpowerspectra_bayespfspec_fulldatafft_std_300_particles_10_pp_32_random1_batch_finalvae_best_sample_train_y_pred_ar",
        batch_size=batch_size,
    )
