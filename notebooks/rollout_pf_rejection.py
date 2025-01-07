# This is a script to run a particle filtering rollout for a model.
# We can choose the number of timesteps, and what we want to filter for.

# hack to go a couple of directories up if we need to import from python files in some parent directory.

import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

import json

import numpy as np
import torch

from climatem.climate_data_loader_explore_ensembles import CausalClimateDataModule
from climatem.model.tsdcd_latent_explore import LatentTSDCD
#import climatem.climate_dataset_explore_ensembles as climate_dataset

# Now I want to apply my particle filter to this rollout.
# Here we have scoring functions and a function to do the particle filtering.

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")


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

    # here we can take the log of the spatial spectra if we want to, to help to focus on the high wavenumbers
    # if take_log:
    # print("Taking the log of the spatial spectra")
    # fft_true = torch.log(fft_true)
    # fft_pred = torch.log(fft_pred)

    # here we focus on only the highest half of the wavenumbers
    # if high_wavenumbers:
    # print("Taking only the top half of the wavenumbers")
    #
    # mid_index = fft_true.shape[-1] // 2
    # fft_true = fft_true[:, :, mid_index:]
    # fft_pred = fft_pred[:, :, mid_index:]

    # calculate the difference between the true and predicted spatial spectra
    spatial_spectra_score = torch.abs(fft_pred - fft_true)

    # take the mean of the spatial spectra score across the variables and the wavenumbers, the final 2 axes
    spatial_spectra_score = torch.mean(spatial_spectra_score, dim=(2, 3))

    # then normalise all the values of spatial_spectra_score by the maximum value
    # print("Spatial spectra score before normalising:", spatial_spectra_score)

    # Do normalisation and 1 - if we want the score to be increasing
    # spatial_spectra_score = spatial_spectra_score / torch.max(spatial_spectra_score)
    # print("Spatial spectra score normalised:", spatial_spectra_score)

    # the do 1 - score to give the score to be increasing...
    # spatial_spectra_score = 1 - spatial_spectra_score
    # print("Spatial spectra score doing 1 - score:", spatial_spectra_score)

    # print("The spatial spectra score shape should be (num_particles, num_batch_size):", spatial_spectra_score.shape)
    # score = ...
    return spatial_spectra_score


def score_the_samples_for_variance(y_true, y_pred_samples, num_particles=100):

    # print("y_true shape:", y_true.shape)
    # print("y_pred_samples shape:", y_pred_samples.shape)

    # calculate the variance of the true values
    var_true = torch.var(y_true, dim=(2))

    # calculate the variance of the predicted values
    var_pred = torch.var(y_pred_samples, dim=(3))

    # print("var true shape:", var_true.shape)
    # print("var pred shape:", var_pred.shape)

    # extend var_true so it is the same value but extended to the same shape as var_pred
    var_true = var_true.repeat(num_particles, 1, 1)

    # print("var true shape", var_true.shape)

    # calculate the difference between the true and predicted variances
    variance_score = torch.abs(var_true - var_pred)
    # print("Variance score raw:", variance_score)
    # print("variance score shape:", variance_score.shape)
    # take the mean of the variance score across the variables and the wavenumbers, the final 2 axes
    variance_score = torch.mean(variance_score, dim=(2))

    # print("minimum variance score:", torch.min(variance_score))
    # print("maximum variance score:", torch.max(variance_score))

    # print("Variance score meaned shape:", variance_score.shape)
    # normalise the variance score
    # variance_score = variance_score / torch.max(variance_score)

    # then do 1 - score
    # variance_score = 1 - variance_score
    # print("variance score return after normalising and 1 -:", variance_score)

    # print("The variance score shape should be the same as the number of samples:", variance_score.shape)
    # print("Variance score shape:", variance_score.shape)

    # print("What element of y_pred_samples has the highest variance - which is max variance?", torch.max(torch.var(y_pred_samples, dim=(2, 3))))

    return variance_score


def particle_filter(
    x,
    y,
    num_particles: torch.tensor = torch.tensor(20),
    timesteps: int = 120,
    batch_size: int = 256,
    score: str = "spatial_spectra",
    save_dir: str = None,
    save_name: str = None,
):
    """
    Implement a particle filter to make a set of autoregressive predictions, where each created sample is evaluated by
    some score, and we do a particle filter to select only best samples to continue the autoregressive rollout.

    We need to pass the directory to save stuff to, and the stem of the filenames...
    """

    print("Number of particles:", num_particles)
    print("Number of timesteps:", timesteps)

    for _ in range(timesteps):
        print(f"Filtering timestep {_}")

        num_particles = torch.tensor(num_particles).to(device)

        # Change the batch size to 128
        x = x[:batch_size]
        y = y[:batch_size]

        print("What is the shape of x and y?", x.shape, y.shape)

        # Prediction
        # make all the new predictions, taking samples from the latents
        unused_samples_from_xs, samples_from_zs, y = model.predict_sample(x, y, num_particles)

        # if values in samples_from_zs are larger than 1e15 then we need to clip them to 1e15
        print("Trying to clip the samples from zs")
        samples_from_zs = torch.clip(samples_from_zs, -1e10, 1e10)

        # then calculate the score of each of the samples
        # Update the weights, where we want the weights to increase as the score improves

        if score == "variance":
            new_weights = score_the_samples_for_variance(y, samples_from_zs, num_particles)
        elif score == "spatial_spectra":
            new_weights = score_the_samples_for_spatial_spectra(
                y, samples_from_zs, coords=coordinates, num_particles=num_particles, mid_latitudes=False
            )
        else:
            raise ValueError("Score must be either variance or spatial_spectra")

        # print("First 10 elements of the new weights:", new_weights[:10])

        # Resampling (e.g., systematic resampling)
        # indices = torch.multinomial(new_weights, num_particles, replacement=True)
        # selected_samples = samples_from_zs[indices]
        # weights = torch.ones(num_particles) / num_particles

        # print the index of the smallest value in new_weights

        # print('Minimum of the new weights:', torch.min(new_weights))
        # print('Maximum of the new weights:', torch.max(new_weights))

        # (new_samples, batch_size)

        # print('Index of the minimum of the new weights:', torch.argmin(new_weights, dim=0))

        # for each of the batch_size dim, choose the element of samples_from_z with the index torch.argmin(new_weights, dim=0)

        # JUST NEED TO MAKE THIS RIGHT!!! THE DIMENSIONS THAT I TAKE THE INDICES OVER ARE NOT QUITE RIGHT FOR THE SPECTRAL CASE
        indices = torch.argmin(new_weights, dim=0)

        # print("Indices shape:", indices.shape)
        # print("Samples from z shape", samples_from_zs.shape)
        # print("First 10 elements of indices:", indices[:10])

        # create a random array with the same shape as indices, with integer values between 0 and 4
        # indices_rnd = torch.randint(0, num_particles, (256,))

        # calculate the range in variances between the 20 samples for each element of the batch
        # print('The range in variance of the samples from z, per batch member:', torch.max(torch.var(samples_from_zs, dim=(2, 3))) - torch.min(torch.var(samples_from_zs, dim=(2, 3))))

        # print("What is the variance of the samples from z:", torch.var(samples_from_zs, dim=(1, 2, 3)))

        # for each of the second dimension of samples_from_z, choose the element of samples_from_z with the index torch.argmin(new_weights, dim=0)
        # THIS MIGHT WORK: selected_samples = torch.stack([samples_from_zs[indices[i], i, :, :] for i in range(samples_from_zs.shape[1])], dim=1)

        selected_samples = samples_from_zs[indices, torch.arange(batch_size), :, :]
        # selected_samples_random = samples_from_zs[indices_rnd, torch.arange(batch_size), :, :]

        # weights = torch.ones(num_particles) / num_particles

        # print("What is the shape of the selected samples", selected_samples.shape)

        # print("What is the variance of the selected samples:", torch.var(selected_samples, dim=(1, 2)))
        # print("What is the variance of the selected samples random:", torch.var(selected_samples_random, dim=(1, 2)))

        # assert that for each element of the two torch.var, for the selected_samples_random it is greater than or equal to the selected_samples
        # print(torch.all(torch.var(selected_samples_random, dim=(1, 2)) >= torch.var(selected_samples, dim=(1, 2))))
        # print("Sum of the number of samples where variance of random is greater than selected:",(torch.var(selected_samples_random, dim=(1, 2)) >= torch.var(selected_samples, dim=(1, 2))).sum())

        # check that selected_samples is the same as the one with the smallest value in new_weights
        # unsqueeze the selected samples

        # save the selected_samples
        np.save(os.path.join(save_dir, f"{save_name}_{_}.npy"), selected_samples.detach().cpu().numpy())
        print("Saved the selected samples with name:", f"{save_name}_{_}.npy")

        # then we are going to be passing the selected samples to the next timestep, so we need to make the input again
        # first drop the first value of x, then
        x = x[:, 1:, :, :]

        # now we just need to unsqueeze the selected samples, so that we can concatenate them to x
        selected_samples = selected_samples.unsqueeze(1)

        # print("What is the shape of x, just before we concatenate?", x.shape)
        # print("What is the shape of the selected samples, just before we concatenate?", selected_samples.shape)

        # then we need to append the selected samples to x, along the right axis
        x = torch.cat([x, selected_samples], dim=1)

        # then we are going back to the top of the loop

    return selected_samples


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Read the coordinates too...

coordinates = np.loadtxt("/home/mila/s/sebastian.hickman/work/icosahedral/mappings/vertex_lonlat_mapping.txt")
coordinates = coordinates[:, 1:]

# path to the results directory that I care about
# Now doing for two models, one where we learned a causal graph (taking the final model) and one where we didn't

results_dir_ts_vae = "/home/mila/s/sebastian.hickman/scratch/results/climatem_spectral/var_['ts']_scenarios_piControl_tau_5_z_90_lr_0.001_spreg_0.743706_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_20_tempspcoef_2000/"
results_dir_ts_novae = "/home/mila/s/sebastian.hickman/scratch/results/climatem_spectral/var_['ts']_scenarios_piControl_tau_5_z_90_lr_0.001_spreg_0.743706_ormuinit_100000.0_spmuinit_0.1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_20_tempspcoef_2000/"

# make sure we use the correct directory here

with open(results_dir_ts_vae + "params.json", "r") as f:
    hp = json.load(f)

# Let's overwrite some of the hyperparameters to see if we can load in some different ssp data...
# overwrite the config_exp_path here:

hp["config_exp_path"] = (
    "/home/mila/s/sebastian.hickman/work/climatem/scripts/configs/climate_predictions_picontrol_icosa_nonlinear_ensembles_hilatent_all_icosa_picontrol.json"
)
# hp['config_exp_path'] = '/home/mila/s/sebastian.hickman/work/climatem/scripts/configs/climate_predictions_picontrol_icosa_nonlinear_ensembles_hilatent_all_icosa_ssp126.json'
# hp['config_exp_path'] = '/home/mila/s/sebastian.hickman/work/climatem/scripts/configs/climate_predictions_picontrol_icosa_nonlinear_ensembles_hilatent_all_icosa_ssp245.json'

# once I have loaded in the state_dict, I can load it into a model
# first I need to define the model architecture

config_fname = hp["config_exp_path"]
with open(config_fname) as f:
    data_params = json.load(f)

datamodule = CausalClimateDataModule(**data_params)  # ...
datamodule.setup()

# getting the training data in place so that I can forecast using this data.
train_dataloader = iter(datamodule.train_dataloader())
# val_dataloader = iter(datamodule.val_dataloader())
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
    # NOTE: seb adding fixed to try to test when we have a fixed graph
    # also
    fixed=hp["fixed"],
    fixed_output_fraction=hp["fixed_output_fraction"],
)


# Here we load a final model, when we do learn the causal graph. Make sure  it is on GPU:

state_dict_vae_final = torch.load(results_dir_ts_vae + "model.pth", map_location=None)
model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict_vae_final.items()})

# Move the model to the GPU
model = model.to(device)
print("Where is the model?", next(model.parameters()).device)

# make sure the model is on GPU, and this all runs on GPU

# model = model.cuda()

scratch_path = "/home/mila/s/sebastian.hickman/scratch/results/dec30_particle_filters/ts_picontrol/"
# scratch_path = "/home/mila/s/sebastian.hickman/scratch/results/dec30_particle_filters/ts_ssp126/"
# scratch_path = "/home/mila/s/sebastian.hickman/scratch/results/dec30_particle_filters/ts_ssp245/"


# NOTE: make sure we specify the correct filepath to save the model in.
# NOTE: make sure we have the right setting for mid_latitudes in the particle_filter function above
# NOTE: and the corresponding correct naming for the save_name

with torch.no_grad():
    final_picontrol_particles = particle_filter(
        x,
        y,
        num_particles=20000,
        timesteps=1200,
        batch_size=16,
        score="spatial_spectra",
        save_dir="/home/mila/s/sebastian.hickman/scratch/results/dec30_particle_filters/ts_picontrol/spectral/",
        save_name="pfspecclip_20000_samples_100_years_16_batch_finalvae_best_sample_train_y_pred_ar",
    )
