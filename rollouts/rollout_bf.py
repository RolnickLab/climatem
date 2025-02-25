# This is a script to run a particle filtering rollout for a model.
# We can choose the number of timesteps, and what we want to filter for.
# Be careful with the number of batches we use for calculating the true data spectra.

# hack to go a couple of directories up if we need to import from python files in some parent directory.

import os
from pathlib import Path

import json

import numpy as np
import torch

from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from climatem.model.tsdcd_latent import LatentTSDCD
from climatem.rollouts.bayesian_filter import calculate_fft_mean_std_across_all_noresm, logscore_the_samples_for_spatial_spectra_bayesian, particle_filter_weighting_bayesian
from climatem.config import *

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")


# if we are doing ssps, and we want to only look at the last 30 years:
final_30_years_of_ssps = False

if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Read the coordinates too...

home_dir_path = Path("/home/mila/j/julien.boussard")
local_folder = home_dir_path / "causal_model"
scratch_dir = home_dir_path / "scratch"  # Where large data is stored
results_dir = scratch_dir / "results"
os.makedirs(results_dir, exist_ok=True)
climatem_repo = local_folder / "climatem"

coordinates_path = climatem_repo / "mappings/vertex_lonlat_mapping.npy"
coordinates = np.load(coordinates_path)

# NOTE: here saving SSP runs...
results_save_folder = results_dir / "new_climatem_spectral_filtered_100_year"
os.makedirs(results_save_folder, exist_ok=True)
# Make below updated with variables automatically + simpler
results_save_folder_var = results_save_folder / "logspectraltrain_ablations"
os.makedirs(results_save_folder_var, exist_ok=True)
results_save_folder_var_spectral = results_save_folder_var / "full_model_1crps_50spec_5000tspec_filtered_train"
os.makedirs(results_save_folder_var_spectral, exist_ok=True)

local_results_dir = results_dir / "climatem_spectral/var_ts_picontrol"
# os.makedirs(local_results_dir, exist_ok=False)
name_res_ts_vae = "var_ts_scenarios_piControl_nonlinear_True_tau_5_z_90_lr_0.001_bs_128_spreg_0.01_ormuinit_100000.0_spmuinit_1_spthres_0.5_fixed_False_num_ensembles_2_instantaneous_False_crpscoef_1_spcoef_20_tempspcoef_2000"
results_dir_ts_vae = local_results_dir / name_res_ts_vae
# os.makedirs(results_dir_ts_vae, exist_ok=False)

with open(results_dir_ts_vae / "params.json", "r") as f:
    hp = json.load(f)

hp["data_params"]["temp_res"] = "mon"
assert hp["data_params"]["seq_len"] == SEQ_LEN_MAPPING[hp["data_params"]["temp_res"]]
hp["data_params"].pop('seq_len', None)
hp["train_params"].pop('ratio_valid', None)

experiment_params = expParams(**hp["exp_params"])
data_params = dataParams(**hp["data_params"])
gt_params = gtParams(**hp["gt_params"])
train_params = trainParams(**hp["train_params"])
model_params = modelParams(**hp["model_params"])
optim_params = optimParams(**hp["optim_params"])

datamodule = CausalClimateDataModule(
    tau=experiment_params.tau,
    num_months_aggregated=data_params.num_months_aggregated,
    train_val_interval_length=data_params.train_val_interval_length,
    in_var_ids=data_params.in_var_ids,
    out_var_ids=data_params.out_var_ids,
    train_years=data_params.train_years,
    train_historical_years=data_params.train_historical_years,
    test_years=data_params.test_years,  # do we want to implement keeping only certain years for testing?
    val_split=1 - train_params.ratio_train,  # fraction of testing to split for valdation
    seq_to_seq=data_params.seq_to_seq,  # if true maps from T->T else from T->1
    channels_last=data_params.channels_last,  # wheather variables come last our after sequence lenght
    train_scenarios=data_params.train_scenarios,
    test_scenarios=data_params.test_scenarios,
    train_models=data_params.train_models,
    # test_models = data_params.test_models,
    batch_size=data_params.batch_size,
    eval_batch_size=data_params.eval_batch_size,
    num_workers=experiment_params.num_workers,
    pin_memory=experiment_params.pin_memory,
    load_train_into_mem=data_params.load_train_into_mem,
    load_test_into_mem=data_params.load_test_into_mem,
    verbose=experiment_params.verbose,
    seed=experiment_params.random_seed,
    seq_len=data_params.seq_len,
    data_dir=data_params.climateset_data,
    output_save_dir=data_params.data_dir,
    num_ensembles=data_params.num_ensembles,  # 1 for first ensemble, -1 for all
    lon=experiment_params.lon,
    lat=experiment_params.lon,
    num_levels=data_params.num_levels,
    global_normalization=data_params.global_normalization,
    seasonality_removal=data_params.seasonality_removal,
    reload_climate_set_data=True,
    icosahedral_coordinates_path=data_params.icosahedral_coordinates_path,
    # Below SAVAR data arguments
#             time_len=savar_params.time_len,
#             comp_size=savar_params.comp_size,
#             noise_val=savar_params.noise_val,
#             n_per_col=savar_params.n_per_col,
#             difficulty=savar_params.difficulty,
#             seasonality=savar_params.seasonality,
#             overlap=savar_params.overlap,
#             is_forced=savar_params.is_forced,
#             plot_original_data=savar_params.plot_original_data,
)
datamodule.setup()

y_true_fft_mean, y_true_fft_std = calculate_fft_mean_std_across_all_noresm(datamodule, number_of_batches=18)
print("y_true_fft_mean shape:", y_true_fft_mean.shape)
print("y_true_fft_std shape:", y_true_fft_std.shape)

train_dataloader = iter(datamodule.train_dataloader(accelerator))
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

x = x.to(device)
y = y.to(device)
print(f"Where is the data? x is on {x.device}, y is on {y.device}")


d = x.shape[2]
num_input = d * experiment_params.tau * (model_params.tau_neigh * 2 + 1)

# Instantiate a model here with the hyperparameters that we have loaded in.
model = LatentTSDCD(
    num_layers=model_params.num_layers,
    num_hidden=model_params.num_hidden,
    num_input=num_input,
    num_output=model_params.num_output,
    num_layers_mixing=model_params.num_layers_mixing,
    num_hidden_mixing=model_params.num_hidden_mixing,
    coeff_kl=optim_params.coeff_kl,
    d=d,
    distr_z0="gaussian",
    distr_encoder="gaussian",
    distr_transition="gaussian",
    distr_decoder="gaussian",
    d_x=experiment_params.d_x,
    d_z=experiment_params.d_z,
    tau=experiment_params.tau,
    instantaneous=model_params.instantaneous,
    nonlinear_mixing=model_params.nonlinear_mixing,
    hard_gumbel=model_params.hard_gumbel,
    no_gt=True,
    debug_gt_graph=None,
    debug_gt_z=None,
    debug_gt_w=None,
    # gt_w=data_loader['gt_w'],
    # gt_graph=data_loader['gt_graph'],
    tied_w=model_params.tied_w,
    # also
    fixed=model_params.fixed,
    fixed_output_fraction=model_params.fixed_output_fraction,
)

# Here we load a final model, when we do learn the causal graph. Make sure  it is on GPU:

state_dict_vae_final = torch.load(results_dir_ts_vae / "training_results/model.pth", map_location=device)
model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict_vae_final.items()})

# Move the model to the GPU
model = model.to(device)
print("Where is the model?", next(model.parameters()).device)


batch_size = 3

# select 16 random samples from the batch
def sample_from_tensor_reproducibly(tensor1, tensor2, num_samples, seed=5):
    if num_samples > tensor1.shape[0]:
        raise ValueError("Number of samples cannot exceed the tensor's first dimension.")

    torch.manual_seed(seed)  # Set the random seed
    indices = torch.randperm(tensor1.shape[0])[:num_samples]
    return tensor1[indices], tensor2[indices]

# First call with the seed
x_samples, y_samples = sample_from_tensor_reproducibly(x, y, batch_size)

np.save(
    results_save_folder_var_spectral / "forpowerspectra_random1_batch_xs_we_start_with.npy",
    x_samples.detach().cpu().numpy(),
)

num_particles = 50
num_particles_per_particle=5
num_timesteps = 1200

with torch.no_grad():
    final_picontrol_particles = particle_filter_weighting_bayesian(
        model,
        x_samples,
        y_samples,
        y_true_fft_mean,
        y_true_fft_std,
        coordinates,
        num_particles=num_particles,
        num_particles_per_particle=num_particles_per_particle,
        timesteps=num_timesteps,
        score="log_bayesian",
        save_dir=results_save_folder_var_spectral,
        save_name=f"forpowerspectra_bayespfspec_fulldatafft_std_{num_particles}_particles_{num_particles_per_particle}_pp_{batch_size}_random1_batch_finalvae_best_sample_train_y_pred_ar",
        batch_size=batch_size,
        tempering=True,
    )
