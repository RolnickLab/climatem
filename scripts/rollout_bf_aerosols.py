# This is a script to run a particle filtering rollout for a model.
# We can choose the number of timesteps, and what we want to filter for.
# Be careful with the number of batches we use for calculating the true data spectra.

# hack to go a couple of directories up if we need to import from python files in some parent directory.

import os
from pathlib import Path

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from climatem.data_loader.causal_datamodule import CausalClimateDataModule,CausalClimateDataMultiScenarioModule
from climatem.model.tsdcd_latent import LatentTSDCD
from climatem.rollouts.bayesian_filter import calculate_fft_mean_std_across_all_noresm, logscore_the_samples_for_spatial_spectra_bayesian, particle_filter_weighting_bayesian_with_aerosols
from climatem.config import *
from climatem.utils import parse_args, update_config_withparse
from shapely.geometry import Polygon, Point
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")

def rename_forcing_keys(state_dict):
    # I updated the model to accomodate more forcings, so previous autoencoder keys is not applicable anymore
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        # encoder mu
        new_key = new_key.replace(
            "co2_forcing_encoder_mu",
            "forcing_encoder_mu.co2"
        )
        new_key = new_key.replace(
            "aerosol_forcing_encoder_mu",
            "forcing_encoder_mu.aerosol"
        )
        # decoder
        new_key = new_key.replace(
            "decoder_co2",
            "forcing_decoder.co2"
        )
        new_key = new_key.replace(
            "decoder_aerosol",
            "forcing_decoder.aerosol"
        )
        # encoder logvar
        new_key = new_key.replace(
            "co2_forcing_encoder_logvar",
            "forcing_logvar_encoder.co2"
        )
        new_key = new_key.replace(
            "aerosol_forcing_encoder_logvar",
            "forcing_logvar_encoder.aerosol"
        )
        # decoder logvar
        new_key = new_key.replace(
            "logvar_co2_decoder",
            "forcing_logvar_decoder.co2"
        )
        new_key = new_key.replace(
            "logvar_aerosol_decoder",
            "forcing_logvar_decoder.aerosol"
        )
        # mask weights
        new_key = new_key.replace(
            "w_co2",
            "forcing_mask.co2"
        )
        new_key = new_key.replace(
            "w_aerosol",
            "forcing_mask.aerosol"
        )
        new_state_dict[new_key] = v
    return new_state_dict
def get_all_y_ssp(datamodule, accelerator):

    # Start again at the beginning of the dataloader.
    test_dataloader = iter(datamodule.test_dataloader(accelerator))

    # iterate through the data and append all the y values together
    y_all = []
    y_co2_all = []
    y_aerosol_all = []
    
    for i in range(len(test_dataloader)):
        batch = next(test_dataloader)
        if isinstance(batch, dict):
            # New format with forcings
            y_whole_dataloader = batch["y"]
            y_co2_dataloader = batch["global_forcings"][:,:,0] #(b, tau, n, 3072)
            y_aerosol_whole_dataloader = batch["spatial_forcings"][:,:,0]
        else:
            # Legacy format (tuple)
            _, y_whole_dataloader = batch
        y_all.append(y_whole_dataloader[:,0][None])
        y_co2_all.append(y_co2_dataloader[None])
        y_aerosol_all.append(y_aerosol_whole_dataloader[None])
    y_all = torch.cat(y_all, dim=0)
    y_co2_all = torch.cat(y_co2_all, dim=0)
    y_aerosol_all = torch.cat(y_aerosol_all,dim=0)
    y_all = torch.nan_to_num(y_all)
    y_co2_all = torch.nan_to_num(y_co2_all)
    y_aerosol_all = torch.nan_to_num(y_aerosol_all)
    print("y_all", y_all.shape, y_co2_all.shape, y_aerosol_all.shape)

    # make sure we reset the dataloader
    test_dataloader = iter(datamodule.test_dataloader(accelerator))

    return y_all, y_co2_all, y_aerosol_all
def create_aoi(coordinates):

    """
    coordinates:
        (npix, 2)
        [:,0] = longitude
        [:,1] = latitude

    return:
        boolean mask (npix,)
    """

    # Pacific equatorial region
    # longitude: -160 ~ -120
    # latitude: -10 ~ 10

    polygon = Polygon([
        (200, -10),
        (240, -10),
        (240, 10),
        (200, 10)
    ])
    # China area
    # polygon = Polygon([
    #     (73, 18),    # southwest
    #     (135, 18),   # southeast
    #     (135, 54),   # northeast
    #     (73, 54)     # northwest
    # ])


    aoi = np.zeros(
        coordinates.shape[0],
        dtype=bool
    )


    for i, (lon, lat) in enumerate(coordinates):

        point = Point(lon, lat)

        if polygon.contains(point):
            aoi[i] = True


    return aoi
def get_perturbed_forcings(datamodule, accelerator, 
                      perturbation_start=30, 
                      increase_factor=1.2,
                      perturbation_end=70,
                      perturbation_type="sinusoidal",
                      apply_to="co2",
                      coordinates=None):
    """
    this time we apply regional perturbation to aerosols
    """
    # Start again at the beginning of the dataloader.
    test_dataloader = iter(datamodule.test_dataloader(accelerator))

    # iterate through the data and append all the y values together
    y_co2_all = []
    y_aerosol_all = []
    
    for i in range(len(test_dataloader)):
        batch = next(test_dataloader)
        if isinstance(batch, dict):
            # New format with forcings
            y_co2_dataloader = batch["global_forcings"][:,:,0] #(b, tau, 1, 1)
            y_aerosol_whole_dataloader = batch["spatial_forcings"][:,:,0]#(b, tau, 1, 1)
        else:
            raise ValueError ("the current dataloader may not support forcings")
        
        y_co2_all.append(y_co2_dataloader[None])
        y_aerosol_all.append(y_aerosol_whole_dataloader[None])
    y_co2_all = torch.cat(y_co2_all, dim=0)
    y_aerosol_all = torch.cat(y_aerosol_all,dim=0)
    y_co2_all = torch.nan_to_num(y_co2_all)
    y_aerosol_all = torch.nan_to_num(y_aerosol_all)

    # baseline
    y_co2_perturbed = y_co2_all.clone()
    y_aerosol_perturbed = y_aerosol_all.clone()

    t = np.arange(perturbation_start, perturbation_end)

    if perturbation_type == "scaling":

        if apply_to in ["co2", "both"]:
            # cons = y_co2_all[perturbation_start]
            y_co2_perturbed[perturbation_start:perturbation_end] = (
                 y_co2_all[perturbation_start:perturbation_end] * increase_factor
             )

        if apply_to in ["aerosol", "both"]:
            y_aerosol_perturbed[perturbation_start:perturbation_end] = (
                 y_aerosol_all[perturbation_start:perturbation_end] * increase_factor
             )
    elif perturbation_type == "constant":
        aoi = create_aoi(coordinates)
        print("coordinates",coordinates[:,0])
        print("coordinates 1",coordinates[:,1])
        print("total_selected", aoi.sum())
        if apply_to in ["co2", "both"]:
            y_co2_perturbed[perturbation_start:perturbation_end,:,:, aoi] = 0

        if apply_to in ["aerosol", "both"]:
            y_aerosol_perturbed[perturbation_start:perturbation_end,:,:, aoi] = 6.0
    else:
        amplitude = 2.0
        if perturbation_type == "sinusoidal":
            perturbed_value = amplitude * np.sin(
                2 * np.pi * (t - perturbation_start) / 12
            )

        elif perturbation_type == "exponential_decay":
            perturbed_value = amplitude * np.exp(
                -0.01 * (t - perturbation_start)
            )

        else:
            raise ValueError(
                f"Unknown perturbation_type: {perturbation_type}"
            )

        # reshape for broadcasting to forcing tensor
        perturbed_value = torch.as_tensor(
            perturbed_value[:, None, None, None],
            dtype=y_co2_all.dtype,
            device=y_co2_all.device,
        )

        if apply_to in ["co2", "both"]:
            y_co2_perturbed[perturbation_start:perturbation_end] = (
                y_co2_all[perturbation_start:perturbation_end]
                + perturbed_value
            )

        if apply_to in ["aerosol", "both"]:
            y_aerosol_perturbed[perturbation_start:perturbation_end] = (
                y_aerosol_all[perturbation_start:perturbation_end]
                + perturbed_value
            )
   
    # make sure we reset the dataloader
    test_dataloader = iter(datamodule.test_dataloader(accelerator))

    return y_co2_perturbed, y_aerosol_perturbed
# select 16 random samples from the batch

def sample_from_tensor_reproducibly(tensor1, tensor2, num_samples, seed=5):
    if num_samples > tensor1.shape[0]:
        raise ValueError("Number of samples cannot exceed the tensor's first dimension.")

    torch.manual_seed(seed)  # Set the random seed
    indices = torch.randperm(tensor1.shape[0])[:num_samples]
    return tensor1[indices], tensor2[indices]

def main_perturbed(
    experiment_params, 
    data_params, 
    # gt_params, 
    train_params, 
    model_params, 
    optim_params, 
    plot_params, 
    savar_params,
    rollout_params,
    exp_id,
    iter_id,
    test_scenarios,
    perturbed_variable="co2",
    perturbed_type="scaling",
    total_time_steps = 1027
    
):
    """
    :param hp: object containing hyperparameter values
    """

    # Control as much randomness as possible
    torch.manual_seed(experiment_params.random_seed)
    np.random.seed(experiment_params.random_seed)
    
    device = torch.device("cuda" if (torch.cuda.is_available() and experiment_params.gpu) else "cpu")

    if experiment_params.gpu and torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    if data_params.data_format == "hdf5":
        print("IS HDF5")
        return
    else: 
        common_args = dict(
            tau=experiment_params.tau,
            future_timesteps=experiment_params.future_timesteps,
            num_months_aggregated=data_params.num_months_aggregated,
            train_val_interval_length=data_params.train_val_interval_length,
            in_var_ids=data_params.in_var_ids,
            out_var_ids=data_params.out_var_ids,
            train_years=data_params.train_years,
            train_historical_years=data_params.train_historical_years,
            test_years=data_params.test_years,
            val_split=1 - train_params.ratio_train,
            seq_to_seq=data_params.seq_to_seq,
            channels_last=data_params.channels_last,
            train_scenarios=data_params.train_scenarios,
            test_scenarios=test_scenarios,
            train_models=data_params.train_models,
            temp_res=data_params.temp_res,
            batch_size=rollout_params.batch_size,
            eval_batch_size=rollout_params.batch_size,
            num_workers=experiment_params.num_workers,
            pin_memory=experiment_params.pin_memory,
            load_train_into_mem=data_params.load_train_into_mem,
            load_test_into_mem=data_params.load_test_into_mem,
            verbose=experiment_params.verbose,
            seed=experiment_params.random_seed,
            seq_len=data_params.seq_len,
            data_dir=data_params.climateset_data,
            output_save_dir=data_params.data_dir,
            num_ensembles=data_params.num_ensembles,
            lon=experiment_params.lon,
            lat=experiment_params.lat,
            num_levels=data_params.num_levels,
            global_normalization=data_params.global_normalization,
            seasonality_removal=data_params.seasonality_removal,
            reload_climate_set_data=data_params.reload_climate_set_data,
            icosahedral_coordinates_path=data_params.icosahedral_coordinates_path,
            time_len=savar_params.time_len,
            comp_size=savar_params.comp_size,
            noise_val=savar_params.noise_val,
            n_per_col=savar_params.n_per_col,
            difficulty=savar_params.difficulty,
            seasonality=savar_params.seasonality,
            overlap=savar_params.overlap,
            is_forced=savar_params.is_forced,
            f_1=savar_params.f_1,
            f_2=savar_params.f_2,
            f_time_1=savar_params.f_time_1,
            f_time_2=savar_params.f_time_2,
            ramp_type=savar_params.ramp_type,
            linearity=savar_params.linearity,
            poly_degrees=savar_params.poly_degrees,
            plot_original_data=savar_params.plot_original_data,
        )
        savar_args=dict(            
            time_len=savar_params.time_len,
            comp_size=savar_params.comp_size,
            noise_val=savar_params.noise_val,
            n_per_col=savar_params.n_per_col,
            difficulty=savar_params.difficulty,
            seasonality=savar_params.seasonality,
            overlap=savar_params.overlap,
            is_forced=savar_params.is_forced,
            f_1=savar_params.f_1,
            f_2=savar_params.f_2,
            f_time_1=savar_params.f_time_1,
            f_time_2=savar_params.f_time_2,
            ramp_type=savar_params.ramp_type,
            linearity=savar_params.linearity,
            poly_degrees=savar_params.poly_degrees,
            plot_original_data=savar_params.plot_original_data
            )
        DATA_REGISTRY = {
            "single": {
                "cls": CausalClimateDataModule,
                "args": {**common_args, **savar_args},
            },
            "multi": { # the multiscenario dataset doesn't support savar input yet.
                "cls": CausalClimateDataMultiScenarioModule,
                "args": {**common_args}
            },
        }

        data_module_type = (
            "multi"
            if len(data_params.train_scenarios) > 1
            else "single"
        )

        cfg = DATA_REGISTRY[data_module_type]
        datamodule = cfg["cls"](**cfg["args"])
        datamodule.setup()

    d = len(data_params.out_var_ids)
    print(f"Using {d} variables")

    if model_params.instantaneous:
        print("Using instantaneous connections")
        num_input = d * (experiment_params.tau + 1) 
    else:
        num_input = d * experiment_params.tau 
    
    class PerturbLatentTSDCD(LatentTSDCD):

        def __init__(
            self,
            **kwargs
        ):
            super().__init__(**kwargs)
        def predict_aerosols(self, x, y, y_co2=None, y_aerosol=None):
            with torch.no_grad():
                # sample Zs (based on X)
                z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol)
                n_climate_latents = self.d_z - self.n_forced_latents_total
                z_forced_target = z[:, -1, 0, n_climate_latents:]
                forcing_outputs = self.autoencoder.decode_forcings(z_forced_target)
                aerosol_mu = forcing_outputs["aerosol_mu"] 
                aerosol_logvar= forcing_outputs["aerosol_logvar"]
                aerosol_var = torch.exp(0.5 * aerosol_logvar)
                samples = self.distr_decoder(aerosol_mu, aerosol_var).sample()

            return samples
    
                            

    # set the model
    model = PerturbLatentTSDCD(
        num_layers=model_params.num_layers,
        num_hidden=model_params.num_hidden,
        num_input=num_input,
        num_output=2,
        num_layers_mixing=model_params.num_layers_mixing,
        num_hidden_mixing=model_params.num_hidden_mixing,
        position_embedding_dim=model_params.position_embedding_dim,
        reduce_encoding_pos_dim=model_params.reduce_encoding_pos_dim,
        transition_param_sharing=model_params.transition_param_sharing,
        position_embedding_transition=model_params.position_embedding_transition,
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
        instantaneous_forcing=model_params.instantaneous_forcing,
        nonlinear_dynamics=model_params.nonlinear_dynamics,
        nonlinear_mixing=model_params.nonlinear_mixing,
        tied_w=model_params.tied_w,
        fixed=model_params.fixed,
        fixed_output_fraction=model_params.fixed_output_fraction,
        use_exogenous=model_params.use_exogenous,
        d_y_co2=model_params.d_y_co2,
        d_y_aerosol=model_params.d_y_aerosol,
        use_forced_latents=model_params.use_forced_latents,
        n_forced_latents_co2=model_params.n_forced_latents_co2,
        n_forced_latents_aerosol=model_params.n_forced_latents_aerosol,
        forcing_arch=model_params.forcing_arch,
        d_y_ch4=model_params.d_y_ch4,
        d_y_so2=model_params.d_y_so2,
        n_forced_latents_ch4=model_params.n_forced_latents_ch4,
        n_forced_latents_so2=model_params.n_forced_latents_so2
        
    )
    
    # read paths 
    coordinates = np.load(data_params.icosahedral_coordinates_path)

    exp_path = Path(experiment_params.exp_path)
    if not os.path.exists(exp_path): 
        raise ValueError(f"Results path {exp_path} doesn't exist. Model should be saved in this folder")
    
    
    if data_params.in_var_ids[0] == "savar":
        name = f"savar_{savar_params.linearity}_{savar_params.is_forced}_{savar_params.difficulty}_{savar_params.n_per_col**2}_nlinmix_{model_params.nonlinear_mixing}_nlindyn_{model_params.nonlinear_dynamics}_tau_{experiment_params.tau}_z_{experiment_params.d_z}_futt_{experiment_params.future_timesteps}_ldec_{optim_params.loss_decay_future_timesteps}_lr_{train_params.lr}_bs_{data_params.batch_size}_ormuin_{optim_params.ortho_mu_init}_spmuin_{optim_params.sparsity_mu_init}_spth_{optim_params.sparsity_upper_threshold}_nummix_hid_{model_params.num_hidden_mixing}_{model_params.num_layers_mixing}_{model_params.num_hidden}_{model_params.num_layers}_embdim_{model_params.position_embedding_dim}_trparamsh_{model_params.transition_param_sharing}_posembdimtr_{model_params.position_embedding_transition}"
    else:
        name = exp_id
    exp_path = exp_path / name
    if not os.path.exists(exp_path): 
        raise ValueError(f"Results path {exp_path} does not exist. Are you using the same parameters?")

    # create path to exp and save hyperparameters
    save_path = exp_path / "rollouts"
    os.makedirs(save_path, exist_ok=True)

    model_path = exp_path #/ "training_results"

    y_true_fft_mean, y_true_fft_std = calculate_fft_mean_std_across_all_noresm(datamodule, accelerator)
    print("y_true_fft_mean shape:", y_true_fft_mean.shape)
    print("y_true_fft_std shape:", y_true_fft_std.shape)

    test_dataloader = iter(datamodule.test_dataloader(accelerator))
    # total_time_steps = len(test_dataloader)
    save_path = save_path / f"bs_{rollout_params.batch_size}_np_{rollout_params.num_particles}_npp_{rollout_params.num_particles_per_particle}_t_{total_time_steps}_sc_{rollout_params.score}_temp_{rollout_params.tempering}_iter{iter_id}_{test_scenarios[0]}_perturbed_A"
    os.makedirs(save_path, exist_ok=True)
    
    batch = next(test_dataloader)

    if rollout_params.final_30_years_of_ssps:
        print("Taking the final 30 years of the SSP data, ~ 2070-2100")
        batch = next(test_dataloader)
        batch = next(test_dataloader)


    if isinstance(batch, dict):
        x = batch["x"]
        y = batch["y"]
        y_global_focings = batch.get("global_forcings", None)
        y_spatial_forcings = batch.get("spatial_forcings", None)
        if y_global_focings is not None and y_spatial_forcings is not None:
            y_co2 = y_global_focings[:,:,0]
            y_aerosol = y_spatial_forcings[:,:,0]
        # Extract ground truth forcing latents for supervision
        gt_co2_latent = batch.get("gt_co2_latent", None)
        gt_aerosol_latent = batch.get("gt_aerosol_latent", None)
    else:
        # Legacy format (tuple)
        x, y = batch
        y_co2, y_aerosol = None, None
        gt_co2_latent, gt_aerosol_latent = None, None


    if y_co2 is not None:
        y_co2 = torch.nan_to_num(y_co2).to(device)
    if y_aerosol is not None:
        y_aerosol = torch.nan_to_num(y_aerosol).to(device)
    if gt_co2_latent is not None:
        gt_co2_latent = torch.nan_to_num(gt_co2_latent).to(device)
    if gt_aerosol_latent is not None:
        gt_aerosol_latent = torch.nan_to_num(gt_aerosol_latent).to(device)
    x = torch.nan_to_num(x)
    y = torch.nan_to_num(y)
    y = y[:, 0]
    z = None

    x = x.to(device)
    y = y.to(device)

    # Here we load a final model, when we do learn the causal graph. Make sure  it is on GPU:
    state_dict_vae_final = torch.load(
        model_path / "model.pth", 
        map_location=device
    )
    state_dict = rename_forcing_keys(state_dict_vae_final)

    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})

    # Move the model to the GPU
    model = model.to(device)
    print("Where is the model?", next(model.parameters()).device)

    # First call with the seed
    x_samples, y_samples = sample_from_tensor_reproducibly(x, y, rollout_params.batch_size)
    np.save(
        save_path / "forpowerspectra_random1_batch_xs_we_start_with.npy",
        x_samples.detach().cpu().numpy(),
    )

    with torch.no_grad():
        thresholded_adj = (model.get_adj() > 0.5).type(torch.Tensor)
        model.mask.fix(thresholded_adj)
    all_y, all_co2, all_aerosols = get_all_y_ssp(datamodule, accelerator)
    all_co2_perturbed, all_aerosol_perturbed = get_perturbed_forcings(datamodule, accelerator,apply_to=perturbed_variable, perturbation_type=perturbed_type, coordinates=coordinates)

    ssp = common_args["test_scenarios"][0]

    np.savez(
        save_path / f"gt_all_forcings_{ssp}_perturbed.npz",
        all_y=all_y.detach().cpu().numpy(),
        all_co2=all_co2.detach().cpu().numpy(),
        all_aerosols=all_aerosols.detach().cpu().numpy(),
        all_co2_perturbed=all_co2_perturbed.detach().cpu().numpy(), # -> but it's not necessary that all two forcings are perturbed
        all_aerosol_perturbed=all_aerosol_perturbed.detach().cpu().numpy(),
    )

    unperturbed_set = (all_y, all_co2, all_aerosols)
    perturbed_set = (all_y, all_co2_perturbed, all_aerosol_perturbed)
    with torch.no_grad():
        final_picontrol_particles = particle_filter_weighting_bayesian_with_aerosols(
            model,
            x_samples,
            y_samples,
            y_true_fft_mean,
            y_true_fft_std,
            coordinates,
            num_particles=rollout_params.num_particles,
            num_particles_per_particle=rollout_params.num_particles_per_particle,
            timesteps=total_time_steps,
            score=rollout_params.score,
            save_dir=save_path,
            save_name=f"trajectory_iteration",
            batch_size=rollout_params.batch_size,
            tempering=rollout_params.tempering,
            sample_trajectories=rollout_params.sample_trajectories,
            batch_memory=rollout_params.batch_memory,
            all_y_ssp=unperturbed_set
        )
        final_picontrol_particles_perturbed = particle_filter_weighting_bayesian_with_aerosols(
            model,
            x_samples,
            y_samples,
            y_true_fft_mean,
            y_true_fft_std,
            coordinates,
            num_particles=rollout_params.num_particles,
            num_particles_per_particle=rollout_params.num_particles_per_particle,
            timesteps=total_time_steps,
            score=rollout_params.score,
            save_dir=save_path,
            save_name=f"trajectory_iteration_perturbed",
            batch_size=rollout_params.batch_size,
            tempering=rollout_params.tempering,
            sample_trajectories=rollout_params.sample_trajectories,
            batch_memory=rollout_params.batch_memory,
            all_y_ssp=perturbed_set
        )
    return final_picontrol_particles_perturbed
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import os
import pandas as pd


def plot_healpix_map(data, coordinates, title, filename, cmap="RdBu_r",
                     vmin=None, vmax=None):

    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Robinson())

    coords = coordinates.copy()

    if np.max(coords[:, 0]) > 91:
        coords = np.flip(coords, axis=1)

    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black")

    sc = ax.scatter(
        coords[:, 1],
        coords[:, 0],
        c=data,
        cmap=cmap,
        s=10,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)

    plt.colorbar(sc, orientation="horizontal",
                 pad=0.05, fraction=0.05)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()



def create_perturbed_gifs(
        traj_path,
        ssp,
        perturbed_variable,
        perturbed_type,
        T,
        coordinates):

    save_dir = f"{traj_path}_{ssp}_perturbed_A"
    

    # ======================
    # Load predictions
    # ======================

    trajs = []
    perturbed_trajs = []
    perturbed_trajs_aerosols = []


    for i in range(T):

        traj = np.load(
            f"{save_dir}/trajectory_iteration_{i}.npy"
        )

        traj_p = np.load(
            f"{save_dir}/trajectory_iteration_perturbed_{i}.npy"
        ) #(50,1,1,3072) 

        # (50,1,1,3072) -> (3072)
        trajs.append(
            traj.mean((0,1,2))
        )

        perturbed_trajs.append(
            traj_p.mean((0,1,2))
        )


        aerosol_p = np.load(
            f"{save_dir}/trajectory_iteration_perturbed_aerosols_{i}.npy"
        ) #(1,3072)

        perturbed_trajs_aerosols.append(
            aerosol_p.mean((0))
        )



    trajs = np.stack(trajs)
    perturbed_trajs = np.stack(perturbed_trajs)
    perturbed_trajs_aerosols = np.stack(
        perturbed_trajs_aerosols
    )


    print("Prediction TS:", trajs.shape)
    print("Prediction perturbed TS:", perturbed_trajs.shape)
    print("Prediction perturbed aerosol:",
          perturbed_trajs_aerosols.shape)



    # ======================
    # Load GT
    # ======================

    data = np.load(
        f"{save_dir}/gt_all_forcings_{ssp}_perturbed.npz"
    )


    gt_data = data["all_y"].squeeze()
    aerosol_data = data["all_aerosols"].squeeze()[:T,-1]
    aerosol_data_perturbed = (
        data["all_aerosol_perturbed"].squeeze()
    )[:T,-1]


    print("GT:", gt_data.shape)
    print("Aerosol:", aerosol_data.shape)



    time_steps = pd.date_range(
        "2015-01-01",
        periods=T,
        freq="MS"
    )


    # ======================
    # GIF 1:
    # TS maps
    # ======================

    ts_frames=[]


    for t in range(T):

        fname = f"{save_dir}/tmp_ts_{t}.png"


        fig = plt.figure(figsize=(15,4))


        datasets=[
            gt_data[t],
            trajs[t],
            perturbed_trajs[t]
        ]
        all_data = np.concatenate(datasets)

        vmin = -3.5#np.percentile(all_data, 1)
        vmax = 6.5#np.percentile(all_data, 99)
        titles=[
            f"GT TS {time_steps[t]}",
            f"Prediction TS {time_steps[t]}",
            f"Prediction TS {time_steps[t]}"
        ]
        if t >= 30 and t <= 70:
            titles[2]+= "-Perturbed"


        for i,(x,title) in enumerate(
                zip(datasets,titles)):


            ax = fig.add_subplot(
                1,3,i+1,
                projection=ccrs.Robinson()
            )

            coords=coordinates.copy()

            if np.max(coords[:,0])>91:
                coords=np.flip(coords,axis=1)


            ax.set_global()
            ax.coastlines()


            sc=ax.scatter(
                coords[:,1],
                coords[:,0],
                c=x,
                cmap="RdBu_r",
                s=8,
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree()
            )
            last_sc = sc


            ax.set_title(title)

        cbar=fig.colorbar(
            last_sc,
            ax=fig.axes,
            orientation="horizontal",
            fraction=0.05,
            pad=0.05
        )

        cbar.set_label("TS")
        # plt.tight_layout()
        # plt.colorbar(sc, label="TS")
        plt.savefig(fname,dpi=120)
        plt.close()


        ts_frames.append(
            imageio.imread(fname)
        )


    gif1=f"{save_dir}/TS_comparison.gif"

    imageio.mimsave(
        gif1,
        ts_frames,
        duration=0.8
    )


    print("Saved:",gif1)



    # ======================
    # GIF 2:
    # Aerosol maps
    # ======================


    aerosol_frames=[]


    for t in range(T):

        fname=f"{save_dir}/tmp_aerosol_{t}.png"


        fig=plt.figure(figsize=(15,4))


        datasets=[
            aerosol_data[t],
            aerosol_data_perturbed[t],
            perturbed_trajs_aerosols[t]
        ]
        all_data = np.concatenate(datasets)

        vmin = -1 #np.percentile(all_data, 1)
        vmax = 12.5#np.percentile(all_data, 99)

        titles=[
            "GT aerosol",
            "GT perturbed aerosol",
            "Prediction aerosol"
        ]
        if t >= 30 and t <= 70:
            titles[2]+= "-Perturbed"


        for i,(x,title) in enumerate(
                zip(datasets,titles)):


            ax=fig.add_subplot(
                1,3,i+1,
                projection=ccrs.Robinson()
            )

            coords=coordinates.copy()

            if np.max(coords[:,0])>91:
                coords=np.flip(coords,axis=1)


            ax.set_global()
            ax.coastlines()


            sc=ax.scatter(
                coords[:,1],
                coords[:,0],
                c=x,
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                s=8,
                transform=ccrs.PlateCarree()
            )
            last_sc=sc

            ax.set_title(
                f"{title}\n{time_steps[t]}"
            )


        cbar=fig.colorbar(
            last_sc,
            ax=fig.axes,
            orientation="horizontal",
            fraction=0.05,
            pad=0.05
        )

        cbar.set_label("Aerosol")
        # plt.tight_layout()

        plt.savefig(fname,dpi=120)
        plt.close()


        aerosol_frames.append(
            imageio.imread(fname)
        )


    gif2=f"{save_dir}/aerosol_comparison.gif"


    imageio.mimsave(
        gif2,
        aerosol_frames,
        duration=0.8
    )


    print("Saved:",gif2)


def check_corr_co2_aerosol(traj_path, ssps):
    
    T= 1027

    variables = ["co2","aerosols"]

    

    # Plot 1: plot all ssp in a single figure

    colors = ['blue', 'red', 'green', 'orange', 'purple'] 

    num_vars = len(variables)
    
    # ---------------- load GT ----------------
    gt_data = {}
    co2_data = {}
    aerosol_data = {}

    for ssp in ssps:
        data = np.load(f"{traj_path}_{ssp}/gt_all_y_{ssp}.npz")
        co2_data[ssp] = data["all_co2"].squeeze()[:,-1]  # (1027,1,6, 1)
        aerosol_data[ssp] = data["all_aerosols"].squeeze()[:,-1,:]  # (1027, 1, 6, 3072)
    print(data["all_aerosols"].shape)
    time_steps = pd.date_range(start="2015-01-01", periods=T, freq="MS")
    # ---------------- plotting ----------------
    from scipy.stats import pearsonr

    # co2: (T,)
    # aerosol: (T, H, W)
    def sliding_corr_10yr_mean(co2, aerosol, window=120):
        """
        Sliding-window correlation between CO2 and spatial-mean aerosol.
        """

        co2 = co2.squeeze()
        aero_mean = aerosol.mean(axis=(-1))  # (T,)

        T = len(co2)
        n_steps = T - window + 1

        corr = np.full(n_steps, np.nan)

        for t in range(n_steps):
            co2_win = co2[t:t+window]
            aero_win = aero_mean[t:t+window]

            # remove mean
            co2_win = co2_win - co2_win.mean()
            aero_win = aero_win - aero_win.mean()

            std_co2 = co2_win.std()
            std_aero = aero_win.std()

            if std_co2 == 0 or std_aero == 0:
                continue

            corr[t] = np.mean(co2_win * aero_win) / (std_co2 * std_aero)

        return corr

    def compute_corr_map(co2, aerosol):
        """
        Compute correlation map over time for each grid cell.
        """
        T, Npix = aerosol.shape
        corr_map = np.zeros(Npix)

        co2 = co2.squeeze()

        for i in range(Npix):
            ts = aerosol[:, i]

            # handle constant series
            if np.std(ts) == 0 or np.std(co2) == 0:
                corr_map[i] = np.nan
            else:
                corr_map[i] = np.corrcoef(co2, ts)[0, 1]

        return corr_map

    def plot_sliding_corr(corr, window=10):
        plt.figure(figsize=(12, 4))
        plt.plot(corr)
        plt.axhline(0, color='black', linewidth=1)

        plt.title(f"Sliding {window}-year correlation (CO2 vs aerosol mean)")
        plt.xlabel("Time window index")
        plt.ylabel("Correlation")
        plt.grid(True)
        plt.show()
    def plot_corr_map(corr_map,coordinates, title="CO2–Aerosol Correlation Map"):

        fig, ax = plt.subplots(
        1, 1, subplot_kw={"projection": ccrs.Robinson()}, layout="constrained", figsize=(10, 5)
    )

        ax.set_global()
        ax.coastlines()
        # Add some map features for context
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.gridlines(draw_labels=False)

        lon = coordinates[:, 0]
        lat = coordinates[:, 1]

        # print('x_past shape:', x_past.shape)
        sc = ax.scatter(
            x=lon,
            y=lat,
            c=corr_map,
            alpha=1,
            s=30,
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            transform=ccrs.PlateCarree(),
        )
       

        plt.colorbar(sc, label="Pearson r")
        plt.title(title)
        plt.axis("off")

    # Example usage:
    # co2: (T,)
    # aerosol: (T, N)
    coordinates = np.load("/home/mila/s/shanz/dev/climatem/mappings/healpix_resdown4_lonlat_mapping.npy")
    corr_map = compute_corr_map(co2_data[ssp], aerosol_data[ssp])
    plot_corr_map(corr_map,coordinates)
    filename = f"{traj_path}_ssp370/corr_co2_aerosols_{ssps}.png"
    # corr_series = sliding_corr_10yr_mean(co2_data[ssp], aerosol_data[ssp])
    # plot_sliding_corr(corr_series)
    # filename = f"{traj_path}_ssp370/corr_series_{ssps}.png"
    
    plt.savefig(filename) # save to the last one
    print(f"Saved to {filename}")
    plt.close()


def check_corr_ts_aerosol(traj_path, ssps):
    
    T=1027
    pred_data = {}
    for ssp in ssps:
        trajs = []
        for i in range(T):
            traj = np.load(f"{traj_path}_{ssp}/trajectory_iteration_{i}.npy")  # shape (50, 1, 1, 3072)
            trajs.append(traj.mean((0,1)))     # -> (5, 3072)
        print(f"a signle traj has the shape of {traj.shape}")
        trajs = np.stack(trajs, axis=0) 
        print("trajs",trajs.shape) 
        pred_data[ssp] = trajs.squeeze()
    # ---------------- load GT ----------------
    gt_data = {}
    co2_data = {}
    aerosol_data = {}
    lags=[0,1,2,3,4,5]
    for lag in lags:
        for ssp in ssps:
            data = np.load(f"{traj_path}_{ssp}/gt_all_y_{ssp}.npz")
            co2_data[ssp] = data["all_co2"].squeeze()[:,-1]  # (1027,1,6, 1)
            aerosol_data[ssp] = data["all_aerosols"].squeeze()[:,-lag-1,:]  # (1027, 1, 6, 3072)
            gt_data[ssp] = data["all_y"].squeeze()
        print(data["all_y"].shape)


        def compute_correlation_map(gt, aerosol):
            """
            Compute Pearson correlation map between gt and aerosol.

            Parameters
            ----------
            gt : np.ndarray
                Ground truth data with shape (T, N)
                T: number of time steps
                N: number of grid cells

            aerosol : np.ndarray
                Aerosol data with shape (T, N)

            Returns
            -------
            corr_map : np.ndarray
                Pearson correlation coefficient for each grid cell.
                Shape: (N,)
            """

            assert gt.shape == aerosol.shape, "gt and aerosol must have the same shape"

            # Remove temporal mean for each grid cell
            gt_anomaly = gt - np.mean(gt, axis=0, keepdims=True)
            aerosol_anomaly = aerosol - np.mean(aerosol, axis=0, keepdims=True)

            # Pearson correlation numerator
            covariance = np.sum(gt_anomaly * aerosol_anomaly, axis=0)

            # Standard deviation term
            std_product = np.sqrt(
                np.sum(gt_anomaly ** 2, axis=0) *
                np.sum(aerosol_anomaly ** 2, axis=0)
            )

            # Correlation map
            corr_map = covariance / std_product

            # Avoid division by zero
            corr_map[std_product == 0] = np.nan

            return corr_map


        def plot_corr_map(corr_map,coordinates, title=f"GT TS(t)–Aerosol (t-{lag}) Correlation Map"):

            fig, ax = plt.subplots(
            1, 1, subplot_kw={"projection": ccrs.Robinson()}, layout="constrained", figsize=(10, 5)
        )

            ax.set_global()
            ax.coastlines()
            # Add some map features for context
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, edgecolor="black")
            ax.gridlines(draw_labels=False)

            lon = coordinates[:, 0]
            lat = coordinates[:, 1]

            # print('x_past shape:', x_past.shape)
            sc = ax.scatter(
                x=lon,
                y=lat,
                c=corr_map,
                alpha=1,
                s=30,
                vmin=-1,
                vmax=1,
                cmap="RdBu_r",
                transform=ccrs.PlateCarree(),
            )
        

            plt.colorbar(sc, label="Pearson r")
            plt.title(title)
            plt.axis("off")

        # Example usage:
        # gt: (T, N)
        # aerosol: (T, N)
        coordinates = np.load("/home/mila/s/shanz/dev/climatem/mappings/healpix_resdown4_lonlat_mapping.npy")
        corr_map = compute_correlation_map(gt_data[ssp], aerosol_data[ssp])
        plot_corr_map(corr_map,coordinates)

        filename = f"{traj_path}_ssp370/corr_map_gt_lag{lag}_{ssps}.png"
        
        plt.savefig(filename) # save to the last one
        print(f"Saved to {filename}")
        plt.close()

if __name__ == "__main__":

    args = parse_args()
    
    cwd = Path.cwd()
    root_path = cwd.parent
    config_path = root_path / f"configs"
    exp_id = args.exp_id
    iter_id = args.iter_id
    test_scenarios = ["ssp370"]#args.test_scenarios
    
    folder = exp_id.split("/")[0] 
    exp_id = exp_id.split("/")[-1]

    # get user's scratch directory:
    scratch_path = os.getenv("SCRATCH")
    config_path = Path(scratch_path) /"results" / args.exp_id
    json_path = config_path / args.config_path

    
    with open(json_path, "r") as f:
        params = json.load(f)
    params = update_config_withparse(params, args)


    params["data_params"]["data_dir"] = params["data_params"]["data_dir"].replace("$SCRATCH", scratch_path)
    print ("new data path:", params["data_params"]["data_dir"])

    params["exp_params"]["exp_path"] = params["exp_params"]["exp_path"].replace("$SCRATCH", scratch_path)
    print ("new exp path:", params["exp_params"]["exp_path"])

    # get directory of project via current file (aka .../climatem/scripts/main_picabu.py)
    params["data_params"]["icosahedral_coordinates_path"] = params["data_params"]["icosahedral_coordinates_path"].replace("$CLIMATEMDIR", root_path.absolute().as_posix())
    print ("new icosahedron path:", params["data_params"]["icosahedral_coordinates_path"])

    experiment_params = expParams(**params["exp_params"])
    data_params = dataParams(**params["data_params"])
    # gt_params = gtParams(**params["gt_params"])
    train_params = trainParams(**params["train_params"])
    model_params = modelParams(**params["model_params"])
    optim_params = optimParams(**params["optim_params"])
    plot_params = plotParams(**params["plot_params"])
    savar_params = savarParams(**params["savar_params"])
    rollout_params = rolloutParams(**params["rollout_params"])
    coordinates = np.load(data_params.icosahedral_coordinates_path)

    #Overwrite arguments if using savar
    if "savar" in data_params.in_var_ids:
        experiment_params.lat = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.lon = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.d_x = int(experiment_params.lat * experiment_params.lon)
        plot_params.savar = True
    else:
        plot_params.savar = False
    rollout_steps = 100
    result_path = f"bs_{rollout_params.batch_size}_np_{rollout_params.num_particles}_npp_{rollout_params.num_particles_per_particle}_t_{rollout_steps}_sc_{rollout_params.score}_temp_{rollout_params.tempering}_iter{iter_id}"
    # check_corr_co2_aerosol(Path(f"{scratch_path}/results/{args.exp_id}/rollouts/{result_path}"),ssps = ["ssp370"])
    # check_corr_ts_aerosol(Path(f"{scratch_path}/results/{args.exp_id}/rollouts/{result_path}"),ssps = ["ssp370"])
    perturbed_variable="aerosol"
    perturbed_type="constant"
    _ = main_perturbed(experiment_params, data_params, train_params, model_params, optim_params, plot_params, savar_params, rollout_params, exp_id, iter_id, test_scenarios, perturbed_variable=perturbed_variable, perturbed_type=perturbed_type, total_time_steps =rollout_steps)
    create_perturbed_gifs(Path(f"{scratch_path}/results/{args.exp_id}/rollouts/{result_path}"),ssp = "ssp370", perturbed_variable=perturbed_variable, perturbed_type=perturbed_type, T=rollout_steps,coordinates=coordinates)
        