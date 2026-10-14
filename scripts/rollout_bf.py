# This is a script to run a particle filtering rollout for a model.
# We can choose the number of timesteps, and what we want to filter for.
# Be careful with the number of batches we use for calculating the true data spectra.

# hack to go a couple of directories up if we need to import from python files in some parent directory.
# 07/10: plot the last step or first step global forcing?
# uncomment row 93 for showing the x4 warming up!
import os
from pathlib import Path

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from climatem.data_loader.causal_datamodule import CausalClimateDataModule,CausalClimateDataMultiScenarioModule,CausalClimateDataMultiScaleScenarioModule
from climatem.model.tsdcd_latent import LatentTSDCD
from climatem.model.tsdcd_latent_hierarchy_raw import HierarchicalLatentTSDCD as HVAE
from climatem.rollouts.bayesian_filter import calculate_fft_mean_std_across_all_noresm, particle_filter_weighting_bayesian_hierachical
from climatem.config import *
from climatem.utils import parse_args, update_config_withparse

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")
plt.rcParams.update({
    "font.size": 16,       
    "axes.titlesize": 18,  
    "axes.labelsize": 16,  
    "xtick.labelsize": 14,  
    "ytick.labelsize": 14,  
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

def get_all_y_ssp(datamodule, accelerator):

    # Start again at the beginning of the dataloader.
    test_dataloader = iter(datamodule.test_dataloader(accelerator))

    # iterate through the data and append all the y values together
    y_all = []
    y_co2_all = []
    y_aerosol_all = []
    y_ch4_all = []
    y_so2_all = []    
    for i in range(len(test_dataloader)):
        batch = next(test_dataloader)
        if isinstance(batch, dict):
            # New format with forcings
            y_whole_dataloader = batch["y"]
            y_co2_dataloader = batch["global_forcings"][:, :, 0] #(b, tau, n, 3072)
            y_aerosol_whole_dataloader = batch["spatial_forcings"][:, :, 0]
            y_ch4_dataloader = batch["global_forcings"][:, :, 1] #(b, tau, n, 3072)
            y_so2_whole_dataloader = batch["spatial_forcings"][:, :, 1]
        else:
            # Legacy format (tuple)
            _, y_whole_dataloader = batch
        y_all.append(y_whole_dataloader[:,0][None])
        y_co2_all.append(y_co2_dataloader[None])
        y_aerosol_all.append(y_aerosol_whole_dataloader[None])
        y_ch4_all.append(y_ch4_dataloader[None])
        y_so2_all.append(y_so2_whole_dataloader[None])

    y_all = torch.cat(y_all, dim=0)
    y_co2_all = torch.cat(y_co2_all, dim=0)
    y_aerosol_all = torch.cat(y_aerosol_all,dim=0)
    y_ch4_all = torch.cat(y_ch4_all, dim=0)
    y_so2_all = torch.cat(y_so2_all,dim=0)

    y_all = torch.nan_to_num(y_all)
    y_co2_all = torch.nan_to_num(y_co2_all)
    y_aerosol_all = torch.nan_to_num(y_aerosol_all)
    y_ch4_all = torch.nan_to_num(y_ch4_all)
    y_so2_all = torch.nan_to_num(y_so2_all)
    print("y_all", y_all.shape, y_co2_all.shape, y_aerosol_all.shape)

    # make sure we reset the dataloader
    test_dataloader = iter(datamodule.test_dataloader(accelerator))

    return y_all, y_co2_all, y_aerosol_all, y_ch4_all, y_so2_all

def get_perturbed_forcings(datamodule, accelerator, 
                      perturbation_start=400, 
                      increase_factor=2.0,
                      perturbation_end=412,
                      perturbation_type="sinusoidal",
                      apply_to="co2"):

    # Start again at the beginning of the dataloader.
    test_dataloader = iter(datamodule.test_dataloader(accelerator))

    # iterate through the data and append all the y values together
    y_co2_all = []
    y_aerosol_all = []
    
    for i in range(len(test_dataloader)):
        batch = next(test_dataloader)
        if isinstance(batch, dict):
            # New format with forcings
            y_co2_dataloader = batch["co2_forcing"] #(b, tau, 1, 1)
            y_aerosol_whole_dataloader = batch["aerosol_forcing"]#(b, tau, 1, 1)
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
        if apply_to in ["co2", "both"]:
            y_co2_perturbed[perturbation_start:perturbation_end] = 0

        if apply_to in ["aerosol", "both"]:
            y_aerosol_perturbed[perturbation_start:perturbation_end] = 0
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
    # indices = torch.randperm(tensor1.shape[0])[:num_samples]
    indices = torch.arange(tensor1.shape[0])[:num_samples]
    return tensor1[indices], tensor2[indices]

def main(
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
    rollout_steps,
    selection,
    
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
            global_lag= data_params.global_lag
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
        global_lag = getattr(data_params, "global_lag", 0)
        DATA_REGISTRY = {
            "single": {
                "cls": CausalClimateDataModule,
                "args": {**common_args, **savar_args},
            },
            "multi": { # the multiscenario dataset doesn't support savar input yet.
                "cls": (
                    CausalClimateDataMultiScaleScenarioModule
                    if global_lag > 0
                    else CausalClimateDataMultiScenarioModule),
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
    ssp = common_args["test_scenarios"][0]

    if model_params.instantaneous:
        print("Using instantaneous connections")
        num_input = d * (experiment_params.tau + 1) 
    else:
        num_input = d * experiment_params.tau 

    # set the model
    common_model_kwargs = dict(
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
        n_forced_latents_so2=model_params.n_forced_latents_so2,
        map_aerosol_to_climate = model_params.map_aerosol_to_climate,
        forcing_mse = model_params.forcing_mse,
    )


    class HierarchicalLatentTSDCD(HVAE):
        # For the unperturbed version, we simply select randomly/logscore from pz

        def __init__(
            self,
            selection,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.selection = selection
        def predict_sample_bayesianfiltering(self, x, y, num_samples, with_zs_logprob: bool = False, y_co2=None, y_aerosol=None, y_ch4=None, y_so2=None):
            """
            This is a prediction function for the model, but where we take samples from the Gaussians of the latents.

            Note this function also returns the option where we sample from the decoders, but of course these samples are
            just chequerboards and not very interesting.

            I can use no_grad here, because I am not going to be using the gradients for anything.
            """

            b = x.size(0)

            with torch.no_grad():
                # sample Zs (based on X)
                z, q_mu_y, q_std_y = self.encode(x, y)
                z2, q_mu_z2, q_std_z2 = self.encode_global(z)

                # get params of the transition model p(z^t | z^{<t})
                mask = self.mask(b)
                if self.use_exogenous:
                    # Take the last step forcing as condition
                    if y_aerosol is not None:
                        y_aerosol = y_aerosol[:,-1]
                    if y_so2 is not None:
                        y_so2 = y_so2[:,-1]
                    # if y_co2 is not None:
                    #     y_co2 = y_co2[:,-1]
                    # if y_ch4 is not None:
                    #     y_ch4 = y_ch4[:,-1]

                if self.instantaneous:
                    pz2_mu, pz2_std = self.transition_global(z2[:, :-1].clone(), y_co2=y_co2.repeat(b,1,1), y_ch4=y_ch4.repeat(b,1,1))
                    pz_mu, pz_std = self.transition(z.clone(), pz2_mu, mask, y_aerosol=y_aerosol.repeat(b,1), y_so2=y_so2.repeat(b,1))
                else:
                    pz2_mu, pz2_std = self.transition_global(z2[:, :-1].clone(), y_co2=y_co2.repeat(b,1,1), y_ch4=y_ch4.repeat(b,1,1))  # (b,1,d_z2)
                    pz_mu, pz_std = self.transition(z[:, :-1].clone(), pz2_mu, mask, y_aerosol=y_aerosol.repeat(b,1), y_so2=y_so2.repeat(b,1))

                dim = pz_mu.ndim
                new_shape = [num_samples]
                for k in range(dim):
                    new_shape.append(1)
                z_samples = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).sample()
                #     print("z_samples_logprob 1",z_samples_logprob.shape) #([10, 50, 1, dz])
                    # GMST of each decoded particle 

                    # self.distr_transition(pz_mu, pz_std).log_prob(z_samples[i]) gives log probability
                samples_from_zs, some_decoded_samples_std = self.decode(
                    z_samples.reshape(z_samples.size(0) * z_samples.size(1), z_samples.size(2), z_samples.size(3))
                )
                samples_from_zs = samples_from_zs.reshape(z_samples.size(0), z_samples.size(1), z_samples.size(2), self.d_x)
                # some_decoded_samples_mu, some_decoded_samples_std = self.decode(z_samples[i])

                # samples_from_zs[i] = some_decoded_samples_mu
                if with_zs_logprob: 
                    if self.selection == "pz2mu":
                        gmst_samples = samples_from_zs.mean(dim=-1) 
                        # shape: (num_traj, num_samples, b, d) [10,50,1,1]

                        # Global transition mean, trained to align with GMST
                        gmst_target = pz2_mu.mean(-1)#([50, 1, 1]) ([50, 1])
  
                        z_samples_logprob = -(
                            gmst_samples - gmst_target.unsqueeze(0)
                        ).pow(2).unsqueeze(-1)#.mean(dim=-1) ([10, 50, 1, 1]) 
                    else:
                        # for random and logscore, we can simply return the score, but random will not use this score
                        z_samples_logprob = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).log_prob(
                            z_samples 
                        ) 

                # decode
                px_mu, px_std = self.decode(pz_mu.unsqueeze(2))
                px_mu = px_mu.squeeze(2)
                px_std = px_std.squeeze(2)

                dim = px_mu.ndim
                new_shape = [num_samples]
                for k in range(dim):
                    new_shape.append(1)
                # here we decode from pz_mu, and then sample from the distribution over xs.
                # note this will simply give us chequerboards.
                samples_from_xs = torch.zeros(num_samples, b, self.d, self.d_x)
                samples_from_xs = self.distr_decoder(px_mu.repeat(new_shape), px_std.repeat(new_shape)).sample()

            if with_zs_logprob:
                return samples_from_xs, samples_from_zs, y, z_samples_logprob, pz2_mu
            return samples_from_xs, samples_from_zs, y    
    match experiment_params.d_z_global:
        case 0:
            model = LatentTSDCD(**common_model_kwargs)
        case _:
            model = HierarchicalLatentTSDCD(
                **common_model_kwargs,
                d_z_global=experiment_params.d_z_global,
                nonlinear_global_dynamics=model_params.nonlinear_global_dynamics,
                selection=selection
            )
    # read paths 
    coordinates = np.load(data_params.icosahedral_coordinates_path)

    exp_path = Path(experiment_params.exp_path)
    if not os.path.exists(exp_path): 
        raise ValueError(f"Results path {exp_path} doesn't exist. Model should be saved in this folder")
    
    
    name = exp_id
    exp_path = exp_path / name
    if not os.path.exists(exp_path): 
        raise ValueError(f"Results path {exp_path} does not exist. Are you using the same parameters?")

    # create path to exp and save hyperparameters
    save_path = exp_path / "rollouts"
    os.makedirs(save_path, exist_ok=True)

    total_time_steps = rollout_steps#len(test_dataloader)

    # seed = 1
    save_path = save_path / f"bs_{rollout_params.batch_size}_np_{rollout_params.num_particles}_npp_{rollout_params.num_particles_per_particle}_t_{total_time_steps}_sc_{selection}_temp_{rollout_params.tempering}_iter{iter_id}_{ssp}"
    os.makedirs(save_path, exist_ok=True)
    
    if str(iter_id) == str(train_params.max_iteration):
        model_path = exp_path
        post_name = "model.pth"
    else:
        model_path = exp_path / "training_results"
        post_name = f"model_{iter_id}.pth"

    y_true_fft_mean, y_true_fft_std = calculate_fft_mean_std_across_all_noresm(datamodule, accelerator)
    print("y_true_fft_mean shape:", y_true_fft_mean.shape)
    print("y_true_fft_std shape:", y_true_fft_std.shape)

    test_dataloader = iter(datamodule.test_dataloader(accelerator))
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
            y_co2 = y_global_focings[:, :, 0]
            y_aerosol = y_spatial_forcings[:, :, 0]
            y_ch4 = y_global_focings[:, :, 1]
            y_so2 = y_spatial_forcings[:, :, 1]
    else:
        x, y = batch
        y_co2, y_aerosol, y_ch4, y_so2 = None, None, None, None


    if y_co2 is not None:
        y_co2 = torch.nan_to_num(y_co2).to(device)
    if y_aerosol is not None:
        y_aerosol = torch.nan_to_num(y_aerosol).to(device)
    if y_ch4 is not None:
        y_ch4 = torch.nan_to_num(y_ch4).to(device)
    if y_so2 is not None:
        y_so2 = torch.nan_to_num(y_so2).to(device)

    x = torch.nan_to_num(x)
    y = torch.nan_to_num(y)
    y = y[:, 0]
    z = None

    x = x.to(device)
    y = y.to(device)

    # Here we load a final model, when we do learn the causal graph. Make sure  it is on GPU:
    
    state_dict_vae_final = torch.load(
        model_path /  post_name, 
        map_location=device
    )
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict_vae_final.items()})

    # Move the model to the GPU
    model = model.to(device)
    print("Where is the model?", next(model.parameters()).device)

    # First call with the seed
    x_samples, y_samples = sample_from_tensor_reproducibly(x, y, rollout_params.batch_size)
    # x_samples, y_samples = x[:rollout_params.batch_size], y[:rollout_params.batch_size]
    x_samples = x_samples[:, 0:1].repeat(1, 5, 1, 1)
    np.save(
        save_path / "forpowerspectra_random1_batch_xs_we_start_with.npy",
        x_samples.detach().cpu().numpy(),
    )

    with torch.no_grad():
        thresholded_adj = (model.get_adj() > 0.5).type(torch.Tensor)
        model.mask.fix(thresholded_adj)
    gt_y = get_all_y_ssp(datamodule, accelerator)

    all_y, all_co2, all_aerosols, all_ch4, all_so2 = gt_y

    np.savez(
        save_path / f"gt_all_y_{ssp}.npz",
        all_y=all_y.detach().cpu().numpy(),
        all_co2=all_co2.detach().cpu().numpy(),
        all_aerosols=all_aerosols.detach().cpu().numpy(),
        all_ch4=all_ch4.detach().cpu().numpy(),
        all_so2=all_so2.detach().cpu().numpy(),
    )
    with torch.no_grad():
        final_picontrol_particles = particle_filter_weighting_bayesian_hierachical(
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
            all_y_ssp=gt_y,
            selection=selection
        )


    return final_picontrol_particles
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
            train_scenarios=["ssp126","historical"],
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

    # set the model
    common_model_kwargs = dict(
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
        n_forced_latents_so2=model_params.n_forced_latents_so2,
        map_aerosol_to_climate = model_params.map_aerosol_to_climate,
        forcing_mse = model_params.forcing_mse,
    )

    match experiment_params.d_z_global:
        case 0:
            model = LatentTSDCD(**common_model_kwargs)
        case _:
            model = HierarchicalLatentTSDCD(
                **common_model_kwargs,
                d_z_global=experiment_params.d_z_global,
                nonlinear_global_dynamics=model_params.nonlinear_global_dynamics,
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

    # seed = 1


    model_path = exp_path #/ "training_results"

    y_true_fft_mean, y_true_fft_std = calculate_fft_mean_std_across_all_noresm(datamodule, accelerator)
    print("y_true_fft_mean shape:", y_true_fft_mean.shape)
    print("y_true_fft_std shape:", y_true_fft_std.shape)

    test_dataloader = iter(datamodule.test_dataloader(accelerator))
    # total_time_steps = len(test_dataloader)
    save_path = save_path / f"bs_{rollout_params.batch_size}_np_{rollout_params.num_particles}_npp_{rollout_params.num_particles_per_particle}_t_{total_time_steps}_sc_{rollout_params.score}_temp_{rollout_params.tempering}_iter{iter_id}_{test_scenarios[0]}_perturbed"
    os.makedirs(save_path, exist_ok=True)
    
    batch = next(test_dataloader)

    if rollout_params.final_30_years_of_ssps:
        print("Taking the final 30 years of the SSP data, ~ 2070-2100")
        batch = next(test_dataloader)
        batch = next(test_dataloader)


    if isinstance(batch, dict):
        # New format with forcings
        x = batch["x"]
        y = batch["y"]
        y_co2 = batch.get("co2_forcing", None)
        y_aerosol = batch.get("aerosol_forcing", None)
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
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict_vae_final.items()})

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
    all_co2_perturbed, all_aerosol_perturbed = get_perturbed_forcings(datamodule, accelerator,apply_to=perturbed_variable, perturbation_type=perturbed_type)

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
        final_picontrol_particles = particle_filter_weighting_bayesian(
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
        final_picontrol_particles_perturbed = particle_filter_weighting_bayesian(
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

def load_rollouts_all(traj_path, ssps, selection):

    T = 1027
    # variables = ["ts", "co2", "aerosols", "ch4", "so2"]

    variables = ["ts"]#, "co2", "aerosols", "ch4", "so2"]

    # ---------------------------------------------------------
    # Global plotting style
    # ---------------------------------------------------------
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.2,
    })

    # Better distinguishable colors
    colors = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:orange",
        "tab:purple",
    ]

    if len(ssps) > len(colors):
        raise ValueError(
            f"Only {len(colors)} colors are defined, "
            f"but {len(ssps)} SSPs were provided."
        )

    # ---------------------------------------------------------
    # Load rollout predictions
    # ---------------------------------------------------------
    pred_data = {}

    for ssp in ssps:
        trajs = []

        for step in range(T):
            traj = np.load(
                f"{traj_path}_{ssp}/trajectory_iteration_{step}.npy"
            )
            # Expected shape: (n_ensemble, 1, 1, 3072)
            trajs.append(traj)

        print(
            f"A single trajectory for {ssp} has shape: "
            f"{traj.shape}"
        )

        trajs = np.stack(trajs, axis=0)

        print(
            f"Stacked trajectories for {ssp}: "
            f"{trajs.shape}"
        )

        # Expected:
        # (T, n_ensemble, 1, 1, n_pixels)
        pred_data[ssp] = trajs

    # ---------------------------------------------------------
    # Load ground truth and forcing variables
    # ---------------------------------------------------------
    gt_data = {}
    co2_data = {}
    aerosol_data = {}
    ch4_data = {}
    so2_data = {}

    for ssp in ssps:
        data = np.load(
            f"{traj_path}_{ssp}/gt_all_y_{ssp}.npz"
        )

        gt = data["all_y"]
        # Original expected shape: (T, 1, 1, 3072)
        gt = gt.squeeze(1)
        # Result: (T, 1, 3072)

        gt_data[ssp] = gt
        co2_data[ssp] = data["all_co2"]
        aerosol_data[ssp] = data["all_aerosols"]
        ch4_data[ssp] = data["all_ch4"]
        so2_data[ssp] = data["all_so2"]

    time_steps = pd.date_range(
        start="2015-01-01",
        periods=T,
        freq="MS",
    )

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        len(variables),
        1,
        figsize=(15, 6 * len(variables)),
        sharex=True,
    )

    if len(variables) == 1:
        axes = [axes]

    variable_labels = {
        "ts": "GMST",
        "co2": r"CO$_2$",
        "aerosols": "BC",
        "ch4": r"CH$_4$",
        "so2": r"SO$_2$",
    }

    # ---------------------------------------------------------
    # Plot each variable
    # ---------------------------------------------------------
    for var_name, ax in zip(variables, axes):

        for i, ssp in enumerate(ssps):
            color = colors[i]

            if var_name == "ts":

                # -------------------------------------------------
                # Ground-truth GMST
                # -------------------------------------------------
                gt = (
                    gt_data[ssp][:, 0, :]
                    .mean(axis=-1)[:T]
                )

                ax.plot(
                    time_steps,
                    gt,
                    color=color,
                    linestyle="-",
                    linewidth=1.7,
                    # alpha=0.30,
                    zorder=3,
                )

                # -------------------------------------------------
                # Predicted ensemble mean GMST
                # -------------------------------------------------
                trajectories = pred_data[ssp]

                # Spatial mean for every ensemble member:
                # (T, ensemble, 1, 1, pixels)
                # -> (T, ensemble, 1, 1)
                ensemble_gmst = trajectories.mean(axis=-1)

                # Mean over ensemble and remaining singleton dimensions
                pred = ensemble_gmst.mean(axis=(1, 2, 3))

                # Standard deviation over ensemble members.
                # First remove singleton dimensions:
                # (T, ensemble, 1, 1) -> (T, ensemble)
                ensemble_gmst = np.squeeze(
                    ensemble_gmst,
                    axis=(2, 3),
                )

                std = ensemble_gmst.std(
                    axis=1,
                    ddof=1,
                )

                ax.fill_between(
                    time_steps,
                    pred - 2.0 * std,
                    pred + 2.0 * std,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                    zorder=1,
                )

                ax.plot(
                    time_steps,
                    pred,
                    color=color,
                    linestyle="--",
                    linewidth=3.0,
                    alpha=0.6,
                    # markevery=24,
                    zorder=4,
                )

            elif var_name == "co2":

                co2 = (
                    co2_data[ssp][:, :, -1]
                    .mean(axis=-1)
                )

                ax.plot(
                    time_steps,
                    co2[:T],
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                    label=ssp.upper(),
                )

            elif var_name == "aerosols":

                aerosol = (
                    aerosol_data[ssp][:, :, -1]
                    .mean(axis=-1)
                )

                ax.plot(
                    time_steps,
                    aerosol[:T],
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                    label=ssp.upper(),
                )

            elif var_name == "ch4":

                ch4 = (
                    ch4_data[ssp][:, :, -1]
                    .mean(axis=-1)
                )

                ax.plot(
                    time_steps,
                    ch4[:T],
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                    label=ssp.upper(),
                )

            elif var_name == "so2":

                so2 = (
                    so2_data[ssp][:, :, -1]
                    .mean(axis=-1)
                )

                ax.plot(
                    time_steps,
                    so2[:T],
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                    label=ssp.upper(),
                )

        # ---------------------------------------------------------
        # Axis formatting
        # ---------------------------------------------------------
        ax.set_ylabel(
            variable_labels[var_name],
            fontweight="bold",
        )

        ax.grid(
            axis="both",
            linestyle="--",
            linewidth=0.7,
            alpha=0.25,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ---------------------------------------------------------
        # Legends
        # ---------------------------------------------------------
        if var_name == "ts":

            # Scenario-color entries
            scenario_handles = [
                Line2D(
                    [0],
                    [0],
                    color=colors[i],
                    linewidth=3,
                    linestyle="-",
                    label=ssp.upper(),
                )
                for i, ssp in enumerate(ssps)
            ]

            # Meaning of line styles
            style_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=2.4,
                    linestyle="-",
                    label="Ground truth",
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=2.0,
                    linestyle="--",
                    # marker="o",
                    label="Prediction",
                ),
                Patch(
                    facecolor="gray",
                    edgecolor="none",
                    alpha=0.18,
                    label=r"Prediction $\pm$ 2 SD",
                ),
            ]

            ax.legend(
                handles=scenario_handles + style_handles,
                loc="upper left",
                ncol=2,
                frameon=True,
                framealpha=0.9,
                handlelength=3.0,
                columnspacing=1.5,
            )

        else:
            ax.legend(
                loc="upper left",
                ncol=len(ssps),
                frameon=False,
                handlelength=2.5,
            )

    axes[-1].set_xlabel(
        "Time",
        fontweight="bold",
    )

    fig.align_ylabels(axes)

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------
    output_path = Path(
        f"{traj_path}_ssp370/rollouts_pred_{'_'.join(ssps)}_{selection}.svg"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        # dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved to {output_path}")

    plt.close(fig)

def load_rollouts_single(traj_path, ssp, selection):
    
    T= 1027

    variables = ["ts","z_global","co2","aerosols","ch4","so2"]

    
    pred_data = {} 
    trajs = []
    trajs_global = []
    for i in range(T):
        traj = np.load(f"{traj_path}_{ssp}/trajectory_iteration_{i}.npy")  # shape (50, 1, 1, 3072)
        trajs.append(traj)     
    print(f"a signle traj has the shape of {traj.shape}")
    trajs = np.stack(trajs, axis=0) 
    print("trajs",trajs.shape) 
    pred_data[ssp] = trajs
    for i in range(T):
        traj_global = np.load(f"{traj_path}_{ssp}/trajectory_iteration_{i}_global.npy")#*1.3174365+ 0.12984121  # shape (50, 1, 1, 3072)
        trajs_global.append(traj_global.mean((0,1)))     # -> (5, 3072) 
    print(f"a signle traj has the shape of {traj_global.shape}")
    trajs_global = np.stack(trajs_global, axis=0)  
    print("trajs_global",trajs_global.shape) 

    # Plot 1: plot each ssp in a single figure

    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green'] 
    ssps = ["ssp126", "ssp245", "ssp370", "abrupt-4xCO2"] 
    color = colors[ssps.index(ssp)]
    num_vars = len(variables)
    
    # ---------------- load GT ----------------
    gt_data = {}
    co2_data = {}
    aerosol_data = {}
    ch4_data = {}
    so2_data = {}
    
    data = np.load(f"{traj_path}_{ssp}/gt_all_y_{ssp}.npz")

    gt = data["all_y"]           # (1027, 1, 1, 3072)
    gt = gt.squeeze(1)  # (1027,1,3072)
    gt_data[ssp] = gt
    co2_data[ssp] = data["all_co2"]
    aerosol_data[ssp] = data["all_aerosols"]
    ch4_data[ssp] = data["all_ch4"]
    so2_data[ssp] = data["all_so2"]


    time_steps = pd.date_range(start="2015-01-01", periods=T, freq="MS")
    # ---------------- plotting ----------------
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 4 * len(variables)), sharex=True)

    if len(variables) == 1:
        axes = [axes]
    for v, ax in enumerate(axes):
        var_name = variables[v]
        if var_name == "ts":
            var_name_label= "GMST"
            # ===== ts：GT and pred =====
            gt = gt_data[ssp][:, 0, :].mean(axis=-1)[:T]
            pred =  np.mean(pred_data[ssp].mean(axis=(2,3,4)),1) 
            ax.plot(time_steps,pred, color=color, label=f"Prediction-{ssp}", alpha=0.7)
            ax.plot(time_steps, gt, color="black", label=f"GT-{ssp}", alpha=0.7)
                        
            std = pred_data[ssp].mean(-1).std(axis=(1,2,3)).squeeze()
            ax.fill_between(
                time_steps,
                pred - std ,
                pred +  std,
                color=color,
                alpha=0.3,
                linewidth=0,
                label='± SD'
            )
        elif var_name == "z_global":
            # ===== pz2_mu：pred and pred_pert =====
            var_name_label = "Global latent"
            pred_global = trajs_global[:, 0, :].mean(axis=-1)
            ax.plot(time_steps, pred_global, color="black", label=f"Global latent-{ssp}")
        elif var_name == "co2":
            # ===== CO2 =====
            var_name_label= "CO2"
            co2 = co2_data[ssp][:, :, -1].mean(axis=-1) # take the last time step
            ax.plot(time_steps, co2[:T], color="green", alpha=0.8, label=f"CO2-{ssp}")
            
        elif var_name == "aerosols":
            # ===== Aerosol =====
            var_name_label= "BC"
            aerosol = aerosol_data[ssp][:, :, -1].mean(axis=-1)# take the last time step
            ax.plot(time_steps, aerosol[:T], color="green", alpha=0.8, label=f"BC-{ssp}")
        elif var_name == "ch4":
            # ===== ch4 =====
            var_name_label= "CH4"

            ch4 = ch4_data[ssp][:, :, -1].mean(axis=-1)# take the last time step
            ax.plot(time_steps, ch4[:T], color="green", alpha=0.8, label=f"CH4-{ssp}")
        elif var_name == "so2":
            # ===== Aerosol =====
            var_name_label= "SO2"
            so2 = so2_data[ssp][:, :, -1].mean(axis=-1)# take the last time step
            ax.plot(time_steps, so2[:T], color="green", alpha=0.8, label=f"SO2-{ssp}")        
        ax.set_ylabel(var_name_label)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_xlabel("Time")
    axes[-1].set_xlabel("Time")

    plt.tight_layout()
    filename = f"{traj_path}_{ssp}/rollouts_pred_{ssp}_{selection}.png"
    
    plt.savefig(filename) # save to the last one
    print(f"Saved to {filename}")
    plt.close()
def load_rollouts_perturbed(traj_path, ssp, perturbed_variable, perturbed_type, T):

    variables = ["ts","co2","aerosols"]
    
    trajs = []
    perturbed_trajs = []
    for i in range(T):
        traj = np.load(f"{traj_path}_{ssp}_perturbed/trajectory_iteration_{i}.npy")  # shape (50, 1, 1, 3072)
        perturbed_traj = np.load(f"{traj_path}_{ssp}_perturbed/trajectory_iteration_perturbed_{i}.npy")
        trajs.append(traj.mean((0,1)))     # -> (5, 3072)
        perturbed_trajs.append(perturbed_traj.mean((0,1))) 
    print(f"a signle traj has the shape of {traj.shape}")
    trajs = np.stack(trajs, axis=0) 
    perturbed_trajs = np.stack(perturbed_trajs, axis=0) 
    print("trajs",trajs.shape) 


    num_vars = len(variables)
    
    # ---------------- load GT ----------------

    data = np.load(f"{traj_path}_{ssp}_perturbed/gt_all_forcings_{ssp}_perturbed.npz")

    gt = data["all_y"]           # (1027, 1, 1, 3072)
    gt = gt.squeeze(1)  # (1027,1,3072)
    gt_data = gt
    co2_data = data["all_co2"]
    aerosol_data = data["all_aerosols"]
    co2_data_perturbed = data["all_co2_perturbed"]
    aerosol_data_perturbed = data["all_aerosol_perturbed"]


    time_steps = pd.date_range(start="2015-01-01", periods=T, freq="MS")
    # ---------------- plotting ----------------
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 4 * len(variables)), sharex=True)

    if len(variables) == 1:
        axes = [axes]
    for v, ax in enumerate(axes):
        var_name = variables[v]
    
        if var_name == "ts":
            # ===== ts：GT and pred =====
            gt = gt_data[:, 0, :].mean(axis=-1)[:T]
            ax.plot(time_steps, gt, color="black", label=f"GT-{ssp}", alpha=0.7)
            
            pred = trajs[:, 0, :].mean(axis=-1)
            ax.plot(time_steps, pred, color="red", label=f"Prediction-{ssp}")

            pred_perturbed = perturbed_trajs[:, 0, :].mean(axis=-1)
            ax.plot(time_steps, pred_perturbed, color="green", label=f"Prediction-{ssp}-perturbed")
            
        elif var_name == "co2":
            # ===== CO2 =====
            co2 = co2_data[:, :, -1].mean(axis=-1)
            ax.plot(time_steps, co2[:T], color="black", label=f"CO2-{ssp}")
            
            co2_perturbed = co2_data_perturbed[:, :, -1].mean(axis=-1)
            ax.plot(time_steps, co2_perturbed[:T], color="green", label=f"CO2-{ssp}-perturbed")

        elif var_name == "aerosols":
            # ===== Aerosol =====
            aerosol = aerosol_data[:, :, -1].mean(axis=-1)
            ax.plot(time_steps, aerosol[:T], color="black", label=f"Aerosol-{ssp}")

            aerosol_perturbed = aerosol_data_perturbed[:, :, -1].mean(axis=-1)
            ax.plot(time_steps, aerosol_perturbed[:T], color="green", label=f"Aerosol-{ssp}-perturbed")
        
        ax.set_ylabel(var_name)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time")

    plt.tight_layout()
    filename = f"{traj_path}_{ssp}_perturbed/rollouts_pred_{ssp}_perturbed_{perturbed_variable}_{perturbed_type}.png"
    plt.savefig(filename) # save to the last one
    plt.close()
    print(f"Saved to {filename}")

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
    # corr_map = compute_corr_map(co2_data[ssp], aerosol_data[ssp])
    # plot_corr_map(corr_map,coordinates)
    corr_series = sliding_corr_10yr_mean(co2_data[ssp], aerosol_data[ssp])
    plot_sliding_corr(corr_series)
    filename = f"{traj_path}_ssp370/corr_series_{ssps}.png"
    
    plt.savefig(filename) # save to the last one
    print(f"Saved to {filename}")
    plt.close()

if __name__ == "__main__":
    PERTURBED = False

    args = parse_args()
    
    cwd = Path.cwd()
    root_path = cwd.parent
    config_path = root_path / f"configs"
    exp_id = args.exp_id
    iter_id = args.iter_id
     # args.test_scenarios
    
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

    #Overwrite arguments if using savar
    if "savar" in data_params.in_var_ids:
        experiment_params.lat = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.lon = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.d_x = int(experiment_params.lat * experiment_params.lon)
        plot_params.savar = True
    else:
        plot_params.savar = False
    rollout_steps = 1027
    selection = "pz2mu"
    total_scenarios = [["abrupt-4xCO2"]]#[["ssp370"],["ssp126"],["ssp245"],["abrupt-4xCO2"]]
    for test_scenarios in total_scenarios:
        result_path = f"bs_{rollout_params.batch_size}_np_{rollout_params.num_particles}_npp_{rollout_params.num_particles_per_particle}_t_{rollout_steps}_sc_{selection}_temp_{rollout_params.tempering}_iter{iter_id}"
        # check_corr_co2_aerosol(Path(f"{scratch_path}/results/{args.exp_id}/rollouts/{result_path}"),ssps = ["ssp370"])
        final_picontrol_particles = main(experiment_params, data_params, train_params, model_params, optim_params, plot_params, savar_params, rollout_params, exp_id, iter_id, test_scenarios,rollout_steps, selection)
        load_rollouts_single(Path(f"{scratch_path}/results/{args.exp_id}/rollouts/{result_path}"),ssp =test_scenarios[0], selection=selection)
    # load_rollouts_all(Path(f"{scratch_path}/results/{args.exp_id}/rollouts/{result_path}"),ssps =total_scenarios, selection=selection)