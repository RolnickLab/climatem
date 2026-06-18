# This is a script to run a particle filtering rollout for a Hierarchical model.
# We can choose the number of timesteps, and what we want to filter for.
# Be careful with the number of batches we use for calculating the true data spectra.

# hack to go a couple of directories up if we need to import from python files in some parent directory.

import os
from pathlib import Path

import json

import numpy as np
import torch
from tqdm import trange
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from latent_analysis import PerturbHierarchicalLatentTSDCD
from climatem.rollouts.bayesian_filter import calculate_fft_mean_std_across_all_noresm, logscore_the_samples_for_spatial_spectra_bayesian
from climatem.config import *
from climatem.utils import parse_args, update_config_withparse

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")

# select 16 random samples from the batch
def sample_from_tensor_reproducibly(tensor1, tensor2, num_samples, seed=5):
    if num_samples > tensor1.shape[0]:
        raise ValueError("Number of samples cannot exceed the tensor's first dimension.")

    torch.manual_seed(seed)  # Set the random seed
    indices = torch.randperm(tensor1.shape[0])[:num_samples]
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
            reload_climate_set_data=data_params.reload_climate_set_data,
            icosahedral_coordinates_path=data_params.icosahedral_coordinates_path,
            # Below SAVAR data arguments
            time_len=savar_params.time_len,
            comp_size=savar_params.comp_size,
            noise_val=savar_params.noise_val,
            n_per_col=savar_params.n_per_col,
            difficulty=savar_params.difficulty,
            seasonality=savar_params.seasonality,
            overlap=savar_params.overlap,
            is_forced=savar_params.is_forced,
            plot_original_data=savar_params.plot_original_data,
        )
        datamodule.setup()

    d = len(data_params.in_var_ids)
    print(f"Using {d} variables")

    if model_params.instantaneous:
        print("Using instantaneous connections")
        num_input = d * (experiment_params.tau + 1) 
    else:
        num_input = d * experiment_params.tau 

    # set the model
    model = PerturbHierarchicalLatentTSDCD(
        num_layers=model_params.num_layers,
        num_hidden=model_params.num_hidden,
        num_input=num_input,
        num_output=2,  # This should be parameterized somewhere?
        num_layers_mixing=model_params.num_layers_mixing,
        num_hidden_mixing=model_params.num_hidden_mixing,
        position_embedding_dim=model_params.position_embedding_dim,
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
        d_z_global=experiment_params.d_z_global,
        tau=experiment_params.tau,
        instantaneous=model_params.instantaneous,
        nonlinear_dynamics=model_params.nonlinear_dynamics,
        nonlinear_mixing=model_params.nonlinear_mixing,
        # no_gt=gt_params.no_gt,
        # debug_gt_graph=gt_params.debug_gt_graph,
        # debug_gt_z=gt_params.debug_gt_z,
        # debug_gt_w=gt_params.debug_gt_w,
        # gt_w=data_loader.gt_w,
        # gt_graph=data_loader.gt_graph,
        tied_w=model_params.tied_w,
        # also
        fixed=model_params.fixed,
        fixed_output_fraction=model_params.fixed_output_fraction,
    )
    
    # read paths 
    coordinates = np.load(data_params.icosahedral_coordinates_path)
    nino34 = []
    for i in trange(0, coordinates.shape[0]):
        if coordinates[i, 0] >= -170+180 and coordinates[i, 0] <= -120+180 and coordinates[i, 1] >= -5 and coordinates[i, 1] <= 5:
            nino34.append(i)

    

    exp_path = Path(experiment_params.exp_path)
    if not os.path.exists(exp_path): 
        raise ValueError(f"Results path {exp_path} doesn't exist. Model should be saved in this folder")
    
    data_var_ids_str = (
        str(data_params.in_var_ids)[1:-1]
        .translate({ord("'"): None})
        .translate({ord(","): None})
        .translate({ord(" "): None})
    )

    name = exp_id
    # name = f"var_{data_var_ids_str}_scen_{data_params.train_scenarios[0]}_nlinmix_{model_params.nonlinear_mixing}_nlindyn_{model_params.nonlinear_dynamics}_tau_{experiment_params.tau}_z_{experiment_params.d_z}_futt_{experiment_params.future_timesteps}_ldec_{optim_params.loss_decay_future_timesteps}_lr_{train_params.lr}_bs_{data_params.batch_size}_ormuin_{optim_params.ortho_mu_init}_spmuin_{optim_params.sparsity_mu_init}_spth_{optim_params.sparsity_upper_threshold}_nens_{data_params.num_ensembles}_inst_{model_params.instantaneous}_crpscoef_{optim_params.crps_coeff}_sspcoef_{optim_params.spectral_coeff}_tspcoef_{optim_params.temporal_spectral_coeff}_frachiwn_{optim_params.fraction_highest_wavenumbers}_nummix_hid_{model_params.num_hidden_mixing}_{model_params.num_hidden}_embdim_{model_params.position_embedding_dim}_trparamsh_{model_params.transition_param_sharing}_posembdimtr_{model_params.position_embedding_transition}"
#     name = f"var_{data_var_ids_str}_scenarios_{data_params.train_scenarios[0]}_nonlinear_{model_params.nonlinear_mixing}_tau_{experiment_params.tau}_z_{experiment_params.d_z}_lr_{train_params.lr}_bs_{data_params.batch_size}_spreg_{optim_params.reg_coeff}_ormuinit_{optim_params.ortho_mu_init}_spmuinit_{optim_params.sparsity_mu_init}_spthres_{optim_params.sparsity_upper_threshold}_fixed_{model_params.fixed}_num_ensembles_{data_params.num_ensembles}_instantaneous_{model_params.instantaneous}_crpscoef_{optim_params.crps_coeff}_spcoef_{optim_params.spectral_coeff}_tempspcoef_{optim_params.temporal_spectral_coeff}"

    exp_path = exp_path / name
    print("exp_path experiment_params.exp_path:", exp_path)
    if not os.path.exists(exp_path): 
        raise ValueError(f"Results path {exp_path} does not exist. Are you using the same parameters?")

    # create path to exp and save hyperparameters
    save_path = exp_path / "rollouts"
    os.makedirs(save_path, exist_ok=True)

    # seed = 1
    save_path = save_path / f"bs_{rollout_params.batch_size}_np_{rollout_params.num_particles}_npp_{rollout_params.num_particles_per_particle}_t_{rollout_params.num_timesteps}_sc_{rollout_params.score}_temp_{rollout_params.tempering}_iter{iter_id}-perturbed"
    os.makedirs(save_path, exist_ok=True)

    

    model_path = exp_path #/ "training_results"

    y_true_fft_mean, y_true_fft_std = calculate_fft_mean_std_across_all_noresm(datamodule, accelerator)
    print("y_true_fft_mean shape:", y_true_fft_mean.shape)
    print("y_true_fft_std shape:", y_true_fft_std.shape)

    train_dataloader = iter(datamodule.train_dataloader(accelerator))
    x, y = next(train_dataloader)

    if rollout_params.final_30_years_of_ssps:
        print("Taking the final 30 years of the SSP data, ~ 2070-2100")
        x, y = next(train_dataloader)
        x, y = next(train_dataloader)


    x = torch.nan_to_num(x)
    y = torch.nan_to_num(y)
    y = y[:, 0]
    z = None

    x = x.to(device)
    y = y.to(device)

    # Here we load a final model, when we do learn the causal graph. Make sure  it is on GPU:
    # model_file = f"model_{iter_id}.pth"
    # model_path = exp_path
    model_file = "model.pth"
    print(f"The model being teseted is under: {model_path / model_file}")
    state_dict_vae_final = torch.load(
        model_path / model_file, 
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

    with torch.no_grad():
        final_picontrol_particles = particle_filter_weighting_bayesian_perturbed(
            model,
            x_samples,
            y_samples,
            y_true_fft_mean,
            y_true_fft_std,
            coordinates,
            num_particles=rollout_params.num_particles,
            num_particles_per_particle=rollout_params.num_particles_per_particle,
            timesteps=rollout_params.num_timesteps,
            score=rollout_params.score,
            save_dir=save_path,
            save_name=f"trajectory_iteration",
            batch_size=rollout_params.batch_size,
            tempering=rollout_params.tempering,
            sample_trajectories=rollout_params.sample_trajectories,
            batch_memory=rollout_params.batch_memory,
            perturbed_area=nino34
        )

    return final_picontrol_particles

def particle_filter_weighting_bayesian_perturbed(
    model,
    x,
    y,
    y_true_fft_mean,
    y_true_fft_std,
    coordinates,
    num_particles: int = 100,
    num_particles_per_particle: int = 10,
    timesteps: int = 120,
    score: str = "variance",
    save_dir: str = None,
    save_name: str = None,
    batch_size: int = 16,
    tempering: bool = False,
    sample_trajectories: bool = False,
    batch_memory: bool = False,
    perturbed_area= None,
):
    """
    Implement a particle filter to make a set of autoregressive predictions, where each created sample is evaluated by
    some score, and we do a particle filter to select only best samples to continue the autoregressive rollout.

    We need to pass the directory to save stuff to, and the stem of the filenames...
    if batch_memory: will loop over initial conditions (batch_size)
    else: no loop, faster but much higher memory

    TODO: REMOVE FOR LOOP OVER BATCH - torch/model can deal with the additional row?
    TODO: Code is quite confusing because here x is latent and z is reconstruction + y is fixed obs corresponding to FFT
    """

    print("Initial number of particles:", num_particles)
    perturbed_values = []

    for iter in trange(timesteps):

        # Prediction
        # make all the new predictions, taking samples from the latents

        if iter == 0:
#             print("This is the first timestep, so I am going to generate samples from the initial latents.")
            if score == "log_bayesian":
                print(f"x shape {x.shape}") # (8, 5, 1, 3072)
                print(f"y shape {y.shape}") #(8, 1, 3072)

                if not batch_memory:
                    unused_samples_from_xs, samples_from_zs, y, logscore_samples_fromzs, pz2_mu, perturbed_value = (
                        model.predict_sample_bayesianfiltering_perturbed_x(
                            x, y, iter, num_particles * num_particles_per_particle, with_zs_logprob=True, perturbed_area=perturbed_area
                        )
                    )
                    print(perturbed_value.shape)
                    print("perturbed_value.detach().cpu().mean()",perturbed_value.detach().cpu().mean().shape)
                    perturbed_values.append(perturbed_value.detach().cpu().mean())
                    torch.cuda.empty_cache()
                    logscore_samples_fromzs = torch.sum(logscore_samples_fromzs, -1).squeeze(2)
                    if tempering: 
                        logscore_samples_fromzs /= np.sqrt(model.d_z)
                else:
                    batch_size = x.shape[0]
                    samples_from_zs = []
                    logscore_samples_fromzs = []
                    pz2_mu = []
                    for k in range(batch_size):
                        unused_samples_from_xs, samples_from_zs_batch, unused_y, logscore_samples_fromzs_batch, pz2_mu_batch, perturbed_value= (
                            model.predict_sample_bayesianfiltering_perturbed_x(
                                x[k][None], y[k][None], iter, num_particles * num_particles_per_particle, with_zs_logprob=True,perturbed_area=perturbed_area
                            )
                        )
                        perturbed_values.append(perturbed_value.detach().cpu().mean())
                        # samples_from_zs_batch ([500, 1, 1, 3072]) pz2_mu_batch torch.Size([1, 1, 1])
                        torch.cuda.empty_cache()
                        logscore_samples_fromzs_batch = torch.sum(logscore_samples_fromzs_batch, -1).squeeze(2)
                        if tempering: 
                            logscore_samples_fromzs_batch /= np.sqrt(model.d_z)
                        samples_from_zs.append(samples_from_zs_batch)
                        logscore_samples_fromzs.append(logscore_samples_fromzs_batch)
                        pz2_mu.append(pz2_mu_batch)
                    samples_from_zs = torch.cat(samples_from_zs, dim=1) #(500,8,1,3072)
                    logscore_samples_fromzs = torch.cat(logscore_samples_fromzs, dim=-1)[None]
                    pz2_mu= torch.cat(pz2_mu, dim=0)[None] #(1,8,1,1)
            else:
                unused_samples_from_xs, samples_from_zs, y = model.predict_sample_bayesianfiltering_perturbed_x(
                    x, y, iter, num_particles * num_particles_per_particle, with_zs_logprob=False,perturbed_area=perturbed_area
                )

        else:
            assert x.ndim == 5
            assert y.ndim == 3
        
            x_reshaped = x.reshape((-1, x.shape[2], x.shape[3], x.shape[4]))
            y_reshaped = y.repeat(x.shape[0], 1, 1, 1).reshape((-1, y.shape[1], y.shape[2]))
            if score == "log_bayesian":
                if not batch_memory:
                    unused_samples_from_xs, samples_from_zs, y_reshaped, logscore_samples_fromzs, pz2_mu, perturbed_value = (
                        model.predict_sample_bayesianfiltering_perturbed_x(
                            x_reshaped, y_reshaped, iter, num_particles_per_particle, with_zs_logprob=True,perturbed_area=perturbed_area
                        )
                    ) # finds n_particles_per_particle * n_particles, here, for each k in n_particles the corresponding n_particles_per_particle are in [k, k+n_particles, ..., k+n_particles_per_particle*n_particles]
                    perturbed_values.append(perturbed_value.detach().cpu().mean())
                    torch.cuda.empty_cache()
                    logscore_samples_fromzs = torch.sum(logscore_samples_fromzs, -1).squeeze()
                    if tempering: 
                        logscore_samples_fromzs /= np.sqrt(model.d_z)
                    logscore_samples_fromzs = logscore_samples_fromzs.reshape((logscore_samples_fromzs.shape[0], x.shape[0], x.shape[1]))
                else:
                    samples_from_zs = []
                    logscore_samples_fromzs = []
                    pz2_mu=[]
                    for k in range(batch_size):
                        unused_samples_from_xs, samples_from_zs_batch, unused_y_reshaped, logscore_samples_fromzs_batch, pz2_mu_batch, perturbed_value = (
                            model.predict_sample_bayesianfiltering_perturbed_x(
                                x[:, k], y[k].repeat(x.shape[0], 1, 1), iter, num_particles_per_particle, with_zs_logprob=True,perturbed_area=perturbed_area
                            )
                        ) # finds n_particles_per_particle * n_particles, here, for each k in n_particles the corresponding n_particles_per_particle are in [k, k+n_particles, ..., k+n_particles_per_particle*n_particles]
                        # samples_from_zs_batch [10, 50, 3072]-> will select the best one particle out of 10
                        # pz2_mu_batch [50, 1, 1]
                        perturbed_values.append(perturbed_value.detach().cpu().mean())
                    
                        torch.cuda.empty_cache()
                        logscore_samples_fromzs_batch = torch.sum(logscore_samples_fromzs_batch, -1).squeeze()
                        if tempering: 
                            logscore_samples_fromzs_batch /= np.sqrt(model.d_z)
                        logscore_samples_fromzs_batch = logscore_samples_fromzs_batch.reshape((logscore_samples_fromzs_batch.shape[0], x.shape[0]))

                        logscore_samples_fromzs.append(logscore_samples_fromzs_batch[:, None])
                        samples_from_zs.append(samples_from_zs_batch[:, :, None])
                        pz2_mu.append(pz2_mu_batch[:, None])
                    samples_from_zs = torch.cat(samples_from_zs, dim=2) # shape (10, 50, 8, 1, 3072)
                    logscore_samples_fromzs = torch.cat(logscore_samples_fromzs, dim=-1)
                    logscore_samples_fromzs = logscore_samples_fromzs.reshape((-1, x.shape[0], x.shape[1]))
                    pz2_mu = torch.cat(pz2_mu, dim=1)

            else:
                samples_from_zs, y, unused_z, unused_pz_mu, unused_pz_std, pz2_mu = model.predict(x_reshaped, y_reshaped)
            
            if not batch_memory: 
                samples_from_zs = samples_from_zs.reshape((samples_from_zs.shape[0], x.shape[0], x.shape[1], samples_from_zs.shape[2], samples_from_zs.shape[3]))

        if score == "spatial_spectra":
            new_weights = score_the_samples_for_spatial_spectra(
                y,
                samples_from_zs,
                coords=coordinates,
                num_particles=num_particles * num_particles_per_particle,
                mid_latitudes=True,
            )
        elif score == "log_bayesian":
            if iter > 0:
                # In correct dimension?? should be
                logscore_samples_fromzs = torch.flatten(logscore_samples_fromzs, start_dim=0, end_dim=1)
                samples_from_zs = torch.flatten(samples_from_zs, start_dim=0, end_dim=1)
            scores_spatial_spectra = logscore_the_samples_for_spatial_spectra_bayesian(
                y_true_fft_mean,
                y_true_fft_std,
                samples_from_zs,
                coords=coordinates,
                num_particles=num_particles * num_particles_per_particle,
                batch_size=batch_size,
                tempering=tempering,
            )
            new_weights = logscore_samples_fromzs + scores_spatial_spectra
            if new_weights.ndim == 3:
                new_weights = new_weights[0]
        else:
            raise ValueError("Score must be either variance or spatial_spectra")

        max_weight = torch.max(new_weights, dim=0)
        if score != "log_bayesian":
            min_weight = torch.min(new_weights, dim=0)
        else:
            # might get overflows here - might need to clip...for torch.exp
            new_weights = torch.exp(new_weights - max_weight.values)
        new_weights = new_weights / torch.sum(new_weights, dim=0)
        # clip the new_weights to avoid numerical instability
        new_weights = torch.clamp(new_weights, min=1e-8, max=1.0)
        new_weights = new_weights / torch.sum(new_weights, dim=0)

        if not sample_trajectories:
            resampled_indices = torch.multinomial(new_weights.T, num_particles, replacement=True).T
        else:
            # Here, every num_particles_per_particle we should sample one i.e. we track each trajectory
            resampled_indices = torch.zeros([num_particles, batch_size], dtype = torch.long)
            for k in range(num_particles):
                idx_trajectory = torch.arange(k, k+(num_particles_per_particle)*num_particles, num_particles)
                resampled_indices[k] = idx_trajectory[new_weights[idx_trajectory].argmax(0)]


        selected_samples = samples_from_zs[resampled_indices, torch.arange(batch_size)]
        if iter == 0:
            pz2_mu = pz2_mu.repeat(num_particles, 1, 1, 1)
        # selected_samples_global = pz2_mu[resampled_indices, torch.arange(batch_size)]
        np.save(save_dir / f"{save_name}_{iter}.npy", selected_samples.detach().cpu().numpy())
        np.save(save_dir / f"{save_name}_{iter}_global.npy", pz2_mu.detach().cpu().numpy())

        if iter == 0:
            x = x.repeat(num_particles, 1, 1, 1, 1)

        x = x[:, :, 1:, :, :]

        # now we just need to unsqueeze the selected samples, so that we can concatenate them to x
        selected_samples = selected_samples.unsqueeze(2)
        # pz2_mu will not be reused for rollout

        x = torch.cat([x, selected_samples], dim=2)
        # then we are going back to the top of the loop
    torch.save(perturbed_values, "perturbed_values.pt")
    return selected_samples

if __name__ == "__main__":

    args = parse_args()
    
    cwd = Path.cwd()
    root_path = cwd.parent
    # config_path = root_path / f"configs"
    exp_id = args.exp_id
    iter_id = args.iter_id
    folder = exp_id.split("/")[0] 
    exp_id = exp_id.split("/")[-1]
    config_path = Path("/home/mila/s/shanz/scratch/results") / folder / exp_id
    json_path = config_path / args.config_path
    
    with open(json_path, "r") as f:
        params = json.load(f)
    params = update_config_withparse(params, args)

    # get user's scratch directory:
    scratch_path = os.getenv("SCRATCH")
    params["data_params"]["data_dir"] = params["data_params"]["data_dir"].replace("$SCRATCH", scratch_path)
    print ("new data path:", params["data_params"]["data_dir"])

    params["exp_params"]["exp_path"] = params["exp_params"]["exp_path"].replace("$SCRATCH", scratch_path)
    print ("new exp path:", params["exp_params"]["exp_path"])

    # get directory of project via current file (aka .../climatem/scripts/main_picabu.py)
    params["data_params"]["icosahedral_coordinates_path"] = params["data_params"]["icosahedral_coordinates_path"].replace("$CLIMATEMDIR", root_path.absolute().as_posix())
    print ("new icosahedron path:", params["data_params"]["icosahedral_coordinates_path"])
    
    # For rollout, most cases we already have the climate dataset during training
    params["data_params"]["reload_climate_set_data"] = True
    print ("new reload_climate_set_data:", params["data_params"]["reload_climate_set_data"])

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

    final_picontrol_particles = main(experiment_params, data_params, train_params, model_params, optim_params, plot_params, savar_params, rollout_params, exp_id, iter_id)

