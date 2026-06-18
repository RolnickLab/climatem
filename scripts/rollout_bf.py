# This is a script to run a particle filtering rollout for a model.
# We can choose the number of timesteps, and what we want to filter for.
# Be careful with the number of batches we use for calculating the true data spectra.

# hack to go a couple of directories up if we need to import from python files in some parent directory.

import os
from pathlib import Path

import json

import numpy as np
import torch

from climatem.data_loader.causal_datamodule import CausalClimateDataModule, CausalClimateDataMultiScenarioModule
# from climatem.model.tsdcd_latent import LatentTSDCD
from climatem.rollouts.bayesian_filter import calculate_fft_mean_std_across_all_noresm, logscore_the_samples_for_spatial_spectra_bayesian, particle_filter_weighting_bayesian
from climatem.config import *
from climatem.utils import parse_args, update_config_withparse

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")
def get_all_y_ssp(datamodule, accelerator):

    # Start again at the beginning of the dataloader.
    test_dataloader = iter(datamodule.test_dataloader(accelerator))

    # iterate through the data and append all the y values together
    y_all = []
    for i in range(len(test_dataloader)):
        batch = next(test_dataloader)
        if isinstance(batch, dict):
            # New format with forcings
            y_whole_dataloader = batch["y"]
        else:
            # Legacy format (tuple)
            _, y_whole_dataloader = batch
        y_all.append(y_whole_dataloader[:,0][None])
    y_all = torch.cat(y_all, dim=0)
    y_all = torch.nan_to_num(y_all)
    print("y_all", y_all.shape)

    # make sure we reset the dataloader
    test_dataloader = iter(datamodule.test_dataloader(accelerator))

    return y_all
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
            test_scenarios=["ssp370"],
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

    d = len(data_params.in_var_ids)
    print(f"Using {d} variables")

    if model_params.instantaneous:
        print("Using instantaneous connections")
        num_input = d * (experiment_params.tau + 1) 
    else:
        num_input = d * experiment_params.tau 

    # set the model
        # set the model
    if experiment_params.d_z_global >= 1:
        from climatem.model.tsdcd_latent_hvae import LatentTSDCD as LatentTSDCDWithGLobalZ
        print(f"Using Hierarchical Model")
    
        class LatentTSDCD:
            def __init__(self, *args, **kwargs):
                self._model = LatentTSDCDWithGLobalZ(*args, d_z_global=experiment_params.d_z_global, **kwargs)
        
            def __getattr__(self, name):
                return getattr(self._model, name)
    else:
        print(f"Using SDCD (non-Hierarchical) Model")

        if len(data_params.train_scenarios)>1:
            from climatem.model.tsdcd_latent import LatentTSDCD as LatentTSDCDBase
            class LatentTSDCD(LatentTSDCDBase):

                def __init__(
                    self,
                    **kwargs
                ):

                    super().__init__(**kwargs)

                def predict_sample_bayesianfiltering(self, x, y, num_samples, with_zs_logprob: bool = False, use_gt_forcings= True):
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

                        # get params of the transition model p(z^t | z^{<t})
                        mask = self.mask(b)

                        if self.instantaneous:
                            pz_mu, pz_std = self.transition(z.clone(), mask)
                        else:
                            pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask)
                        # here I am taking the approach of sampling from the Z distributions, and then decoding.
                        #             samples_from_zs = torch.zeros(num_samples, b, self.d, self.d_x)
                        #             z_samples = torch.zeros(num_samples, b, self.d, self.d_z)
                        #             if with_zs_logprob:
                        #                 z_samples_logprob = torch.zeros(num_samples, b, self.d, self.d_z)

                        #             print(f"FOR LOOP MODEL num_samples {num_samples}")
                        #             print(f"z_samples.shape {z_samples.shape}")
                        #             print(f"pz_mu.shape {pz_mu.shape}")
                        #             print(f"pz_std.shape {pz_std.shape}")
                        dim = pz_mu.ndim
                        new_shape = [num_samples] #num_samples=500
                        for k in range(dim):
                            new_shape.append(1)
                        z_samples = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).sample()
                        # print("z_samples",z_samples.shape) z_samples [500, 1, 5, 60]
                        #             for i in trange(num_samples):
                        #                 #TODO: remove this FOR loop
                        #                 z_samples[i] = self.distr_transition(pz_mu, pz_std).sample()
                        #                 print(f"z_samples[i].shape {z_samples[i].shape}")

                        if with_zs_logprob:
                            z_samples_logprob = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).log_prob(
                                z_samples
                            )#(500, 1, 5, 60)

                            # self.distr_transition(pz_mu, pz_std).log_prob(z_samples[i]) gives log probability
                        t = z_samples.reshape(z_samples.size(0) * z_samples.size(1), z_samples.size(2), z_samples.size(3))

                        samples_from_zs, some_decoded_samples_std = self.decode(
                            z_samples.reshape(z_samples.size(0) * z_samples.size(1), z_samples.size(2), z_samples.size(3))
                        ) # (500, 5, 60) -> (500, 5, 3072)
                        samples_from_zs = samples_from_zs.reshape(z_samples.size(0), z_samples.size(1), z_samples.size(2), self.d_x) #(500, 1, 5, 3072)
                        # some_decoded_samples_mu, some_decoded_samples_std = self.decode(z_samples[i])

                        # samples_from_zs[i] = some_decoded_samples_mu

                        # decode
                        # if unsqueeze(1), then the 2nd dim is 1 and decoder loop over the 2nd dim resulting a wrong input of z
                        px_mu, px_std = self.decode(pz_mu.unsqueeze(2)) #pz_mu torch.Size([1, 5, 60])
                        # px_mu, px_std = self.decode(pz_mu)
                        px_mu = px_mu.squeeze(2)
                        px_std = px_std.squeeze(2)

                        dim = px_mu.ndim
                        new_shape = [num_samples]
                        for k in range(dim):
                            new_shape.append(1)
                        # here we decode from pz_mu, and then sample from the distribution over xs.
                        # note this will simply give us chequerboards.
                        samples_from_xs = torch.zeros(num_samples, b, self.d, self.d_x)

                        #             for i in range(num_samples):
                        samples_from_xs = self.distr_decoder(px_mu.repeat(new_shape), px_std.repeat(new_shape)).sample()
                        gt_expand = y.repeat(new_shape)
                    if use_gt_forcings and self.d>1:
                        samples_from_zs[:,:,1:] = gt_expand[:,:,1:]
                    if with_zs_logprob:
                        return samples_from_xs, samples_from_zs, y, z_samples_logprob, pz_mu.mean(-1, keepdim=True)
                    return samples_from_xs, samples_from_zs, y
        else:
            from climatem.model.tsdcd_latent import LatentTSDCD

    model = LatentTSDCD(
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

    exp_path = exp_path / name
    print("exp_path experiment_params.exp_path:", exp_path)
    if not os.path.exists(exp_path): 
        raise ValueError(f"Results path {exp_path} does not exist. Are you using the same parameters?")

    # create path to exp and save hyperparameters
    save_path = exp_path / "rollouts"
    os.makedirs(save_path, exist_ok=True)

    # seed = 1
    save_path = save_path / f"bs_{rollout_params.batch_size}_np_{rollout_params.num_particles}_npp_{rollout_params.num_particles_per_particle}_t_{rollout_params.num_timesteps}_sc_{rollout_params.score}_temp_{rollout_params.tempering}_iter{iter_id}"
    os.makedirs(save_path, exist_ok=True)

    

    model_path = exp_path #/ "training_results"

    y_true_fft_mean, y_true_fft_std = calculate_fft_mean_std_across_all_noresm(datamodule, accelerator)
    print("y_true_fft_mean shape:", y_true_fft_mean.shape)
    print("y_true_fft_std shape:", y_true_fft_std.shape)

    train_dataloader = iter(datamodule.test_dataloader(accelerator))
    batch = next(train_dataloader)
    if isinstance(batch, dict):
        # New format with forcings
        x = batch["x"]
        y = batch["y"]
    else:
        # Legacy format (tuple)
        x, y = batch

    if rollout_params.final_30_years_of_ssps:
        print("Taking the final 30 years of the SSP data, ~ 2070-2100")
        batch = next(train_dataloader)
        batch = next(train_dataloader)
        x, y = batch


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
    print(" x_samples, y_samples", x_samples.shape, y_samples.shape)
    np.save(
        save_path / "forpowerspectra_random1_batch_xs_we_start_with.npy",
        x_samples.detach().cpu().numpy(),
    )

    with torch.no_grad():
        thresholded_adj = (model.get_adj() > 0.5).type(torch.Tensor)
        model.mask.fix(thresholded_adj)
    gt_y = get_all_y_ssp(datamodule, accelerator)
    ssp=common_args["test_scenarios"][0]
    np.save(
        save_path / f"gt_all_y_{ssp}.npy",
        gt_y.detach().cpu().numpy(),
    )
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
            timesteps=rollout_params.num_timesteps,
            score=rollout_params.score,
            save_dir=save_path,
            save_name=f"trajectory_iteration",
            batch_size=rollout_params.batch_size,
            tempering=rollout_params.tempering,
            sample_trajectories=rollout_params.sample_trajectories,
            batch_memory=rollout_params.batch_memory,
            all_y_ssp=gt_y
        )

    return final_picontrol_particles



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

