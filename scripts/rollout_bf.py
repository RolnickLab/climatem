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
from climatem.rollouts.bayesian_filter import (
    calculate_fft_mean_std_across_all_noresm,
    logscore_the_samples_for_spatial_spectra_bayesian,
    particle_filter_weighting_bayesian,
)
from climatem.config import *
from climatem.utils import parse_args, update_config_withparse
from climatem import *

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")


# select 16 random samples from the batch
def sample_from_tensor_reproducibly(tensor1, tensor2, num_samples, seed=5):
    if num_samples > tensor1.shape[0]:
        print("num_samples: ", num_samples)
        print("tensor1.shape: ", tensor1.shape)
        raise ValueError("Number of samples cannot exceed the tensor's first dimension.")

    torch.manual_seed(seed)  # Set the random seed
    indices = torch.randperm(tensor1.shape[0])[:num_samples]
    return tensor1[indices], tensor2[indices]


def main(
    experiment_params,
    data_params,
    gt_params,
    train_params,
    model_params,
    optim_params,
    plot_params,
    savar_params,
    rollout_params,
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
        num_input = d * (experiment_params.tau + 1) * (model_params.tau_neigh * 2 + 1)
    else:
        num_input = d * experiment_params.tau * (model_params.tau_neigh * 2 + 1)

    # set the model
    model = LatentTSDCD(
        num_layers=model_params.num_layers,
        num_hidden=model_params.num_hidden,
        num_input=num_input,
        num_output=2,  # This should be parameterized somewhere?
        num_layers_mixing=model_params.num_layers_mixing,
        num_hidden_mixing=model_params.num_hidden_mixing,
        position_embedding_dim=model_params.position_embedding_dim,
        reduce_encoding_pos_dim=model_params.reduce_encoding_pos_dim,
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
        hard_gumbel=model_params.hard_gumbel,
        no_gt=gt_params.no_gt,
        debug_gt_graph=gt_params.debug_gt_graph,
        debug_gt_z=gt_params.debug_gt_z,
        debug_gt_w=gt_params.debug_gt_w,
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

    name = f"var_{data_var_ids_str}_scen_{data_params.train_scenarios[0]}_nlinmix_{model_params.nonlinear_mixing}_nlindyn_{model_params.nonlinear_dynamics}_tau_{experiment_params.tau}_z_{experiment_params.d_z}_futt_{experiment_params.future_timesteps}_ldec_{optim_params.loss_decay_future_timesteps}_lr_{train_params.lr}_bs_{data_params.batch_size}_ormuin_{optim_params.ortho_mu_init}_spmuin_{optim_params.sparsity_mu_init}_spth_{optim_params.sparsity_upper_threshold}_nens_{data_params.num_ensembles}_inst_{model_params.instantaneous}_crpscoef_{optim_params.crps_coeff}_sspcoef_{optim_params.spectral_coeff}_tspcoef_{optim_params.temporal_spectral_coeff}_fracnhiwn_{optim_params.fraction_highest_wavenumbers}_nummix_{model_params.num_hidden_mixing}_numhid_{model_params.num_hidden}_embdim_{model_params.position_embedding_dim}"
    #     name = f"var_{data_var_ids_str}_scenarios_{data_params.train_scenarios[0]}_nonlinear_{model_params.nonlinear_mixing}_tau_{experiment_params.tau}_z_{experiment_params.d_z}_lr_{train_params.lr}_bs_{data_params.batch_size}_spreg_{optim_params.reg_coeff}_ormuinit_{optim_params.ortho_mu_init}_spmuinit_{optim_params.sparsity_mu_init}_spthres_{optim_params.sparsity_upper_threshold}_fixed_{model_params.fixed}_num_ensembles_{data_params.num_ensembles}_instantaneous_{model_params.instantaneous}_crpscoef_{optim_params.crps_coeff}_spcoef_{optim_params.spectral_coeff}_tempspcoef_{optim_params.temporal_spectral_coeff}"
    exp_path = exp_path / name
    if not os.path.exists(exp_path):
        raise ValueError(f"Results path {exp_path} does not exist. Are you using the same parameters?")

    # create path to exp and save hyperparameters
    save_path = exp_path / "rollouts"
    os.makedirs(save_path, exist_ok=True)

    # seed = 1
    save_path = (
        save_path
        / f"bs_{rollout_params.batch_size}_np_{rollout_params.num_particles}_npp_{rollout_params.num_particles_per_particle}_t_{rollout_params.num_timesteps}_sc_{rollout_params.score}_temp_{rollout_params.tempering}"
    )
    os.makedirs(save_path, exist_ok=True)

    model_path = exp_path / "training_results"
    # model_path = Path(experiment_params.exp_path) / f"seb_best_model_seed{seed}"

    # with open(model_path / "params.json", "r") as f:
    #     hp = json.load(f)

    # hp["data_params"]["temp_res"] = "mon"
    # assert hp["data_params"]["seq_len"] == SEQ_LEN_MAPPING[hp["data_params"]["temp_res"]]
    # hp["data_params"].pop('seq_len', None)
    # hp["train_params"].pop('ratio_valid', None)

    y_true_fft_mean, y_true_fft_std = calculate_fft_mean_std_across_all_noresm(datamodule, accelerator)
    print("y_true_fft_mean shape:", y_true_fft_mean.shape)
    print("y_true_fft_std shape:", y_true_fft_std.shape)

    # ================================
    # ORIGINAL DATALOADER

    # train_dataloader = iter(datamodule.train_dataloader(accelerator))
    # print("the length of the dl is: ", len(dl))
    # x, y = next(train_dataloader)
    # print("y before: ", y)
    # print("shape of y before: ", y.shape)
    # print("")

    # if rollout_params.final_30_years_of_ssps:
    #     print("Taking the final 30 years of the SSP data, ~ 2070-2100")
    #     # I'm not sure this does what it says it does also for the trainloader shuffle = True
    #     x, y = next(train_dataloader)
    #     x, y = next(train_dataloader)

    # x = torch.nan_to_num(x)
    # y = torch.nan_to_num(y)
    # y = y[:, 0]
    # z = None
    # print("the value of x[0] is: ", x[0])
    # print("the value of y[0] is: ", y[0])
    # print("shape of x: ", x.shape)
    # print("shape of y: ", y.shape)
    # quit()

    # x = x.to(device)
    # y = y.to(device)

    # # First call with the seed
    # x_samples, y_samples = sample_from_tensor_reproducibly(x, y, rollout_params.batch_size)
    # np.save(
    #     save_path / "forpowerspectra_random1_batch_xs_we_start_with.npy",
    #     x_samples.detach().cpu().numpy(),
    # )

    # ================================

    # NEW DATALOADER
    test_dataset = datamodule._data_val

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=rollout_params.batch_size, shuffle=False)
    test_dataloader = iter(test_dataloader)
    # test_dataloader = iter(datamodule.val_dataloader())
    # print("the length of the dl is: ", len(test_dataloader))
    x, y = next(test_dataloader)
    # print("y before: ", y)
    # print("shape of y before: ", y.shape)
    # print("")

    if rollout_params.final_30_years_of_ssps:
        print("Taking the final 30 years of the SSP data, ~ 2070-2100")
        # I'm not sure this does what it says it does also for the trainloader shuffle = True
        x, y = next(test_dataloader)
        x, y = next(test_dataloader)

    x = torch.nan_to_num(x)
    y = torch.nan_to_num(y)
    y = y[:, 0]
    # z = None
    # print("the value of x[0] is: ", x[0])
    # print("the value of y[0] is: ", y[0])
    # print("================================================")
    # print("the value of x[1] is: ", x[1])
    # print("the value of y[1] is: ", y[1])
    # print("================================================")

    # print("shape of x: ", x.shape)
    # print("shape of y: ", y.shape)
    # quit()

    x = x.to(device)
    y = y.to(device)

    # First call with the seed
    x_samples, y_samples = sample_from_tensor_reproducibly(x, y, rollout_params.batch_size)
    np.save(
        save_path / "forpowerspectra_random1_batch_xs_we_start_with.npy",
        x_samples.detach().cpu().numpy(),
    )

    # ================================

    # Here we load a final model, when we do learn the causal graph. Make sure  it is on GPU:
    state_dict_vae_final = torch.load(model_path / "model.pth", map_location=device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict_vae_final.items()})

    # Move the model to the GPU
    model = model.to(device)
    print("Where is the model?", next(model.parameters()).device)

    with torch.no_grad():
        thresholded_adj = (model.get_adj() > 0.5).type(torch.Tensor)
        model.mask.fix(thresholded_adj)

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
            # validation_targets=val_y_full,  # Pass validation targets
            # validation_inputs=val_x_full,  # Pass validation inputs
        )

    return final_picontrol_particles


if __name__ == "__main__":

    args = parse_args()

    # cwd = Path.cwd()
    # root_path = cwd.parent
    # config_path = root_path / f"configs"
    json_path = CONFIGS_PATH / args.config_path

    with open(json_path, "r") as f:
        params = json.load(f)
    params = update_config_withparse(params, args)

    # get user's scratch directory:
    scratch_path = os.getenv("SCRATCH")
    params["data_params"]["data_dir"] = params["data_params"]["data_dir"].replace("$SCRATCH", scratch_path)
    print("new data path:", params["data_params"]["data_dir"])

    params["exp_params"]["exp_path"] = params["exp_params"]["exp_path"].replace("$SCRATCH", scratch_path)
    print("new exp path:", params["exp_params"]["exp_path"])

    # get directory of project via current file (aka .../climatem/scripts/main_picabu.py)
    params["data_params"]["icosahedral_coordinates_path"] = params["data_params"][
        "icosahedral_coordinates_path"
    ].replace("$CLIMATEMDIR", PROJECT_ROOT.absolute().as_posix())
    print("new icosahedron path:", params["data_params"]["icosahedral_coordinates_path"])

    experiment_params = expParams(**params["exp_params"])
    data_params = dataParams(**params["data_params"])
    gt_params = gtParams(**params["gt_params"])
    train_params = trainParams(**params["train_params"])
    model_params = modelParams(**params["model_params"])
    optim_params = optimParams(**params["optim_params"])
    plot_params = plotParams(**params["plot_params"])
    savar_params = savarParams(**params["savar_params"])
    rollout_params = rolloutParams(**params["rollout_params"])

    # Overwrite arguments if using savar
    if "savar" in data_params.in_var_ids:
        experiment_params.lat = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.lon = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.d_x = int(experiment_params.lat * experiment_params.lon)
        plot_params.savar = True
    else:
        plot_params.savar = False

    final_picontrol_particles = main(
        experiment_params,
        data_params,
        gt_params,
        train_params,
        model_params,
        optim_params,
        plot_params,
        savar_params,
        rollout_params,
    )
