# Here we have a quick main where we are testing data loading with different ensemble members and ideally with different climate models.
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from climatem.config import *
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from climatem.model.metrics import edge_errors, mcc_latent, precision_recall, shd, w_mae
from climatem.model.train_model import TrainingLatent
from climatem.model.tsdcd_latent import LatentTSDCD
from climatem.utils import parse_args, update_config_withparse

torch.set_warn_always(False)

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")

class Bunch:
    """A class that has one variable for each entry of a dictionary."""

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def to_dict(self):
        return self.__dict__

    # def fancy_print(self, prefix=''):
    #     str_list = []
    #     for key, val in self.__dict__.items():
    #         str_list.append(prefix + f"{key} = {val}")
    #     return '\n'.join(str_list)


def main(
    experiment_params, data_params, gt_params, train_params, model_params, optim_params, plot_params, savar_params
):
    """
    :param hp: object containing hyperparameter values
    """
    t0 = time.time()

    # Control as much randomness as possible
    torch.manual_seed(experiment_params.random_seed)
    np.random.seed(experiment_params.random_seed)

    if experiment_params.gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # Create folder
    # args.exp_path = os.path.join(args.exp_path, f"exp{args.exp_id}")
    # if not os.path.exists(args.exp_path):
    #     os.makedirs(args.exp_path)

    # generate data and split train/test
    device = torch.device("cuda" if (torch.cuda.is_available() and experiment_params.gpu) else "cpu")

    if data_params.data_format == "hdf5":
        print("IS HDF5")
        return
    else:
        datamodule = CausalClimateDataModule(
            tau=experiment_params.tau,
            future_timesteps=experiment_params.future_timesteps,
            num_months_aggregated=data_params.num_months_aggregated,
            train_val_interval_length=data_params.train_val_interval_length,
            in_var_ids=data_params.in_var_ids,
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
            lat=experiment_params.lat,
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

    # train_dataloader = iter(datamodule.train_dataloader())
    # val_dataloader = iter(datamodule.val_dataloader())

    # WE SHOULD REMOVE THIS, and initialize with params
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
        obs_to_latent_mask=datamodule.obs_to_latent_mask,
        # gt_w=data_loader.gt_w,
        # gt_graph=data_loader.gt_graph,
        tied_w=model_params.tied_w,
        # also
        fixed=model_params.fixed,
        fixed_output_fraction=model_params.fixed_output_fraction,
    )

    # Make folder to save run results
    exp_path = Path(experiment_params.exp_path)
    os.makedirs(exp_path, exist_ok=True)
    data_var_ids_str = (
        str(data_params.in_var_ids)[1:-1]
        .translate({ord("'"): None})
        .translate({ord(","): None})
        .translate({ord(" "): None})
    )
    name = f"var_{data_var_ids_str}_scen_{data_params.train_scenarios[0]}_nlinmix_{model_params.nonlinear_mixing}_nlindyn_{model_params.nonlinear_dynamics}_tau_{experiment_params.tau}_z_{experiment_params.d_z}_futt_{experiment_params.future_timesteps}_ldec_{optim_params.loss_decay_future_timesteps}_lr_{train_params.lr}_bs_{data_params.batch_size}_ormuin_{optim_params.ortho_mu_init}_spmuin_{optim_params.sparsity_mu_init}_spth_{optim_params.sparsity_upper_threshold}_nens_{data_params.num_ensembles}_inst_{model_params.instantaneous}_crpscoef_{optim_params.crps_coeff}_sspcoef_{optim_params.spectral_coeff}_tspcoef_{optim_params.temporal_spectral_coeff}_fracnhiwn_{optim_params.fraction_highest_wavenumbers}_nummix_{model_params.num_hidden_mixing}_numhid_{model_params.num_hidden}_embdim_{model_params.position_embedding_dim}"
    exp_path = exp_path / name
    os.makedirs(exp_path, exist_ok=True)

    # create path to exp and save hyperparameters
    save_path = exp_path / "training_results"
    os.makedirs(save_path, exist_ok=True)
    plots_path = exp_path / "plots"
    os.makedirs(plots_path, exist_ok=True)
    # Here could maybe implement a "save()" function inside each class
    hp = {}
    hp["exp_params"] = experiment_params.__dict__
    hp["data_params"] = data_params.__dict__
    hp["gt_params"] = gt_params.__dict__
    hp["train_params"] = train_params.__dict__
    hp["model_params"] = model_params.__dict__
    hp["optim_params"] = optim_params.__dict__
    with open(exp_path / "params.json", "w") as file:
        json.dump(hp, file, indent=4)

    # # load the best metrics
    # with open(os.path.join(hp.data_path, "best_metrics.json"), 'r') as f:
    #     best_metrics = json.load(f)
    best_metrics = {"recons": 0, "kl": 0, "mcc": 0, "elbo": 0}

    # train, always with the latent version
    trainer = TrainingLatent(
        model,
        datamodule,
        experiment_params,
        gt_params,
        model_params,
        train_params,
        optim_params,
        plot_params,
        save_path,
        plots_path,
        best_metrics,
        d,
        accelerator,
        wandbname=name,
        profiler=False,
    )

    # where is the model at this point?
    print("Where is my model?", next(trainer.model.parameters()).device)

    valid_loss = trainer.train_with_QPM()
    print("[DEBUG] Training finished, valid_loss:", valid_loss)

    # save final results, (MSE)
    metrics = {"shd": 0.0, "precision": 0.0, "recall": 0.0, "train_mse": 0.0, "val_mse": 0.0, "mcc": 0.0}
    # if we have the GT, also compute (SHD, Pr, Re, MCC)
    if not gt_params.no_gt:
        # Here can remove this ---
        if model_params.instantaneous:
            gt_graph = trainer.gt_dag
        else:
            gt_graph = trainer.gt_dag[:-1]  # remove the graph G_t

        learned_graph = trainer.model.get_adj().detach().numpy().reshape(gt_graph.shape[0], gt_graph.shape[1], -1)

        score, cc_program_perm, assignments, z, z_hat, _ = mcc_latent(trainer.model, trainer.data)
        permutation = np.zeros((gt_graph.shape[1], gt_graph.shape[1]))
        permutation[np.arange(gt_graph.shape[1]), assignments[1]] = 1
        gt_graph = permutation.T @ gt_graph @ permutation

        metrics["mcc"] = score
        metrics["w_mse"] = w_mae(
            trainer.model.autoencoder.get_w_decoder().detach().numpy()[:, :, assignments[1]], datamodule.gt_w
        )
        metrics["shd"] = shd(learned_graph, gt_graph)
        metrics["precision"], metrics["recall"] = precision_recall(learned_graph, gt_graph)
        errors = edge_errors(learned_graph, gt_graph)
        metrics["tp"] = errors["tp"]
        metrics["fp"] = errors["fp"]
        metrics["tn"] = errors["tn"]
        metrics["fn"] = errors["fn"]
        metrics["n_edge_gt_graph"] = np.sum(gt_graph)
        metrics["n_edge_learned_graph"] = np.sum(learned_graph)
        metrics["execution_time"] = time.time() - t0

        for key, val in valid_loss.items():
            metrics[key] = val

    # assert that trainer.model is in eval mode
    if trainer.model.training:
        print("Model is in train mode")
    else:
        print("Model is in eval mode")

    # NOTE: just dummies here for now
    train_mse, train_smape, val_mse, val_smape = 10.0, 10.0, 10.0, 10.0

    # save the metrics
    metrics["train_mse"] = train_mse
    metrics["train_smape"] = train_smape
    metrics["val_mse"] = val_mse
    metrics["val_smape"] = val_smape

    # save the metrics
    with open(os.path.join(experiment_params.exp_path, "results.json"), "w") as file:
        json.dump(metrics, file, indent=4)

    # finally, save the model
    torch.save(trainer.model.state_dict(), os.path.join(experiment_params.exp_path, "model.pth"))


def assert_args(
    experiment_params,
    data_params,
    gt_params,
    optim_params,
):
    """Raise errors or warnings if some args should not take some combination of values."""
    # raise errors if some args should not take some combination of values
    if gt_params.no_gt and (gt_params.debug_gt_graph or gt_params.debug_gt_z or gt_params.debug_gt_w):
        raise ValueError("Since no_gt==True, all other args should not use ground-truth values")

    if experiment_params.latent and (
        experiment_params.d_z is None
        or experiment_params.d_x is None
        or experiment_params.d_z <= 0
        or experiment_params.d_x <= 0
    ):
        raise ValueError("When using latent model, you need to define d_z and d_x with integer values greater than 0")

    # string input with limited possible values
    supported_dataformat = ["numpy", "hdf5"]
    if data_params.data_format not in supported_dataformat:
        raise ValueError(
            f"This file format ({data_params.data_format}) is not \
                         supported. Supported types are: {supported_dataformat}"
        )
    supported_optimizer = ["sgd", "rmsprop"]
    if optim_params.optimizer not in supported_optimizer:
        raise ValueError(
            f"This optimizer type ({optim_params.optimizer}) is not \
                         supported. Supported types are: {supported_optimizer}"
        )

    # warnings, strange choice of args combination
    if not experiment_params.latent and gt_params.debug_gt_z:
        warnings.warn("Are you sure you want to use gt_z even if you don't have latents")
    if experiment_params.latent and (experiment_params.d_z > experiment_params.d_x):
        warnings.warn("Are you sure you want to have a higher dimension for d_z than d_x")

    return


if __name__ == "__main__":

    args = parse_args()
    
    root_path = Path(__file__).resolve().parent.parent
    config_path = root_path / f"configs"
    json_path = config_path / args.config_path
    
    with open(json_path, "r") as f:
        params = json.load(f)
    config_obj_list = update_config_withparse(params, args)

    # get user's scratch directory on Mila cluster:
    scratch_path = os.getenv("SCRATCH")
    params["data_params"]["data_dir"] = params["data_params"]["data_dir"].replace("$SCRATCH", scratch_path)
    print ("new data path:", params["data_params"]["data_dir"])

    params["exp_params"]["exp_path"] = params["exp_params"]["exp_path"].replace("$SCRATCH", scratch_path)
    print ("new exp path:", params["exp_params"]["exp_path"])

    # get directory of project via current file (aka .../climatem/scripts/main_picabu.py)
    params["data_params"]["icosahedral_coordinates_path"] = params["data_params"]["icosahedral_coordinates_path"].replace("$CLIMATEMDIR", root_path.absolute().as_posix())
    print ("new icosahedron path:", params["data_params"]["icosahedral_coordinates_path"])

    experiment_params = expParams(**params["exp_params"])
    data_params = dataParams(**params["data_params"])
    gt_params = gtParams(**params["gt_params"])
    train_params = trainParams(**params["train_params"])
    model_params = modelParams(**params["model_params"])
    optim_params = optimParams(**params["optim_params"])
    plot_params = plotParams(**params["plot_params"])
    savar_params = savarParams(**params["savar_params"])

    #Overwrite arguments if using savar
    if "savar" in data_params.in_var_ids:
        experiment_params.lat = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.lon = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.d_x = int(experiment_params.lat * experiment_params.lon)
        plot_params.savar = True
    else:
        plot_params.savar = False

    assert_args(
        experiment_params,
        data_params,
        gt_params,
        optim_params,
    )

    main(experiment_params, data_params, gt_params, train_params, model_params, optim_params, plot_params, savar_params)

